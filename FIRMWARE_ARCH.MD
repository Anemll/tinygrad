# NVIDIA GSP Firmware Architecture - Implemented

## Overview

This document describes the actual implementation of GSP (GPU System Processor) firmware loading in the tinygrad eGPU system on macOS. The GSP is a dedicated RISC-V processor on modern NVIDIA GPUs that handles system management tasks.

## Architecture

### Implementation Stack

```
┌─────────────────────────────────────────────────────────────────┐
│                    Python Application (tinygrad)                │
│  ┌─────────────────────────────────────────────────────────────┤
│  │ egpudev.py                                                  │
│  │ ├─ allocate_dma_buffer() -> dict                           │
│  │ ├─ get_memory_type_for_handle() -> int                     │
│  │ └─ Uses libegpu_pcidevice.dylib via ctypes                 │
│  └─────────────────────────────────────────────────────────────┤
├─────────────────────────────────────────────────────────────────┤
│                    System Layer (system.py)                     │
│  ┌─────────────────────────────────────────────────────────────┤
│  │ System.alloc_sysmem() - DMA buffer allocation              │
│  │ ├─ macOS: Uses EGPUDev.allocate_dma_buffer()               │
│  │ ├─ Maps to user space via IOConnectMapMemory()             │
│  │ └─ Returns (virtual_addr, [physical_addresses])            │
│  └─────────────────────────────────────────────────────────────┤
└─────────────────────────────────────────────────────────────────┘
```

## Key Components

### 1. DMA Buffer Allocation (system.py:80)

The core allocation function that provides DMA-coherent memory for GPU firmware loading:

```python
def alloc_sysmem(self, size:int, vaddr:int=0, contiguous:bool=False, 
                 data:bytes|None=None, direction:int=DMA_DIRECTION_CPU_TO_GPU):
    if OSX:
        # Allocate DMA buffer using DriverKit
        buffer_info = self._egpu_device.allocate_dma_buffer(size, direction)
        
        # Map to user space with IOConnectMapMemory
        connection = self._egpu_device.get_connection()
        memory_type = self._egpu_device.get_memory_type_for_handle(buffer_info['handle'])
        
        # IOKit mapping for user space access
        mapped_addr = ctypes.c_uint64(0)
        IOKit.IOConnectMapMemory(connection, memory_type, task, 
                                ctypes.byref(mapped_addr), ...)
        
        # Copy data if provided
        if data is not None:
            ctypes.memmove(mapped_addr.value, data, len(data))
        
        return mapped_addr.value, physical_pages
```

### 2. GSP Firmware Loading Components (ip.py)

Multiple firmware components are loaded via DMA:

| Component | File:Line | Size | Purpose |
|-----------|-----------|------|---------|
| FRTS Firmware | ip.py:139 | Variable | Falcon Real-Time Scheduler |
| FMC Booter | ip.py:272 | Variable | Falcon Microcode bootloader |
| RPC Queues | ip.py:329 | ~528KB | Command/status communication |
| Log Buffer | ip.py:349 | 2MB | Debug logging from GPU |
| LibOS Args | ip.py:350 | 4KB | LibOS initialization |
| GSP Booter | ip.py:395 | Variable | GSP bootloader binary |

### 3. DMA Direction Support

Three transfer directions are implemented:

```python
# DMA Direction constants (system.py:12)
DMA_DIRECTION_CPU_TO_GPU = 0      # Default: firmware, commands
DMA_DIRECTION_GPU_TO_CPU = 1      # Log buffers, results  
DMA_DIRECTION_BIDIRECTIONAL = 2   # RPC queues, shared memory
```

## Memory Architecture

### DMA Buffer Flow

```
User Space (Python)
      │ firmware_data: bytes
      ▼ allocate_dma_buffer()
 DriverKit DEXT
  ┌─────────────────────────────────────────┐
  │ IOBufferMemoryDescriptor                 │ ← DMA-coherent
  │ ├─ Physical address for GPU             │
  │ └─ Memory type ID for mapping           │
  └─────────────────────────────────────────┘
      │ IOConnectMapMemory()
      ▼
  ┌─────────────────────────────────────────┐
  │ User Space Virtual Address               │ ← Direct access
  │ ├─ Read/write via ctypes                │
  │ └─ Standard memory operations           │
  └─────────────────────────────────────────┘
      │ Physical address to GPU
      ▼
GPU Hardware Registers
  ┌─────────────────────────────────────────┐
  │ MAILBOX0 = libos_args_physical_addr     │
  │ MAILBOX1 = firmware_base_physical_addr  │
  │ FALCON_CPUCTL = START_CPU               │
  └─────────────────────────────────────────┘
```

## Implementation Tables

### DMA Allocations Summary

| Usage | Direction | Size | Has Data | Container | Status |
|-------|-----------|------|----------|-----------|---------|
| Boot Structures | CPU→GPU | Variable | ✅ | 🔵 Yes | ✅ Working |
| FRTS Firmware | CPU→GPU | Variable | ✅ | ❌ No | ✅ Working |
| FMC Booter | CPU→GPU | Variable | ✅ | ❌ No | ✅ Working |
| RPC Queues | 🔄 BIDIRECTIONAL | ~528KB | ❌ | 🔵 Yes | ✅ Working |
| Log Buffer | GPU→CPU | 2MB | ❌ | ❌ No | ✅ Working |
| LibOS Args | CPU→GPU | 4KB | ❌ | 🔵 Yes | ✅ Working |
| GSP Booter | CPU→GPU | Variable | ✅ | ❌ No | ✅ Working |
| Method Buffer | CPU→GPU | 20KB | ❌ | ❌ No | ✅ Working |

### Physical Address Dependencies

Container allocations embed physical addresses of child allocations:

```
GPU Mailbox Registers
└── LibOS Args
    ├── Log Buffer 
    └── RM Args
        └── RPC Queues

Boot Structures (WPR Metadata)
├── GSP Booter
├── GSP Signature  
└── GSP Radix (Linux only)
```

## Key Features Implemented

### ✅ Complete Features

1. **DMA Management**: IOBufferMemoryDescriptor allocation via DriverKit
2. **User Space Mapping**: IOConnectMapMemory for direct memory access
3. **Direction Support**: CPU→GPU, GPU→CPU, and bidirectional transfers
4. **Resource Tracking**: Handle-based cleanup with proper teardown
5. **Data Initialization**: Support for pre-loaded firmware data
6. **Container/Child**: Physical address embedding for dependencies

### 🔧 Implementation Details

- **Memory Type IDs**: DriverKit returns memory type ID (≥1000) not virtual address
- **IOKit Mapping**: Required for user space access to DMA buffers
- **Physical Addresses**: Real hardware addresses for GPU register programming
- **Cleanup Critical**: Must unmap and destroy buffers to prevent leaks

## Usage Example

```python
# Allocate DMA buffer for firmware
firmware_va, firmware_paddrs = System.alloc_sysmem(
    len(firmware_data), 
    contiguous=True, 
    data=firmware_data,
    direction=System.DMA_DIRECTION_CPU_TO_GPU
)

# Allocate RPC queues (bidirectional)
queue_va, queue_paddrs = System.alloc_sysmem(
    queue_size,
    contiguous=False,
    direction=System.DMA_DIRECTION_BIDIRECTIONAL  
)

# Program GPU registers with physical addresses
gpu.MAILBOX0.write(lo32(firmware_paddrs[0]))
gpu.MAILBOX1.write(hi32(firmware_paddrs[0]))
```

## Architecture Benefits

- **Hardware Direct**: No intermediate layers between driver and GPU
- **Memory Safety**: DriverKit manages all DMA operations securely
- **Performance**: Minimal overhead with direct memory mapping
- **Compatibility**: Drop-in replacement for Linux System.alloc_sysmem()
- **Maintainability**: Clean separation of platform-specific code

This implementation provides complete GSP firmware loading support on macOS through native DriverKit APIs while maintaining compatibility with the existing tinygrad codebase.

## Platform Differences

### nvdev.py Preservation

The `nvdev.py` file remains largely unchanged between Linux and macOS implementations. The core NVDev class handles:

- Register definitions and access patterns
- Firmware extraction and preparation
- GSP initialization sequences
- Memory manager integration

### 64-bit BAR Investigation

Currently investigating differences in 64-bit BAR handling between platforms:

#### Linux BAR Layout
```
BAR0: MMIO registers (32-bit)
BAR1: VRAM (64-bit lower)
BAR2: VRAM (64-bit upper) - combined with BAR1
BAR3: Instruction memory (64-bit lower)
BAR4: Instruction memory (64-bit upper) - combined with BAR3
BAR5: I/O ports
```

#### macOS BAR Layout (via DriverKit)
```
BAR0: MMIO registers (32-bit)
BAR2: VRAM (64-bit) - skips upper 32-bits
BAR4: Instruction memory (64-bit) - skips upper 32-bits
```

**Key Difference**: macOS DriverKit skips the upper 32-bits of 64-bit BARs in its memory index, affecting:
- Firmware loading addresses
- VRAM access patterns
- Memory-mapped register calculations

This difference is handled in `PCIDevice.a()` method (system.py:564) with platform-specific BAR remapping.