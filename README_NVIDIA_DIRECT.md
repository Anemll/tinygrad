# Tinygrad NVIDIA Direct Runtime Testing

This document explains how to test and use tinygrad's NVIDIA direct runtime, which allows running GPU computations without NVIDIA drivers by directly accessing the GPU hardware through PCI/MMIO.

## Overview

Tinygrad includes a sophisticated NVIDIA direct implementation (`anemll-tinygrad/runtime/support/nv/`) that bypasses the need for NVIDIA drivers by:

1. **Direct Hardware Access**: Using PCI/MMIO to directly communicate with NVIDIA GPUs
2. **Firmware Management**: Loading and managing GPU firmware (GSP, FLCN)
3. **Memory Management**: Direct VRAM allocation and management
4. **Command Execution**: Direct GPU command submission and execution

## Prerequisites

### Hardware Requirements
- NVIDIA GPU (tested with Ampere, Turing, and Hopper architectures)
- PCIe connection to the GPU
- Sufficient system memory for GPU operations

### Software Requirements
- Python 3.8+
- Tinygrad installed from source
- Root/sudo access (for PCI device access)

### System Setup
```bash
# Clone tinygrad
git clone https://github.com/tinygrad/tinygrad.git
cd tinygrad

# Install dependencies
pip install -e .

# Set environment variables for debugging
export NV_DEBUG=1
export DEBUG=2
```

## Testing the NVIDIA Direct Runtime

### 1. Basic Component Test

Run the simple component test to verify the core NV runtime components:

```bash
cd ~/SourceRelease/GITHUB/ML_playground/anemll-tinygrad
python3  demo_rtx_5070_sudo.py
```

This test will:
- Import all NV-related modules
- Test device instantiation
- Check device capabilities
- Test memory allocation
- Verify device cleanup

### 2. Full Runtime Test

Run the comprehensive test to verify full tensor operations:

```bash
cd ~/SourceRelease/GITHUB/ML_playground/anemll-tinygrad
python3 demo_rtx_5070_sudo.py
```

This test will:
- Test device availability
- Perform basic tensor operations
- Test memory transfers (CPU ↔ GPU)
- Execute compute operations
- Test device synchronization
- Verify error handling
- Test advanced features

### 3. Manual Testing

You can also test manually in Python:


```

## Understanding the Architecture

### Core Components

1. **NVDev** (`anemll-tinygrad/runtime/support/nv/nvdev.py`)
   - Main device class for NVIDIA GPUs
   - Handles MMIO access and register management
   - Manages GPU firmware and initialization

2. **NV_FLCN/NV_GSP** (`anemll-tinygrad/runtime/support/nv/ip.py`)
   - Falcon and GSP (GPU System Processor) management
   - Handles firmware loading and execution
   - Manages GPU communication protocols

3. **NVDevice** (`anemll-tinygrad/runtime/ops_nv.py`)
   - High-level device interface for tinygrad
   - Manages memory allocation and command queues
   - Handles tensor operations and compute execution

### Key Features

- **Driverless Operation**: No NVIDIA drivers required
- **Direct Hardware Access**: PCI/MMIO-based communication
- **Firmware Management**: Automatic GPU firmware loading
- **Memory Management**: Direct VRAM allocation
- **Command Execution**: Direct GPU command submission

## Troubleshooting

### Common Issues

1. **Permission Denied**
   ```
   Error: Permission denied accessing PCI device
   ```
   **Solution**: Run with sudo or ensure proper PCI device permissions

2. **No GPU Found**
   ```
   Error: No NVIDIA GPU detected
   ```
   **Solution**: Verify GPU is properly connected and detected by system

3. **Firmware Loading Failed**
   ```
   Error: Failed to load GPU firmware
   ```
   **Solution**: Check if GPU firmware files are available and accessible

4. **Memory Allocation Failed**
   ```
   Error: Failed to allocate GPU memory
   ```
   **Solution**: Check available VRAM and system memory

5. **BAR Mapping Errors (macOS/Linux)**
   ```
   KeyError: 3
   Error: BAR3 does not exist
   ```
   **Solution**: This was a platform-specific BAR mapping issue that has been fixed. The code now automatically uses the correct BAR indices for each platform:
   - Linux: BAR1 (VRAM), BAR3 (instruction memory)
   - macOS: BAR2 (VRAM), BAR4 (instruction memory)

6. **eGPU Driver Access Denied (macOS)**
   ```
   Error: Cannot connect to EGPUMapperDriver
   ```
   **Solution**: Ensure your application is properly signed with DriverKit entitlements (see Code Signing section)

### Debug Information

Enable detailed debugging by setting environment variables:

```bash
export NV_DEBUG=4      # Maximum NVIDIA debug output
export DEBUG=3         # Maximum general debug output
```

### Log Analysis

The debug output will show:
- GPU detection and initialization
- Firmware loading progress
- Memory allocation details
- Command execution status
- Error details and stack traces

## Performance Considerations

### Memory Management
- The direct runtime uses huge pages for large allocations
- Memory is allocated directly in GPU VRAM
- CPU-GPU transfers are optimized for performance

### Compute Performance
- Direct hardware access can provide better performance than driver-based approaches
- Command submission is optimized for minimal latency
- Memory bandwidth is maximized through direct access

### Limitations
- Requires root/sudo access for PCI device access
- Limited to supported GPU architectures
- May not support all GPU features available through drivers

## Platform-Specific Implementation

### BAR (Base Address Register) Mapping

NVIDIA GPUs use different BAR configurations on different platforms due to 64-bit BAR enumeration differences:

#### Linux (Traditional PCI Enumeration)
```
BAR 0: 64MB MMIO registers (32-bit)
BAR 1: 256MB VRAM (64-bit)
BAR 2: Upper 32-bits of BAR1 (not accessible)
BAR 3: 32MB instruction memory (64-bit)
BAR 4: Upper 32-bits of BAR3 (not accessible)
BAR 5: I/O ports
```

#### macOS (DriverKit Memory Index Mapping)
```
Memory Index 0 (BAR 0): 64MB MMIO registers (32-bit)
Memory Index 1 (BAR 2): 256MB VRAM (64-bit, skips BAR1)
Memory Index 2 (BAR 4): 32MB instruction memory (64-bit, skips BAR3)
BAR 5: I/O ports
```

The code automatically handles these differences at multiple levels:

#### 1. Physical Address Mapping (GSP System Info)
```python
# In ip.py - Platform-specific physical address mapping
if OSX:
    gpu_phys_vram_addr = self.nvdev.bars[2][0]  # macOS: BAR2 = VRAM
    gpu_phys_inst_addr = self.nvdev.bars[4][0]  # macOS: BAR4 = instruction memory
else:
    gpu_phys_vram_addr = self.nvdev.bars[1][0]  # Linux: BAR1 = VRAM
    gpu_phys_inst_addr = self.nvdev.bars[3][0]  # Linux: BAR3 = instruction memory
```

#### 2. Memory Mapping (map_bar calls)
```python
# In ops_nv.py - Platform-specific BAR mapping calls
vram_bar = 2 if OSX else 1  # macOS uses BAR2, Linux uses BAR1
self.dev_impl = NVDev(pcibus, self.pci_dev.map_bar(0, fmt='I'), self.pci_dev.map_bar(vram_bar), ...)
self.p2p_base_addr = self.pci_dev.bar_info[vram_bar][0]
```

#### 3. Low-level Implementation
The `map_bar()` method itself has platform-specific implementations:
- **macOS**: Uses eGPU driver via DriverKit (`self.egpu_device.map_bar()`)
- **Linux**: Uses direct PCI memory mapping (`mmap()` on PCI device files)

#### Design Philosophy
This multi-layer approach ensures:
- **Clean separation**: Platform differences handled at appropriate abstraction levels
- **Maintainability**: BAR index translation at call sites, not buried in low-level code
- **Clarity**: Explicit platform-specific choices rather than hidden translations
- **Testability**: Each layer can be tested independently

### eGPU Support (macOS)

The runtime includes specialized support for external GPUs on macOS through DriverKit:

#### eGPU Components
1. **EGPUDev** (`anemll-tinygrad/runtime/support/nv/egpudev.py`)
   - macOS-specific eGPU device implementation
   - Uses libdkmmio.dylib for hardware access
   - Handles DriverKit memory mapping

2. **EGPUDevice** (`anemll-tinygrad/runtime/ops_egpu.py`)
   - High-level eGPU interface for tinygrad
   - Automatic architecture detection
   - Platform-optimized memory management

#### eGPU Usage
```python
from tinygrad.runtime.ops_egpu import EGPUDevice

# Create eGPU device (macOS only)
device = EGPUDevice("egpu:0")

# Test operations
device.synchronize()
device.finalize()
```

#### eGPU Requirements
- macOS with DriverKit support
- External GPU connected via Thunderbolt/PCIe
- libdkmmio.dylib library
- Proper entitlements for third-party driver access

#### Testing eGPU Functionality
```bash
# Test basic eGPU connection
python test_gsp_simple.py

# Test GSP firmware loading
python test_gsp_load.py

# Test third-party driver access (requires signed binary)
./test_third_party_access
```

#### Code Signing for eGPU Access
For third-party DriverKit access, applications must be signed with appropriate entitlements:

```xml
<!-- driver_client.entitlements -->
<?xml version="1.0" encoding="UTF-8"?>
<!DOCTYPE plist PUBLIC "-//Apple//DTD PLIST 1.0//EN" "http://www.apple.com/DTDs/PropertyList-1.0.dtd">
<plist version="1.0">
<dict>
    <key>com.apple.developer.driverkit</key>
    <true/>
    <key>com.apple.developer.driverkit.allow-third-party-userclients</key>
    <true/>
    <key>com.apple.developer.driverkit.userclient-access</key>
    <array>
        <string>*</string>  <!-- Wildcard allows access to any driver -->
    </array>
    <key>com.apple.security.app-sandbox</key>
    <false/>
</dict>
</plist>
```

Sign your application:
```bash
codesign --entitlements driver_client.entitlements --force --sign "Your Developer ID" your_app
```

## Advanced Usage

### Custom GPU Operations

```python
from tinygrad.runtime.ops_nv import NVDevice

# Create custom device
device = NVDevice("")

# Allocate memory
buffer = device.allocator.alloc(1024)

# Submit custom commands
# (Advanced usage requires understanding of GPU command formats)

# Cleanup
device.finalize()
```

### Firmware Customization

The runtime automatically downloads and manages GPU firmware. For custom firmware:

1. Place firmware files in the expected location
2. Modify firmware loading code in `nvdev.py`
3. Ensure firmware compatibility with your GPU

## Contributing

When contributing to the NVIDIA direct runtime:

1. Test with multiple GPU architectures
2. Verify error handling and recovery
3. Ensure proper cleanup and resource management
4. Document any new features or changes
5. Update tests to cover new functionality

## References

- [Tinygrad Documentation](https://github.com/tinygrad/tinygrad)
- [NVIDIA GPU Architecture Documentation](https://developer.nvidia.com/gpu-architecture)
- [PCI/MMIO Programming](https://en.wikipedia.org/wiki/Memory-mapped_I/O)
- [GPU Firmware Management](https://github.com/NVIDIA/open-gpu-kernel-modules) 