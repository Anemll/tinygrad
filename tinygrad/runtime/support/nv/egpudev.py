"""
eGPU Device Support for tinygrad on macOS

This module provides NVIDIA GPU support for tinygrad on macOS using the EGPUMapperDriver.
Updated to use libegpu_pcidevice.dylib API with DMA buffer allocation.

Key features:
- Direct PCI device access via DriverKit driver
- MMIO register access for GPU control
- DMA buffer allocation and management
- Memory management operations
- Compatible with tinygrad's NVDev interface
"""

from __future__ import annotations
import ctypes, time, functools, re, gzip, struct, os
from tinygrad.helpers import getenv, DEBUG, fetch, getbits, to_mv
from tinygrad.runtime.support.hcq import MMIOInterface
from tinygrad.runtime.support.memory import TLSFAllocator, MemoryManager
from tinygrad.runtime.support.nv.ip import NV_FLCN, NV_FLCN_COT, NV_GSP
from tinygrad.runtime.support.nv.nvdev import NVReg, NVPageTableEntry, NVMemoryManager
from tinygrad.runtime.support.system import System
from tinygrad.device import Allocator
from ctypes import c_uint32, c_uint64, c_void_p, byref, POINTER, c_char_p, c_int, Structure, CDLL, c_uint16, c_uint8

NV_DEBUG = getenv("NV_DEBUG", 0)

# Load libegpu_pcidevice.dylib (new unified API)
LIBEGPU_PATH = os.path.expanduser(os.environ.get('LIBEGPU_PATH', '~/SourceRelease/GITHUB/eGPU/eGPU/EGPUMapperDriver/libegpu_pcidevice.dylib'))

try:
    if NV_DEBUG >= 1:
        print(f"Attempting to load libegpu_pcidevice.dylib from: {LIBEGPU_PATH}")
    libegpu = CDLL(LIBEGPU_PATH)
    if NV_DEBUG >= 1:
        print(f"Successfully loaded libegpu_pcidevice.dylib")
except OSError as e:
    print(f"Failed to load libegpu_pcidevice.dylib from {LIBEGPU_PATH}: {e}")
    # Try alternative locations
    alt_paths = [
        "/usr/local/lib/libegpu_pcidevice.dylib",
        os.path.expanduser("~/lib/libegpu_pcidevice.dylib"),
        os.path.join(os.path.dirname(__file__), "libegpu_pcidevice.dylib")
    ]
    libegpu = None
    for alt_path in alt_paths:
        if os.path.exists(alt_path):
            try:
                if NV_DEBUG >= 1:
                    print(f"Trying alternative path: {alt_path}")
                libegpu = CDLL(alt_path)
                print(f"Successfully loaded libegpu_pcidevice.dylib from: {alt_path}")
                break
            except OSError:
                continue

# Define libegpu_pcidevice structures and function prototypes
class egpu_device_info_t(Structure):
    _fields_ = [
        ("vendor_id", c_uint16),
        ("device_id", c_uint16),
        ("bar_count", c_uint32),
        ("bar_sizes", c_uint64 * 6),
        ("bar_bases", c_uint64 * 6)
    ]

# Define function prototypes for libegpu_pcidevice
if libegpu:
    # EGPUPCIDevice* egpu_device_create(int device_id, int* bars, int bar_count)
    libegpu.egpu_device_create.restype = c_void_p
    libegpu.egpu_device_create.argtypes = [c_int, POINTER(c_int), c_int]
    
    # void egpu_device_destroy(EGPUPCIDevice* device)
    libegpu.egpu_device_destroy.restype = None
    libegpu.egpu_device_destroy.argtypes = [c_void_p]
    
    # const DeviceInfo* egpu_device_get_info(EGPUPCIDevice* device)
    libegpu.egpu_device_get_info.restype = POINTER(egpu_device_info_t)
    libegpu.egpu_device_get_info.argtypes = [c_void_p]
    
    # MMIOInterface* egpu_device_map_bar(EGPUPCIDevice* device, int bar)
    libegpu.egpu_device_map_bar.restype = c_void_p
    libegpu.egpu_device_map_bar.argtypes = [c_void_p, c_int]
    
    # uint32_t egpu_mmio_read32(MMIOInterface* mmio, uint64_t offset)
    libegpu.egpu_mmio_read32.restype = c_uint32
    libegpu.egpu_mmio_read32.argtypes = [c_void_p, c_uint64]
    
    # void egpu_mmio_write32(MMIOInterface* mmio, uint64_t offset, uint32_t value)
    libegpu.egpu_mmio_write32.restype = None
    libegpu.egpu_mmio_write32.argtypes = [c_void_p, c_uint64, c_uint32]
    
    # uint16_t egpu_mmio_read16(MMIOInterface* mmio, uint64_t offset)
    libegpu.egpu_mmio_read16.restype = c_uint16
    libegpu.egpu_mmio_read16.argtypes = [c_void_p, c_uint64]
    
    # void egpu_mmio_write16(MMIOInterface* mmio, uint64_t offset, uint16_t value)
    libegpu.egpu_mmio_write16.restype = None
    libegpu.egpu_mmio_write16.argtypes = [c_void_p, c_uint64, c_uint16]
    
    # uint8_t egpu_mmio_read8(MMIOInterface* mmio, uint64_t offset)
    libegpu.egpu_mmio_read8.restype = c_uint8
    libegpu.egpu_mmio_read8.argtypes = [c_void_p, c_uint64]
    
    # void egpu_mmio_write8(MMIOInterface* mmio, uint64_t offset, uint8_t value)
    libegpu.egpu_mmio_write8.restype = None
    libegpu.egpu_mmio_write8.argtypes = [c_void_p, c_uint64, c_uint8]
    
    # void egpu_mmio_memory_barrier(MMIOInterface* mmio)
    libegpu.egpu_mmio_memory_barrier.restype = None
    libegpu.egpu_mmio_memory_barrier.argtypes = [c_void_p]
    
    # DMA Buffer Allocation Functions
    # uint64_t egpu_device_allocate_dma_buffer(EGPUPCIDevice* device, uint64_t size, uint32_t direction, uint64_t* physical_addr, uint64_t* virtual_addr)
    libegpu.egpu_device_allocate_dma_buffer.restype = c_uint64
    libegpu.egpu_device_allocate_dma_buffer.argtypes = [c_void_p, c_uint64, c_uint32, POINTER(c_uint64), POINTER(c_uint64)]
    
    # int egpu_device_destroy_dma_buffer(EGPUPCIDevice* device, uint64_t handle)
    libegpu.egpu_device_destroy_dma_buffer.restype = c_int
    libegpu.egpu_device_destroy_dma_buffer.argtypes = [c_void_p, c_uint64]
    
    # uint32_t egpu_device_get_memory_type_for_handle(EGPUPCIDevice* device, uint64_t handle)
    libegpu.egpu_device_get_memory_type_for_handle.restype = c_uint32
    libegpu.egpu_device_get_memory_type_for_handle.argtypes = [c_void_p, c_uint64]
    
    # int egpu_device_copy_dma_buffers(EGPUPCIDevice* device, uint64_t src_handle, uint64_t dst_handle, uint64_t offset, uint64_t size)
    libegpu.egpu_device_copy_dma_buffers.restype = c_int
    libegpu.egpu_device_copy_dma_buffers.argtypes = [c_void_p, c_uint64, c_uint64, c_uint64, c_uint64]
    
    # uint32_t egpu_device_get_connection(EGPUPCIDevice* device)
    libegpu.egpu_device_get_connection.restype = c_uint32
    libegpu.egpu_device_get_connection.argtypes = [c_void_p]

class EGPUMMIOInterface(MMIOInterface):
    """MMIO Interface using libegpu_pcidevice.dylib for fast direct memory access"""
    def __init__(self, device_handle: c_void_p, bar_index: int, fmt='I'):
        self.device_handle = device_handle
        self.bar_index = bar_index
        self.fmt = fmt
        self._mmio_interface = None
        self._mapped_size = 0
        self._setup_mapping()
    
    def _setup_mapping(self):
        """Map GPU BAR memory using libegpu_pcidevice"""
        if not libegpu:
            raise RuntimeError("libegpu_pcidevice.dylib not available")
            
        # Map the BAR using the new API
        self._mmio_interface = libegpu.egpu_device_map_bar(self.device_handle, c_int(self.bar_index))
        
        if not self._mmio_interface:
            raise RuntimeError(f"Failed to map BAR{self.bar_index} memory")
        
        # For size, use the device info or default size
        # We'll set a reasonable default for now
        self._mapped_size = 64 * 1024 * 1024  # 64MB default
        
        if NV_DEBUG >= 2:
            print(f"EGPU: Mapped BAR{self.bar_index}, mmio interface at 0x{self._mmio_interface:x}")
    
    def __len__(self):
        return self._mapped_size // struct.calcsize(self.fmt)
    
    def __getitem__(self, k):
        """Read from mapped memory"""
        if not self._mmio_interface:
            raise RuntimeError("Memory not mapped")
            
        if isinstance(k, slice):
            # Handle slice access
            start, stop, step = k.indices(len(self))
            if step != 1:
                raise NotImplementedError("Step slicing not supported")
            
            # Read multiple values
            values = []
            for i in range(start, stop):
                offset = i * struct.calcsize(self.fmt)
                if self.fmt == 'I':
                    values.append(libegpu.egpu_mmio_read32(self._mmio_interface, c_uint64(offset)))
                elif self.fmt == 'B':
                    values.append(libegpu.egpu_mmio_read8(self._mmio_interface, c_uint64(offset)))
            return values
        else:
            # Single value access
            offset = k * struct.calcsize(self.fmt)
            if self.fmt == 'I':
                return libegpu.egpu_mmio_read32(self._mmio_interface, c_uint64(offset))
            elif self.fmt == 'B':
                return libegpu.egpu_mmio_read8(self._mmio_interface, c_uint64(offset))
            else:
                raise NotImplementedError(f"Format {self.fmt} not supported")
    
    def __setitem__(self, k, v):
        """Write to mapped memory"""
        if not self._mmio_interface:
            raise RuntimeError("Memory not mapped")
            
        if isinstance(k, slice):
            # Handle slice assignment
            start, stop, step = k.indices(len(self))
            if step != 1:
                raise NotImplementedError("Step slicing not supported")
            
            # Handle bytes object assignment
            if isinstance(v, (bytes, bytearray)):
                for i, idx in enumerate(range(start, stop)):
                    if i >= len(v):
                        break
                    offset = idx * struct.calcsize(self.fmt)
                    if self.fmt == 'I':
                        # For 32-bit writes with bytes, group 4 bytes
                        if i % 4 == 0 and i + 3 < len(v):
                            value = int.from_bytes(v[i:i+4], 'little')
                            libegpu.egpu_mmio_write32(self._mmio_interface, c_uint64(offset), c_uint32(value))
                    elif self.fmt == 'B':
                        byte_val = v[i] if isinstance(v[i], int) else ord(v[i]) if isinstance(v[i], str) else v[i]
                        libegpu.egpu_mmio_write8(self._mmio_interface, c_uint64(offset), c_uint8(byte_val))
            else:
                # Handle list/array assignment
                for i, val in enumerate(range(start, stop)):
                    offset = val * struct.calcsize(self.fmt)
                    if self.fmt == 'I':
                        value = v[i] if isinstance(v, (list, tuple)) else v
                        libegpu.egpu_mmio_write32(self._mmio_interface, c_uint64(offset), c_uint32(value))
                    elif self.fmt == 'B':
                        value = v[i] if isinstance(v, (list, tuple)) else v
                        byte_val = value if isinstance(value, int) else ord(value) if isinstance(value, str) else value
                        libegpu.egpu_mmio_write8(self._mmio_interface, c_uint64(offset), c_uint8(byte_val))
        else:
            # Single value assignment
            offset = k * struct.calcsize(self.fmt)
            if self.fmt == 'I':
                libegpu.egpu_mmio_write32(self._mmio_interface, c_uint64(offset), c_uint32(v))
            elif self.fmt == 'B':
                byte_val = v if isinstance(v, int) else ord(v) if isinstance(v, str) else v
                libegpu.egpu_mmio_write8(self._mmio_interface, c_uint64(offset), c_uint8(byte_val))
            else:
                raise NotImplementedError(f"Format {self.fmt} not supported")
        
        # Ensure writes complete
        libegpu.egpu_mmio_memory_barrier(self._mmio_interface)
    
    def view(self, offset: int = 0, size: int | None = None, fmt=None):
        """Create a view of this interface at a different offset"""
        # For now, create a new interface instance
        # In a full implementation, this would handle offset properly
        return EGPUMMIOInterface(self.device_handle, self.bar_index, fmt or self.fmt)
    
    def __del__(self):
        """Cleanup - memory unmapping handled by device cleanup"""
        # Memory cleanup is handled by the device destructor in C++
        pass

class EGPUDev:
    """macOS eGPU device using libegpu_pcidevice.dylib - compatible with tinygrad NVDev interface"""
    
    def __init__(self, devfmt: str, device_id: int = 0):
        self.devfmt = devfmt
        self.device_id = device_id
        self.device_handle = None
        self.mmio = None
        self.vram = None
        self.lock_fd = None
        self.device_info = None
        
        # Initialize eGPU connection 
        self._connect_to_driver()
        self._get_device_info()
        
        # Create MMIO interfaces
        self.mmio = EGPUMMIOInterface(self.device_handle, 0, fmt='I')  # BAR0 for registers
        self.vram = EGPUMMIOInterface(self.device_handle, 2, fmt='I')  # BAR2 for VRAM (if available)
        
        # NVDev compatibility
        self.venid = self.device_info.vendor_id if self.device_info else 0x10de
        self.subvenid = 0  # Not available from libdkmmio
        self.rev = 0      # Not available from libdkmmio
        # Store bars as (address, size) tuples to match PCI implementation
        if self.device_info:
            print(f"🔍 Building bars dict from device_info:")
            print(f"   bar_count: {self.device_info.bar_count}")
            print(f"   Raw bar_bases array: {[hex(self.device_info.bar_bases[i]) for i in range(6)]}")
            print(f"   Raw bar_sizes array: {[self.device_info.bar_sizes[i] for i in range(6)]}")
            for i in range(6):  # Check all possible BARs 0-5
                try:
                    base = self.device_info.bar_bases[i]
                    size = self.device_info.bar_sizes[i]
                    size_gb = size / (1024**3)
                    human = f"{size_gb:.2f} GiB" if size_gb >= 0.01 else f"{size / (1024**2):.2f} MiB"
                    print(f"   BAR{i}: base=0x{base:x}, size={size} ({human}) ({'included' if size > 0 else 'skipped'})")
                except Exception as e:
                    print(f"   BAR{i}: not accessible - {e}")
            
            self.bars = {i: (self.device_info.bar_bases[i], self.device_info.bar_sizes[i]) 
                        for i in range(6)  # Check all possible BARs instead of just bar_count
                        if i < len(self.device_info.bar_bases) and 
                           i < len(self.device_info.bar_sizes) and
                           self.device_info.bar_sizes[i] > 0}
            # Also print a concise summary with GB units
            summary = {i: (hex(self.device_info.bar_bases[i]), f"{self.device_info.bar_sizes[i]/(1024**3):.2f} GiB") for i in self.bars.keys()}
            print(f"🔍 Final bars dict: {self.bars}")
            print(f"🔍 Final bars (human): {summary}")
        else:
            self.bars = {}
        
        self.smi_dev, self.is_booting = False, True
        self._early_init()
        
        # Use BAR2 size for VRAM if available, otherwise fallback
        self.vram_size = self.bars[2][1] if 2 in self.bars else 64 * 1024 * 1024
        
        # Initialize memory manager
        try:
            bits, shifts = (56, [12, 21, 29, 38, 47, 56]) if self.mmu_ver == 3 else (48, [12, 21, 29, 38, 47])
            self.mm = NVMemoryManager(self, self.vram_size, boot_size=(2 << 20), pt_t=NVPageTableEntry, 
                                     va_bits=bits, va_shifts=shifts, va_base=0,
                                     palloc_ranges=[(x, x) for x in [512 << 20, 2 << 20, 4 << 10]])
        except Exception as e:
            print(f"Warning: Memory manager init failed: {e}")
            self.mm = None
        
        # Initialize firmware components if supported
        try:
            self.flcn: NV_FLCN|NV_FLCN_COT = NV_FLCN_COT(self) if self.fmc_boot else NV_FLCN(self)
            self.gsp: NV_GSP = NV_GSP(self)
            
            self.is_booting = False
            
            print("init_sw")
            for ip in [self.flcn, self.gsp]: ip.init_sw()
            print("init_hw")
            #exit() #debug-crash-msi-interrupts
            for ip in [self.flcn, self.gsp]: ip.init_hw()
        except Exception as e:
            print(f"Warning: Firmware init failed (this may be normal for eGPU): {e}")
            self.flcn = None
            self.gsp = None
            self.is_booting = False
        
        print(f"EGPUDev initialized: devfmt={devfmt}, device_id={device_id}, VRAM={self.vram_size} bytes ({self.vram_size/(1024**3):.2f} GiB)")
    
    def _connect_to_driver(self):
        """Connect to the EGPUMapperDriver using libegpu_pcidevice"""
        if not libegpu:
            raise RuntimeError("libegpu_pcidevice.dylib not available")
        
        # Create device with default BARs (0, 2)
        bars = [0, 2, 4 ]
        bars_array = (c_int * len(bars))(*bars)
        self.device_handle = libegpu.egpu_device_create(self.device_id, bars_array, len(bars))
        if not self.device_handle:
            raise RuntimeError(f"Failed to open device {self.device_id} via libegpu_pcidevice")
        
        print(f"✅ Connected to EGPUMapperDriver via libegpu_pcidevice (device {self.device_id})")
    
    def _get_device_info(self):
        """Get device information from libegpu_pcidevice"""
        if not self.device_handle:
            return
            
        info_ptr = libegpu.egpu_device_get_info(self.device_handle)
        if info_ptr:
            self.device_info = info_ptr.contents
            print(f"✅ Device: 0x{self.device_info.vendor_id:04x}:0x{self.device_info.device_id:04x}, {self.device_info.bar_count} BARs")
            count =0
            if NV_DEBUG >= 1:
                for i in range(self.device_info.bar_count):
                    if self.device_info.bar_sizes[i] > 0:
                        size_mb = self.device_info.bar_sizes[i] / (1024 * 1024)
                        size_gb = self.device_info.bar_sizes[i] / (1024 * 1024 * 1024)
                        print(f"  BAR{i}: base=0x{self.device_info.bar_bases[i]:x}, size={self.device_info.bar_sizes[i]} ({size_mb:.0f}MB / {size_gb:.2f}GB)")
                        count+=1
                if count <3 :
                    print("expected 3 or more bars!")
                    exit()
                    
        else:
            print(f"⚠️  Failed to get device info")
    
    def _early_init(self):
        """Early initialization - create register definitions"""
        self.reg_names: set[str] = set()
        self.reg_offsets: dict[str, tuple[int, int]] = {}
        
        # Try to initialize chip detection (may fail on eGPU)
        try:
            self.include("src/common/inc/swref/published/nv_ref.h")
            self.chip_id = self.reg("NV_PMC_BOOT_0").read()
            self.chip_details = self.reg("NV_PMC_BOOT_42").read_bitfields()
            self.chip_name = {0x17: "GA1", 0x19: "AD1", 0x1b: "GB2"}[self.chip_details['architecture']] + f"{self.chip_details['implementation']:02d}"
            if self.chip_name =="GB205": 
                self.chip_name = "GB202" #keep it! require for firmawar 

            self.mmu_ver, self.fmc_boot = (3, True) if self.chip_details['architecture'] >= 0x1a else (2, False)
        except Exception as e:
            print(f"Warning: Chip detection failed (using defaults): {e}")
            # Use safe defaults for unknown GPU - use GB202 for firmware compatibility
            self.chip_id = 0x1b2000a1  # Changed to GB202 for f/w download
            self.chip_details = {'architecture': 0x1b, 'implementation': 2}
            self.chip_name = "GB202" 
            self.mmu_ver, self.fmc_boot = (3, True)
        
        print(f"Detected chip: {self.chip_name}, MMU v{self.mmu_ver}")
        
        # Initialize basic register structure
        self._create_basic_registers()
    
    def _create_basic_registers(self):
        """Create basic register structure for eGPU compatibility"""
        # Create essential registers for memory management
        try:
            # Include basic GPU register definitions
            self.include("src/common/inc/swref/published/turing/tu102/dev_fb.h")
            self.include("src/common/inc/swref/published/turing/tu102/dev_vm.h")
            self.include("src/common/inc/swref/published/ampere/ga102/dev_gc6_island.h")
            self.include("src/common/inc/swref/published/ampere/ga102/dev_gc6_island_addendum.h")
            
            # MMU Init
            mmu_pd_names = [f'NV_MMU_VER{self.mmu_ver}_PTE', f'NV_MMU_VER{self.mmu_ver}_PDE', f'NV_MMU_VER{self.mmu_ver}_DUAL_PDE']
            self.reg_names.update(mmu_pd_names)
            for name in mmu_pd_names: 
                self.__dict__[name] = NVReg(self, None, None, fields={})
            
            self.include(f"kernel-open/nvidia-uvm/hwref/{'hopper/gh100' if self.mmu_ver == 3 else 'turing/tu102'}/dev_mmu.h")
            self.pte_t, self.pde_t, self.dual_pde_t = tuple([self.__dict__[name] for name in mmu_pd_names])
            
        except Exception as e:
            print(f"Warning: Register initialization failed: {e}")
            # Create minimal fallback registers
            self.pte_t = NVReg(self, None, None, fields={})
            self.pde_t = NVReg(self, None, None, fields={})
            self.dual_pde_t = NVReg(self, None, None, fields={})
    
    def reg(self, reg: str) -> NVReg: 
        return self.__dict__[reg]
    
    def wreg(self, addr: int, value: int):
        """Write register via MMIO"""
        if not self.mmio:
            print(f"ERROR: MMIO not available for wreg at {hex(addr)}")
            return
            
        try:
            self.mmio[addr // 4] = value
            if NV_DEBUG >= 4: 
                print(f"wreg: {hex(addr)} = {hex(value)}")
        except Exception as e:
            print(f"ERROR: wreg failed at {hex(addr)}: {e}")
    
    def rreg(self, addr: int) -> int:
        """Read register via MMIO"""
        if not self.mmio:
            print(f"ERROR: MMIO not available for rreg at {hex(addr)}")
            return 0xffffffff
            
        try:
            value = self.mmio[addr // 4]
            if NV_DEBUG >= 4:
                print(f"rreg: {hex(addr)} = {hex(value)}")
            return value
        except Exception as e:
            print(f"ERROR: rreg failed at {hex(addr)}: {e}")
            return 0xffffffff
    
    def fini(self):
        """Cleanup"""
        try:
            if self.gsp and self.flcn:
                for ip in [self.gsp, self.flcn]: 
                    ip.fini_hw()
        except:
            pass
            
        if self.device_handle and libegpu:
            libegpu.egpu_device_destroy(self.device_handle)
            print("EGPUDev connection closed")
    
    def allocate_dma_buffer(self, size: int, direction: int = 2):
        """Allocate DMA buffer using the libegpu_pcidevice API"""
        if not self.device_handle or not libegpu:
            print("ERROR: Device not connected or libegpu not available")
            return None
            
        print(f"🔄 Allocating DMA buffer ({size} bytes, direction={direction})...")
        
        # Prepare output parameters for physical and virtual addresses
        physical_addr = c_uint64(0)
        virtual_addr = c_uint64(0)
        
        # Call the C function - returns handle directly
        handle = libegpu.egpu_device_allocate_dma_buffer(
            self.device_handle,
            size,
            direction,
            byref(physical_addr),
            byref(virtual_addr)
        )
        
        if handle != 0:
            buffer_info = {
                'handle': handle,
                'size': size,
                'physical_addr': physical_addr.value,
                'virtual_addr': virtual_addr.value
            }
            print(f"✅ DMA buffer allocated: handle=0x{handle:x}, paddr=0x{physical_addr.value:x}, vaddr=0x{virtual_addr.value:x}")
            return buffer_info
        else:
            print(f"❌ DMA buffer allocation failed")
            return None
    
    def get_connection(self):
        """Get the IOKit connection handle for memory mapping"""
        if not self.device_handle or not libegpu:
            print("ERROR: Device not connected or libegpu not available")
            return None
            
        # Get the IOKit connection handle from the device
        connection = libegpu.egpu_device_get_connection(self.device_handle)
        if connection:
            print(f"✅ Got IOKit connection handle: 0x{connection:x}")
            return connection
        else:
            print("❌ Failed to get IOKit connection handle")
            return None
    
    def destroy_dma_buffer(self, handle: int) -> bool:
        """Destroy DMA buffer using the libegpu_pcidevice API"""
        if not self.device_handle or not libegpu:
            print("ERROR: Device not connected or libegpu not available")
            return False
            
        result = libegpu.egpu_device_destroy_dma_buffer(self.device_handle, handle)
        
        success = result == 1
        if success:
            print("✅ DMA buffer destroyed successfully!")
        else:
            print(f"❌ DMA buffer destroy failed (result: {result})")
        
        return success
    
    def get_memory_type_for_handle(self, handle: int) -> int:
        """Get the 32-bit memory type ID for a 64-bit handle (for IOConnectMapMemory)"""
        if not self.device_handle or not libegpu:
            print("ERROR: Device not connected or libegpu not available")
            return 0
            
        memory_type = libegpu.egpu_device_get_memory_type_for_handle(self.device_handle, handle)
        
        if NV_DEBUG >= 1:
            print(f"🔄 Memory type for handle 0x{handle:x}: {memory_type}")
        
        return memory_type
    
    def copy_dma_buffers(self, src_handle: int, dst_handle: int, offset: int, size: int) -> bool:
        """Copy data between DMA buffers in kernel space"""
        if not self.device_handle or not libegpu:
            print("ERROR: Device not connected or libegpu not available")
            return False
            
        result = libegpu.egpu_device_copy_dma_buffers(self.device_handle, src_handle, dst_handle, offset, size)
        
        success = result == 1
        if NV_DEBUG >= 1:
            if success:
                print(f"✅ DMA copy completed: src=0x{src_handle:x} -> dst=0x{dst_handle:x}, offset={offset}, size={size}")
            else:
                print(f"❌ DMA copy failed (result: {result})")
        
        return success
    
    def get_connection(self) -> int:
        """Get IOKit connection handle for IOConnectMapMemory"""
        if not self.device_handle or not libegpu:
            print("ERROR: Device not connected or libegpu not available")
            return 0
            
        connection = libegpu.egpu_device_get_connection(self.device_handle)
        return connection
    
    # Include file processing (from NVDev)
    def _alloc_boot_struct(self, struct: ctypes.Structure) -> tuple[ctypes.Structure, int]:
        # For eGPU on macOS, we'll use a simpler allocation method
        sz = ctypes.sizeof(type(struct))
        # Allocate memory using ctypes
        buf = (ctypes.c_byte * sz)()
        va = ctypes.addressof(buf)
        # Copy struct data
        ctypes.memmove(va, ctypes.addressof(struct), sz)
        # Return struct at new address and dummy physical address
        return type(struct).from_address(va), va

    def _download(self, file: str) -> str:
        url = f"https://raw.githubusercontent.com/NVIDIA/open-gpu-kernel-modules/8ec351aeb96a93a4bb69ccc12a542bf8a8df2b6f/{file}"
        return fetch(url, subdir="defines").read_text()

    def extract_fw(self, file: str, dname: str) -> bytes:
        # Extracts the firmware binary from the given header
        tname = file.replace("kgsp", "kgspGet")
        text = self._download(f"src/nvidia/generated/g_bindata_{tname}_{self.chip_name}.c")
        info, sl = text[text[:text.index(dnm:=f'{file}_{self.chip_name}_{dname}')].rindex("COMPRESSION:"):][:16], text[text.index(dnm) + len(dnm) + 7:]
        image = bytes.fromhex(sl[:sl.find("};")].strip().replace("0x", "").replace(",", "").replace(" ", "").replace("\n", ""))
        return gzip.decompress(struct.pack("<4BL2B", 0x1f, 0x8b, 8, 0, 0, 0, 3) + image) if "COMPRESSION: YES" in info else image

    def include(self, file: str):
        regs_off = {'NV_PFALCON_FALCON': 0x0, 'NV_PGSP_FALCON': 0x0, 'NV_PSEC_FALCON': 0x0, 'NV_PRISCV_RISCV': 0x1000, 'NV_PGC6_AON': 0x0, 'NV_PFSP': 0x0,
          'NV_PGC6_BSI': 0x0, 'NV_PFALCON_FBIF': 0x600, 'NV_PFALCON2_FALCON': 0x1000, 'NV_PBUS': 0x0, 'NV_PFB': 0x0, 'NV_PMC': 0x0, 'NV_PGSP_QUEUE': 0x0,
          'NV_VIRTUAL_FUNCTION':0xb80000}

        for raw in self._download(file).splitlines():
          if not raw.startswith("#define "): continue

          if m:=re.match(r'#define\s+(\w+)\s+([0-9\+\-\*\(\)]+):([0-9\+\-\*\(\)]+)', raw): # bitfields
            name, hi, lo = m.groups()

            reg = next((r for r in self.reg_names if name.startswith(r+"_")), None)
            if reg is not None: self.__dict__[reg].add_field(name[len(reg)+1:].lower(), eval(lo), eval(hi))
            else: self.reg_offsets[name] = (eval(lo), eval(hi))
            continue

          if m:=re.match(r'#define\s+(\w+)\s*\(\s*(\w+)\s*\)\s*(.+)', raw): # reg set
            fn = m.groups()[2].strip().rstrip('\\').split('/*')[0].rstrip()
            name, value = m.groups()[0], eval(f"lambda {m.groups()[1]}: {fn}")
          elif m:=re.match(r'#define\s+(\w+)\s+([0-9A-Fa-fx]+)(?![^\n]*:)', raw): name, value = m.groups()[0], int(m.groups()[1], 0) # reg value
          else: continue

          reg_pref = next((prefix for prefix in regs_off.keys() if name.startswith(prefix)), None)
          not_already_reg = not any(name.startswith(r+"_") for r in self.reg_names)

          if reg_pref is not None and not_already_reg:
            fields = {k[len(name)+1:]: v for k, v in self.reg_offsets.items() if k.startswith(name+'_')}
            self.__dict__[name] = NVReg(self, regs_off[reg_pref], value, fields=fields)
            self.reg_names.add(name)
          else: self.__dict__[name] = value

class EGPUAllocator(Allocator):
    """Memory allocator for eGPU using VRAM"""
    
    def __init__(self, device: EGPUDev):
        self.device = device
        self.allocations = {}  # Track allocated memory regions
        self.next_addr = 0x1000  # Start allocations at 4KB offset
        
    def _alloc(self, size: int, options) -> int:
        """Allocate GPU memory"""
        # Align to 256 bytes (common GPU alignment)
        aligned_size = (size + 255) & ~255
        
        # Simple linear allocator for now
        addr = self.next_addr
        self.next_addr += aligned_size
        
        # Check if we exceed VRAM size
        if self.next_addr > self.device.vram_size:
            raise RuntimeError(f"Out of VRAM: requested {aligned_size}, available {self.device.vram_size - addr}")
        
        self.allocations[addr] = aligned_size
        
        if NV_DEBUG >= 3:
            print(f"EGPU alloc: addr=0x{addr:x}, size={aligned_size}")
        
        return addr
    
    def copyin(self, dest: int, src: memoryview) -> None:
        """Copy data from host to GPU memory"""
        if not self.device.vram:
            raise RuntimeError("VRAM interface not available")
        
        # Write data to VRAM using MMIO
        try:
            src_bytes = bytes(src)
            for i in range(0, len(src_bytes), 4):
                # Read 4 bytes at a time, pad if necessary
                chunk = src_bytes[i:i+4]
                if len(chunk) < 4:
                    chunk += b'\x00' * (4 - len(chunk))
                
                # Convert to uint32 and write
                value = int.from_bytes(chunk, byteorder='little')
                self.device.vram[(dest + i) // 4] = value
                
            if NV_DEBUG >= 4:
                print(f"EGPU copyin: dest=0x{dest:x}, size={len(src_bytes)}")
                
        except Exception as e:
            print(f"ERROR: EGPU copyin failed: {e}")
            raise
    
    def copyout(self, dest: memoryview, src: int) -> None:
        """Copy data from GPU memory to host"""
        if not self.device.vram:
            raise RuntimeError("VRAM interface not available")
        
        try:
            # Read data from VRAM using MMIO
            for i in range(0, len(dest), 4):
                value = self.device.vram[(src + i) // 4]
                # Convert uint32 to bytes
                chunk = value.to_bytes(4, byteorder='little')
                
                # Copy to destination, handling partial chunks
                end_idx = min(i + 4, len(dest))
                dest[i:end_idx] = chunk[:end_idx-i]
                
            if NV_DEBUG >= 4:
                print(f"EGPU copyout: src=0x{src:x}, size={len(dest)}")
                
        except Exception as e:
            print(f"ERROR: EGPU copyout failed: {e}")
            raise

    def _free(self, opaque: int, size: int, options) -> None:
        """Free GPU memory"""
        if opaque in self.allocations:
            del self.allocations[opaque]
            if NV_DEBUG >= 3:
                print(f"EGPU free: addr=0x{opaque:x}, size={size}")
        else:
            print(f"WARNING: Attempt to free unknown address 0x{opaque:x}")

def create_egpu_device(device_id: int = 0) -> EGPUDev:
    """Create an eGPU device instance for tinygrad with DMA buffer support"""
    devfmt = f"egpu{device_id}"
    return EGPUDev(devfmt, device_id)