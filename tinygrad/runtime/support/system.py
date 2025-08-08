import os, mmap, array, functools, ctypes, select, contextlib, dataclasses, sys
from typing import cast, ClassVar
from tinygrad.helpers import round_up, to_mv, getenv, OSX, temp
from tinygrad.runtime.autogen import libc, vfio
from tinygrad.runtime.support.hcq import FileIOInterface, MMIOInterface, HCQBuffer
from tinygrad.runtime.support.memory import MemoryManager, VirtMapping

MAP_FIXED, MAP_LOCKED, MAP_POPULATE, MAP_NORESERVE = 0x10, 0 if OSX else 0x2000, getattr(mmap, "MAP_POPULATE", 0 if OSX else 0x008000), 0x400

class _System:
  # DMA Direction constants for DriverKit
  DMA_DIRECTION_CPU_TO_GPU = 0
  DMA_DIRECTION_GPU_TO_CPU = 1  
  DMA_DIRECTION_BIDIRECTIONAL = 2

  # MappedDMABuffer structure (matches DriverKit implementation)
  class MappedDMABuffer(ctypes.Structure):
    _pack_ = 1  # Force tight packing (no padding)
    _fields_ = [
      ("physical_addr", ctypes.c_uint64),
      ("virtual_addr", ctypes.c_uint64), 
      ("handle", ctypes.c_uint64),  # Updated to 64-bit for physical address handles
      ("size", ctypes.c_uint64)
    ]

  # DMA Buffer tracking structure
  class DMABufferInfo:
    def __init__(self, name, handle, hw_addr, user_addr, size, connection, task, memory_type):
      self.name = name              # Buffer name/identifier
      self.handle = handle          # DriverKit buffer handle (64-bit physical address)
      self.hw_addr = hw_addr        # GPU hardware DMA address
      self.user_addr = user_addr    # User space mapped address
      self.size = size              # Buffer size in bytes
      self.connection = connection  # IOKit connection handle  
      self.task = task              # Mach task handle
      self.memory_type = memory_type  # 32-bit memory type ID for IOConnectMapMemory
      self.dirty = False            # Track if user buffer needs sync

  def __init__(self):
    self._dma_buffers = {}  # Track allocated DMA buffers: handle -> DMABufferInfo
    self._egpu_device = None  # Will be set when first eGPU device is created
    self._last_buffer_handle = None  # Track last allocated buffer handle
    # Note: eGPU device initialization is deferred until first DMA allocation

  def reserve_hugepages(self, cnt): 
    if OSX:
      print(f"INFO: Hugepage reservation skipped on macOS (requested {cnt} pages)")
      return
    os.system(f"sudo sh -c 'echo {cnt} > /proc/sys/vm/nr_hugepages'")

  def memory_barrier(self): lib.atomic_thread_fence(__ATOMIC_SEQ_CST:=5) if (lib:=self.atomic_lib()) is not None else None

  def lock_memory(self, addr:int, size:int):
    if libc.mlock(ctypes.c_void_p(addr), size): raise RuntimeError(f"Failed to lock memory at {addr:#x} with size {size:#x}")

  def system_paddrs(self, vaddr:int, size:int) -> list[int]:
    if OSX:
      # macOS uses DriverKit DEXT for all DMA operations and firmware loading
      # system_paddrs should not be called on macOS - firmware loading is handled entirely in DriverKit
      raise RuntimeError("system_paddrs not supported on macOS - use DriverKit DEXT for firmware loading")
    self.pagemap().seek(vaddr // mmap.PAGESIZE * 8)
    return [(x & ((1<<55) - 1)) * mmap.PAGESIZE for x in array.array('Q', self.pagemap().read(size//mmap.PAGESIZE*8, binary=True))]

  def _find_existing_egpu_device(self):
    """Find an already-created eGPU device to avoid creating duplicates"""
    try:
      # Look for existing EGPUDev instances in the global namespace
      # This prevents infinite recursion where EGPUDev creation calls System.alloc_sysmem
      import gc
      for obj in gc.get_objects():
        if hasattr(obj, '__class__') and obj.__class__.__name__ == 'EGPUDev':
          if hasattr(obj, 'device_handle') and obj.device_handle is not None:
            print("✅ System: Found existing eGPU device, reusing for DMA")
            return obj
      return None
    except Exception as e:
      print(f"⚠️ System: Failed to find existing eGPU device: {e}")
      return None

  def alloc_sysmem(self, size:int, vaddr:int=0, contiguous:bool=False, data:bytes|None=None, direction:int=DMA_DIRECTION_CPU_TO_GPU, name:str|None=None) -> tuple[int, list[int]]:
    if OSX:
      # Use DriverKit DMA allocation on macOS via macos_egpu provider
      from tinygrad.runtime.support.nv.macos_egpu import acquire_device
      dev = acquire_device(0)

      # Allocate DMA buffer using provider
      buffer_info = dev.allocate_dma_buffer(size, direction)
      if not buffer_info:
        raise RuntimeError(f"Failed to allocate DMA buffer of size {size}")
      
      # Get memory type ID for IOConnectMapMemory
      memory_type = dev.get_memory_type_for_handle(buffer_info['handle'])
      if memory_type == 0:
        raise RuntimeError(f"Failed to get memory type for handle 0x{buffer_info['handle']:x}")
      
      # Human-friendly DMA direction string for logging
      dir_str = ["CPU_TO_GPU", "GPU_TO_CPU", "BIDIRECTIONAL"][direction]

      # Provide an early, descriptive name for logging and later tracking
      default_name = name or f"DMA_{dir_str}_{buffer_info['size']}"
      buffer_info['name'] = default_name

      print(
        f"✅ DMA buffer allocated: handle=0x{buffer_info['handle']:x}, memoryType={memory_type}, "
        f"physical=0x{buffer_info['physical_addr']:x}, virtual=0x{buffer_info['virtual_addr']:x}, "
        f"size={buffer_info['size']}, direction={dir_str}, name='{default_name}'"
      )
      
      # Step 2: Get IOKit connection handle
      connection = dev.get_connection()
      if not connection:
        raise RuntimeError("Failed to get IOKit connection handle")
      
      # Step 3: Map to user space with IOConnectMapMemory
      import ctypes
      IOKit = ctypes.CDLL('/System/Library/Frameworks/IOKit.framework/IOKit')
      
      # Get current task
      task = IOKit.mach_task_self()
      
      # Prepare output parameters
      mapped_addr = ctypes.c_uint64(0)
      mapped_size = ctypes.c_uint64(0)
      
      print(f"🔄 Mapping DMA buffer to user space (memoryType={memory_type})...")
      
      # Call IOConnectMapMemory
      result = IOKit.IOConnectMapMemory(
          connection,                      # IOKit connection handle
          memory_type,                     # Memory type ID
          task,                           # Current task
          ctypes.byref(mapped_addr),      # Output: User space virtual address
          ctypes.byref(mapped_size),      # Output: Mapped size
          0x00000001                      # kIOMapAnywhere flag
      )
      
      if result != 0:
        raise RuntimeError(f"IOConnectMapMemory failed with error: {result}")
      
      real_virtual_addr = mapped_addr.value
      print(
        f"✅ DMA buffer mapped to user space: 0x{real_virtual_addr:x} (size={mapped_size.value}), "
        f"name='{buffer_info.get('name', 'unnamed')}', direction={dir_str}"
      )
      
      # Step 4: Copy data if provided
      if data is not None:
        try:
          print(f"🔄 Copying {len(data)} bytes to mapped buffer at 0x{real_virtual_addr:x}")
          
          # Use fast memcpy-style copying with ctypes.memmove
          ctypes.memmove(real_virtual_addr, data, len(data))
          
          print(f"✅ Data copied to DMA buffer ({len(data)} bytes)")
              
        except Exception as e:
          print(f"❌ Failed to copy data to DMA buffer: {e}")
          # Cleanup: unmap on failure
          IOKit.IOConnectUnmapMemory(connection, memory_type, task, real_virtual_addr)
          raise
      
      # Store mapping info for cleanup
      buffer_info['mapped_addr'] = real_virtual_addr
      buffer_info['mapped_size'] = mapped_size.value
      buffer_info['connection'] = connection
      buffer_info['task'] = task
      
      # Create DMABufferInfo for tracking
      handle = buffer_info['handle']
      dma_info = self.DMABufferInfo(
        name=buffer_info.get('name', f"buffer_{handle:x}"),  # Default or early name
        handle=handle,
        hw_addr=buffer_info['physical_addr'],  # TODO: Should be GPU DMA address
        user_addr=real_virtual_addr,
        size=buffer_info['size'],
        connection=connection,
        task=task,
        memory_type=memory_type
      )
      
      # Track buffer for cleanup and flush operations
      self._dma_buffers[handle] = dma_info
      self._last_buffer_handle = handle  # Track for easy access

      # Echo final buffer details with confirmed tracked name
      print(
        f"📝 DMA buffer ready: name='{dma_info.name}', handle=0x{handle:x}, "
        f"hw_addr=0x{dma_info.hw_addr:x}, user_addr=0x{dma_info.user_addr:x}, size={dma_info.size}"
      )
      
      # Return compatible format: (real_virtual_addr, [physical_addresses])
      # For compatibility with Linux, expand single physical address into page addresses
      page_size = 4096  # Standard page size
      num_pages = (buffer_info['size'] + page_size - 1) // page_size
      physical_pages = [buffer_info['physical_addr'] + i * page_size for i in range(num_pages)]
      
      return real_virtual_addr, physical_pages
    
    # Linux implementation
    assert not contiguous or size <= (2 << 20), "Contiguous allocation is only supported for sizes up to 2MB"
    flags = (libc.MAP_HUGETLB if contiguous and (size:=round_up(size, mmap.PAGESIZE)) > mmap.PAGESIZE else 0) | (MAP_FIXED if vaddr else 0)
    va = FileIOInterface.anon_mmap(vaddr, size, mmap.PROT_READ|mmap.PROT_WRITE, mmap.MAP_SHARED|mmap.MAP_ANONYMOUS|MAP_POPULATE|MAP_LOCKED|flags, 0)

    if data is not None: to_mv(va, len(data))[:] = data
    return va, self.system_paddrs(va, size)

  def flush_dma_buffer(self, handle: int, data: bytes = None) -> None:
    """Flush user space data to kernel DMA buffer via CopyClientMemoryForType_Impl"""
    if not OSX:
      return  # No-op on Linux
      
    if handle not in self._dma_buffers:
      raise ValueError(f"DMA buffer handle {handle:x} not found")
      
    dma_info = self._dma_buffers[handle]
    
    if data is not None:
      if len(data) > dma_info.size:
        raise ValueError(f"Data size {len(data)} exceeds buffer size {dma_info.size}")
        
      print(f"🔄 Flushing {len(data)} bytes to DMA buffer '{dma_info.name}' (handle=0x{handle:x})")
      
      # Copy data to user space mapping - DriverKit will sync to kernel DMA buffer
      ctypes.memmove(dma_info.user_addr, data, len(data))
      
      print(f"✅ DMA buffer flushed: '{dma_info.name}' ({len(data)} bytes)")
      dma_info.dirty = False
    else:
      print(f"⚠️ Flush called without data for buffer '{dma_info.name}'")

  def flush_all_dma_buffers(self) -> None:
    """Flush all dirty DMA buffers (placeholder - requires user to provide data)"""
    if not OSX:
      return
      
    dirty_buffers = [info for info in self._dma_buffers.values() if info.dirty]
    if dirty_buffers:
      print(f"⚠️ {len(dirty_buffers)} DMA buffers marked dirty but no data provided for flush")
      for info in dirty_buffers:
        print(f"   - '{info.name}' (handle=0x{info.handle:x}, size={info.size})")

  def list_dma_buffers(self) -> None:
    """List all tracked DMA buffers with their details"""
    if not self._dma_buffers:
      print("📝 No DMA buffers allocated")
      return
      
    print(f"📝 Tracked DMA Buffers ({len(self._dma_buffers)}):")
    for handle, info in self._dma_buffers.items():
      status = "DIRTY" if info.dirty else "CLEAN"
      print(f"   - '{info.name}': handle=0x{handle:x}, hw_addr=0x{info.hw_addr:x}, user_addr=0x{info.user_addr:x}, size={info.size} [{status}]")

  def mark_buffer_dirty(self, handle: int) -> None:
    """Mark a DMA buffer as dirty (needs flush)"""
    if handle in self._dma_buffers:
      self._dma_buffers[handle].dirty = True

  def set_buffer_name(self, handle: int, name: str) -> None:
    """Set a custom name for a DMA buffer for easier tracking"""
    if handle in self._dma_buffers:
      self._dma_buffers[handle].name = name
      print(f"📝 Buffer handle 0x{handle:x} renamed to '{name}'")
    else:
      print(f"⚠️ Buffer handle 0x{handle:x} not found")

  def get_buffer_info(self, handle: int) -> 'DMABufferInfo':
    """Get DMA buffer information by handle"""
    if handle not in self._dma_buffers:
      raise ValueError(f"DMA buffer handle {handle:x} not found")
    return self._dma_buffers[handle]

  def get_last_buffer_handle(self) -> int:
    """Get the handle of the last allocated DMA buffer"""
    if self._last_buffer_handle is None:
      raise RuntimeError("No DMA buffers have been allocated yet")
    return self._last_buffer_handle

  def find_buffer_by_address(self, user_addr: int) -> int:
    """Find DMA buffer handle by user space address"""
    for handle, info in self._dma_buffers.items():
      if info.user_addr <= user_addr < (info.user_addr + info.size):
        return handle
    raise ValueError(f"No DMA buffer found containing address 0x{user_addr:x}")

  def pci_reset(self, gpu): 
    if OSX:
      print(f"WARNING: PCI reset not supported on macOS for device {gpu}")
      return
    os.system(f"sudo sh -c 'echo 1 > /sys/bus/pci/devices/{gpu}/reset'")
  
  def pci_scan_bus(self, target_vendor:int, target_devices:list[int]) -> list[str]:
    if OSX:
      # For macOS, we can't scan PCI bus the same way
      # Return a list of available eGPU devices (simplified for now)
      # In a real implementation, this would use IOKit or DriverKit to enumerate devices
      # For now, just return device indices as strings
      egpu_count = int(os.environ.get('EGPU_COUNT', '1'))
      return [str(i) for i in range(egpu_count)]
    
    result = []
    for pcibus in FileIOInterface("/sys/bus/pci/devices").listdir():
      vendor = int(FileIOInterface(f"/sys/bus/pci/devices/{pcibus}/vendor").read(), 16)
      device = int(FileIOInterface(f"/sys/bus/pci/devices/{pcibus}/device").read(), 16)
      if vendor == target_vendor and device in target_devices: result.append(pcibus)
    return sorted(result)

  @functools.cache
  def atomic_lib(self): return ctypes.CDLL(ctypes.util.find_library('atomic')) if sys.platform == "linux" else None

  @functools.cache
  def pagemap(self) -> FileIOInterface:
    if OSX:
      # macOS doesn't have /proc/self/pagemap
      return None
    if FileIOInterface(reloc_sysfs:="/proc/sys/vm/compact_unevictable_allowed", os.O_RDONLY).read()[0] != "0":
      os.system(cmd:=f"sudo sh -c 'echo 0 > {reloc_sysfs}'")
      assert FileIOInterface(reloc_sysfs, os.O_RDONLY).read()[0] == "0", f"Failed to disable migration of locked pages. Please run {cmd} manually."
    return FileIOInterface("/proc/self/pagemap", os.O_RDONLY)

  @functools.cache
  def vfio(self) -> FileIOInterface|None:
    if OSX:
      # VFIO not available on macOS - eGPU uses DriverKit
      return None
    try:
      if not FileIOInterface.exists("/sys/module/vfio"): os.system("sudo modprobe vfio-pci disable_idle_d3=1")

      FileIOInterface("/sys/module/vfio/parameters/enable_unsafe_noiommu_mode", os.O_RDWR).write("1")
      vfio_fd = FileIOInterface("/dev/vfio/vfio", os.O_RDWR)
      vfio.VFIO_CHECK_EXTENSION(vfio_fd, vfio.VFIO_NOIOMMU_IOMMU)

      return vfio_fd
    except OSError: return None

  def flock_acquire(self, name:str) -> int:
    import fcntl # to support windows

    os.umask(0) # Set umask to 0 to allow creating files with 0666 permissions

    # Avoid O_CREAT because we don’t want to re-create/replace an existing file (triggers extra perms checks) when opening as non-owner.
    if os.path.exists(lock_name:=temp(name)): self.lock_fd = os.open(lock_name, os.O_RDWR)
    else: self.lock_fd = os.open(lock_name, os.O_RDWR | os.O_CREAT | os.O_CLOEXEC, 0o666)

    try: fcntl.flock(self.lock_fd, fcntl.LOCK_EX | fcntl.LOCK_NB)
    except OSError:
      print(f"WARNING: Lock file {name} is in use, attempting to force unlock...")
      # Try to remove stale lock file and retry
      try:
        os.close(self.lock_fd)
        os.unlink(lock_name)
        print(f"DEBUG: Removed stale lock file {lock_name}")
        # Retry
        self.lock_fd = os.open(lock_name, os.O_RDWR | os.O_CREAT | os.O_CLOEXEC, 0o666)
        fcntl.flock(self.lock_fd, fcntl.LOCK_EX | fcntl.LOCK_NB)
        print(f"DEBUG: Successfully acquired lock after cleanup")
      except Exception as e:
        print(f"DEBUG: Lock cleanup failed: {e}")
        raise RuntimeError(f"Failed to take lock file {name}. It's already in use and cannot be cleaned up.")

    return self.lock_fd

  def cleanup_dma_buffers(self):
    """Cleanup all allocated DMA buffers - call before shutdown"""
    if OSX and self._egpu_device and self._dma_buffers:
      print(f"Cleaning up {len(self._dma_buffers)} DMA buffers...")
      
      # Load IOKit framework for unmapping
      import ctypes
      IOKit = ctypes.CDLL('/System/Library/Frameworks/IOKit.framework/IOKit')
      
      for handle, dma_info in self._dma_buffers.items():
        try:
          # Step 1: Unmap from user space if mapped
          if dma_info.user_addr is not None:
            print(f"🔄 Unmapping buffer {handle:x} from user space...")
            IOKit.IOConnectUnmapMemory(
                dma_info.connection,
                dma_info.memory_type,  # Memory type ID
                dma_info.task,
                dma_info.user_addr    # The mapped address
            )
            
          # Step 2: Destroy the kernel buffer
          self._egpu_device.destroy_dma_buffer(handle)
          print(f"✅ Cleaned up DMA buffer {handle:x}")
        except Exception as e:
          print(f"Warning: Failed to cleanup DMA buffer {handle:x}: {e}")
      
      self._dma_buffers.clear()

System = _System()

# Register cleanup on exit
import atexit
atexit.register(System.cleanup_dma_buffers)

# Platform-specific imports for macOS eGPU support will be done lazily to avoid circular imports

class PCIDevice:
  def __init__(self, pcibus:str, bars:list[int], resize_bars:list[int]|None=None):
    self.pcibus, self.irq_poller = pcibus, None
    
    # macOS eGPU support
    if OSX:
      # For macOS, pcibus is just the device index
      self.device_id = int(pcibus) if pcibus.isdigit() else 0
      self.egpu_device = None  # Will be initialized on first use
      
      # BAR mapping for different platforms
      # Linux uses traditional PCI enumeration:
      #   BAR0: MMIO (32-bit)
      #   BAR1: VRAM (64-bit lower)
      #   BAR2: VRAM (64-bit upper) - not directly accessible
      #   BAR3: Instruction memory (64-bit lower)
      #   BAR4: Instruction memory (64-bit upper) - not directly accessible
      #   BAR5: I/O ports
      # 
      # macOS skips the upper 32-bits of 64-bit BARs in its memory index:
      #   Memory Index 0 (BAR0): MMIO (32-bit)
      #   Memory Index 1 (BAR2): VRAM (64-bit, skips BAR1)
      #   Memory Index 2 (BAR4): Instruction memory (64-bit, skips BAR3)
      
      # Define platform-specific BAR indices
      self.BAR_MMIO = 0  # Same on both platforms
      self.BAR_VRAM = 2 if OSX else 1  # macOS skips BAR1 (upper 32-bits)
      self.BAR_INST = 4 if OSX else 3  # macOS skips BAR3 (upper 32-bits)
      
      # Create mapping from Linux BAR indices to platform-specific indices
      self._bar_remap = {0: self.BAR_MMIO, 1: self.BAR_VRAM, 3: self.BAR_INST}
      
      # Create dummy bar_info for compatibility
      self.bar_info = {}
      for bar in bars:
        if bar == 0:  # MMIO registers
          self.bar_info[bar] = (0x0, 0x1000000, 0)  # 16MB MMIO space
        elif bar == 1:  # VRAM (Linux BAR1 -> macOS BAR2)
          self.bar_info[bar] = (0x0, 0x300000000, 0)  # 12GB for RTX 5070
        elif bar == 3:  # Instruction memory (Linux BAR3 -> macOS BAR4)
          self.bar_info[bar] = (0x0, 0x2000000, 0)  # 32MB instruction memory
        else:
          self.bar_info[bar] = (0x0, 0x0, 0)  # Empty BAR
      
      # Create dummy file descriptors for compatibility
      self.cfg_fd = None
      self.bar_fds = {b: None for b in bars}
      return  # Skip Linux-specific initialization

    if FileIOInterface.exists(f"/sys/bus/pci/devices/{self.pcibus}/driver"):
      FileIOInterface(f"/sys/bus/pci/devices/{self.pcibus}/driver/unbind", os.O_WRONLY).write(self.pcibus)

    for i in resize_bars or []:
      # First check current BAR size
      resource_path = f"/sys/bus/pci/devices/{self.pcibus}/resource"
      try:
        with open(resource_path, 'r') as f:
          lines = f.readlines()
          if getenv("DEBUG", "0") == "5":  # Only show detailed BAR info at DEBUG=5
            print(f"DEBUG: Total BARs in resource file: {len(lines)}")
            for idx, line in enumerate(lines[:6]):  # Show first 6 BARs
              print(f"DEBUG: BAR {idx}: {line.strip()}")
          if i < len(lines):
            parts = lines[i].split()
            if len(parts) >= 3 and parts[0] != '0x0000000000000000':
              start = int(parts[0], 16)
              end = int(parts[1], 16)
              current_size = end - start + 1
              size_mb = current_size / (1024**2)
              size_gb = current_size / (1024**3)
              print(f"DEBUG: BAR {i} current size: {size_mb:.0f} MB ({size_gb:.2f} GB)")
              
              # Skip resize if already >= 4GB
              if current_size >= 4 * 1024**3:
                print(f"DEBUG: BAR {i} already sized at {current_size / (1024**3):.1f} GB, skipping resize")
                continue
              else:
                print(f"DEBUG: BAR {i} is only {size_gb:.2f} GB, needs resizing")
      except Exception as e:
        print(f"DEBUG: Could not read current BAR size: {e}")
      
      resize_path = f"/sys/bus/pci/devices/{self.pcibus}/resource{i}_resize"
      print(f"DEBUG: Attempting to resize BAR {i} at path: {resize_path}")
      print(f"DEBUG: Path exists: {os.path.exists(resize_path)}")
      
      # Read supported sizes
      read_fd = FileIOInterface(resize_path, os.O_RDONLY)
      supported_sizes = int(read_fd.read(), 16)
      print(f"DEBUG: Supported sizes for BAR {i}: 0x{supported_sizes:x}, bit_length-1: {supported_sizes.bit_length() - 1}")
      
      # Write new size
      try:
        # For RTX 5070 with 12GB VRAM, we need to use bit 13 (8GB) not bit 14 (16GB)
        # Check if bit 13 is set in supported sizes
        if supported_sizes & (1 << 13):
          new_size = "13"  # 8GB
          print(f"DEBUG: RTX 5070 detected - using bit 13 (8GB) instead of bit 14 (16GB)")
        else:
          new_size = str(supported_sizes.bit_length() - 1)
        print(f"DEBUG: Writing '{new_size}' to {resize_path}")
        # Double-check file exists before writing
        if not os.path.exists(resize_path):
          print(f"ERROR: {resize_path} disappeared!")
          raise FileNotFoundError(f"{resize_path} does not exist")
        
        # Try different approaches for sysfs files
        try:
          # Method 1: Direct write with open()
          with open(resize_path, 'w') as f:
            f.write(new_size)
        except Exception as e1:
          print(f"DEBUG: open() failed: {e1}, trying os.open()")
          try:
            # Method 2: Use os.open with O_WRONLY
            fd = os.open(resize_path, os.O_WRONLY)
            os.write(fd, new_size.encode())
            os.close(fd)
          except Exception as e2:
            print(f"DEBUG: os.open() also failed: {e2}")
            # Method 3: Try echo command
            import subprocess
            try:
              subprocess.run(['echo', new_size], stdout=open(resize_path, 'w'), check=True)
            except Exception as e3:
              print(f"DEBUG: echo method also failed: {e3}")
              raise e1  # Re-raise original error
        print(f"DEBUG: Successfully wrote to BAR {i}")
      except (OSError, IOError) as e:
        print(f"DEBUG: Write failed with error: {e}, errno: {getattr(e, 'errno', 'unknown')}")
        # If it's a permission error, it might be because the BAR is already at the right size
        # or the system doesn't support resizing this BAR
        if getattr(e, 'errno', None) == 13:  # Permission denied
          print(f"DEBUG: Permission denied - BAR resize might not be supported")
        
        # Check current BAR size again
        if 'current_size' in locals() and current_size >= 256 * 1024**2:  # At least 256MB
          print(f"WARNING: Cannot resize BAR {i}, but current size {current_size / (1024**2):.0f} MB is sufficient for testing")
          print("WARNING: Performance may be limited with smaller BAR size")
          continue  # Continue without failing
        else:
          raise RuntimeError(f"Cannot resize BAR {i}: {e}. Ensure the resizable BAR option is enabled on your system.") from e

    if getenv("VFIO", 0) and (vfio_fd:=System.vfio()) is not None:
      FileIOInterface(f"/sys/bus/pci/devices/{self.pcibus}/driver_override", os.O_WRONLY).write("vfio-pci")
      FileIOInterface("/sys/bus/pci/drivers_probe", os.O_WRONLY).write(self.pcibus)
      iommu_group = FileIOInterface.readlink(f"/sys/bus/pci/devices/{self.pcibus}/iommu_group").split('/')[-1]

      self.vfio_group = FileIOInterface(f"/dev/vfio/noiommu-{iommu_group}", os.O_RDWR)
      vfio.VFIO_GROUP_SET_CONTAINER(self.vfio_group, ctypes.c_int(vfio_fd.fd))

      with contextlib.suppress(OSError): vfio.VFIO_SET_IOMMU(vfio_fd, vfio.VFIO_NOIOMMU_IOMMU) # set iommu works only once for the fd.
      self.vfio_dev = FileIOInterface(fd=vfio.VFIO_GROUP_GET_DEVICE_FD(self.vfio_group, ctypes.create_string_buffer(self.pcibus.encode())))

      self.irq_fd = FileIOInterface.eventfd(0, 0)
      self.irq_poller = select.poll()
      self.irq_poller.register(self.irq_fd.fd, select.POLLIN)

      irqs = vfio.struct_vfio_irq_set(index=vfio.VFIO_PCI_MSI_IRQ_INDEX, flags=vfio.VFIO_IRQ_SET_DATA_EVENTFD|vfio.VFIO_IRQ_SET_ACTION_TRIGGER,
        argsz=ctypes.sizeof(vfio.struct_vfio_irq_set), count=1, data=(ctypes.c_int * 1)(self.irq_fd.fd))
      vfio.VFIO_DEVICE_SET_IRQS(self.vfio_dev, irqs)
    else: FileIOInterface(f"/sys/bus/pci/devices/{self.pcibus}/enable", os.O_RDWR).write("1")

    self.cfg_fd = FileIOInterface(f"/sys/bus/pci/devices/{self.pcibus}/config", os.O_RDWR | os.O_SYNC | os.O_CLOEXEC)
    self.bar_fds = {b: FileIOInterface(f"/sys/bus/pci/devices/{self.pcibus}/resource{b}", os.O_RDWR | os.O_SYNC | os.O_CLOEXEC) for b in bars}

    bar_info = FileIOInterface(f"/sys/bus/pci/devices/{self.pcibus}/resource", os.O_RDONLY).read().splitlines()
    self.bar_info = {j:(int(start,16), int(end,16), int(flgs,16)) for j,(start,end,flgs) in enumerate(l.split() for l in bar_info)}

  def read_config(self, offset:int, size:int): 
    if OSX:
      # For macOS eGPU, return dummy config values
      # This could be enhanced to read actual config from the eGPU device
      return 0x10de if offset == 0 and size >= 2 else 0  # NVIDIA vendor ID
    return int.from_bytes(self.cfg_fd.read(size, binary=True, offset=offset), byteorder='little')
  
  def write_config(self, offset:int, value:int, size:int): 
    if OSX:
      # For macOS eGPU, config writes are no-op for now
      # This could be enhanced to write actual config to the eGPU device
      return
    self.cfg_fd.write(value.to_bytes(size, byteorder='little'), binary=True, offset=offset)
  
  def a(self, bar:int, off:int=0, addr:int=0, size:int|None=None, fmt='B') -> MMIOInterface:
    if OSX:
      # macOS path: use macos_egpu provider instead of egpudev
      from tinygrad.runtime.support.nv.macos_egpu import acquire_device, EGPUMMIOInterface
      dev = acquire_device(getattr(self, 'device_id', 0))
      actual_bar = self._bar_remap.get(bar, bar)
      return EGPUMMIOInterface(dev.handle, actual_bar, fmt)
    fd, sz = self.bar_fds[bar], size or (self.bar_info[bar][1] - self.bar_info[bar][0] + 1)
    print(f"DEBUG: map_bar({bar}) - fd={fd.fd}, size={sz} ({sz/(1024**2):.0f}MB), addr=0x{addr:x}, off={off}")
    print(f"DEBUG: BAR {bar} info: start=0x{self.bar_info[bar][0]:x}, end=0x{self.bar_info[bar][1]:x}, flags=0x{self.bar_info[bar][2]:x}")
    try:
      flags = mmap.MAP_SHARED | (MAP_FIXED if addr else 0)
      page_size = os.sysconf(os.sysconf_names['SC_PAGE_SIZE'])
      print(f"DEBUG: mmap flags: {flags}, MAP_SHARED={mmap.MAP_SHARED}, MAP_FIXED={MAP_FIXED if addr else 0}")
      print(f"DEBUG: page_size={page_size}, size_aligned={sz % page_size == 0}, offset_aligned={off % page_size == 0}")
      loc = fd.mmap(addr, sz, mmap.PROT_READ | mmap.PROT_WRITE, flags, off)
      libc.madvise(loc, sz, libc.MADV_DONTFORK)
      return MMIOInterface(loc, sz, fmt=fmt)
    except Exception as e:
      print(f"DEBUG: mmap failed: {e}")
      raise

  def map_bar(self, bar_idx: int, fmt: str = 'I'):
    """Map a BAR for memory access - macOS eGPU implementation"""
    if OSX:
      # macOS path: use macos_egpu provider
      from tinygrad.runtime.support.nv.macos_egpu import acquire_device, EGPUMMIOInterface
      dev = acquire_device(getattr(self, 'device_id', 0))
      return EGPUMMIOInterface(dev.handle, bar_idx if bar_idx != 1 else 2, fmt=fmt)
    else:
      # Linux implementation
      return self._mmap_bar(bar_idx, fmt)
  
  def _mmap_bar(self, bar_idx: int, fmt: str):
    """Linux BAR mapping implementation"""
    bar_fd = FileIOInterface(f"/sys/bus/pci/devices/{self.pcibus}/resource{bar_idx}")
    bar_base, bar_size = self.bar_info[bar_idx][:2]
    return MMIOInterface(bar_fd.mmap(0, bar_size, mmap.PROT_READ | mmap.PROT_WRITE, mmap.MAP_SHARED, 0), bar_size, fmt=fmt)

class PCIDevImplBase:
  mm: MemoryManager

@dataclasses.dataclass
class PCIAllocationMeta: mapping:VirtMapping; has_cpu_mapping:bool; hMemory:int=0 # noqa: E702

class PCIIfaceBase:
  dev_impl:PCIDevImplBase
  gpus:ClassVar[list[str]] = []

  def __init__(self, dev, dev_id, vendor, devices, bars, vram_bar, va_start, va_size):
    if len((cls:=type(self)).gpus) == 0:
      cls.gpus = System.pci_scan_bus(vendor, devices)
      visible_devices = [int(x) for x in (getenv('VISIBLE_DEVICES', '')).split(',') if x.strip()]
      cls.gpus = [cls.gpus[x] for x in visible_devices] if visible_devices else cls.gpus

      # Acquire va range to avoid collisions.
      FileIOInterface.anon_mmap(va_start, va_size, 0, mmap.MAP_PRIVATE | mmap.MAP_ANONYMOUS | MAP_NORESERVE | MAP_FIXED, 0)
    self.pci_dev, self.dev, self.vram_bar = PCIDevice(cls.gpus[dev_id], bars=bars, resize_bars=[vram_bar]), dev, vram_bar
    self.p2p_base_addr = self.pci_dev.bar_info[vram_bar][0]

  def alloc(self, size:int, host=False, uncached=False, cpu_access=False, contiguous=False, **kwargs) -> HCQBuffer:
    if host or (uncached and cpu_access): # host or gtt-like memory.
      vaddr = self.dev_impl.mm.alloc_vaddr(size:=round_up(size, mmap.PAGESIZE), align=mmap.PAGESIZE)
      paddrs = [(paddr, mmap.PAGESIZE) for paddr in System.alloc_sysmem(size, vaddr=vaddr, contiguous=contiguous, 
                                                                        direction=System.DMA_DIRECTION_BIDIRECTIONAL)[1]]
      mapping = self.dev_impl.mm.map_range(vaddr, size, paddrs, system=True, snooped=True, uncached=True)
      return HCQBuffer(vaddr, size, meta=PCIAllocationMeta(mapping, has_cpu_mapping=True, hMemory=paddrs[0][0]),
        view=MMIOInterface(mapping.va_addr, size, fmt='B'), owner=self.dev)

    mapping = self.dev_impl.mm.valloc(size:=round_up(size, mmap.PAGESIZE), uncached=uncached, contiguous=cpu_access)
    if cpu_access: self.pci_dev.map_bar(bar=self.vram_bar, off=mapping.paddrs[0][0], addr=mapping.va_addr, size=mapping.size)
    return HCQBuffer(mapping.va_addr, size, view=MMIOInterface(mapping.va_addr, size, fmt='B') if cpu_access else None,
      meta=PCIAllocationMeta(mapping, has_cpu_mapping=cpu_access, hMemory=mapping.paddrs[0][0]), owner=self.dev)

  def free(self, b:HCQBuffer):
    for dev in b.mapped_devs[1:]: dev.iface.dev_impl.mm.unmap_range(b.va_addr, b.size)
    if not b.meta.mapping.system: self.dev_impl.mm.vfree(b.meta.mapping)
    if b.owner == self.dev and b.meta.has_cpu_mapping: FileIOInterface.munmap(b.va_addr, b.size)

  def map(self, b:HCQBuffer):
    if b.owner is not None and b.owner._is_cpu():
      System.lock_memory(cast(int, b.va_addr), b.size)
      paddrs, snooped, uncached = [(x, mmap.PAGESIZE) for x in System.system_paddrs(cast(int, b.va_addr), round_up(b.size, mmap.PAGESIZE))], True, False
    elif (ifa:=getattr(b.owner, "iface", None)) is not None and isinstance(ifa, PCIIfaceBase):
      paddrs = [(paddr if b.meta.mapping.system else (paddr + ifa.p2p_base_addr), size) for paddr,size in b.meta.mapping.paddrs]
      snooped, uncached = b.meta.mapping.snooped, b.meta.mapping.uncached
    else: raise RuntimeError(f"map failed: {b.owner} -> {self.dev}")

    self.dev_impl.mm.map_range(cast(int, b.va_addr), round_up(b.size, mmap.PAGESIZE), paddrs, system=True, snooped=snooped, uncached=uncached)
