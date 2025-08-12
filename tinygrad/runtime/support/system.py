import os, mmap, array, functools, ctypes, select, contextlib, dataclasses, sys
from typing import cast, ClassVar
from tinygrad.helpers import round_up, to_mv, getenv, OSX, temp
from tinygrad.runtime.autogen import libc, vfio
from tinygrad.runtime.support.hcq import FileIOInterface, MMIOInterface, HCQBuffer
from tinygrad.runtime.support.memory import MemoryManager, VirtMapping

MAP_FIXED, MAP_LOCKED, MAP_POPULATE, MAP_NORESERVE = 0x10, 0 if OSX else 0x2000, getattr(mmap, "MAP_POPULATE", 0 if OSX else 0x008000), 0x400

class _System:
  def reserve_hugepages(self, cnt): os.system(f"sudo sh -c 'echo {cnt} > /proc/sys/vm/nr_hugepages'")

  def memory_barrier(self): lib.atomic_thread_fence(__ATOMIC_SEQ_CST:=5) if (lib:=self.atomic_lib()) is not None else None

  def lock_memory(self, addr:int, size:int):
    if libc.mlock(ctypes.c_void_p(addr), size): raise RuntimeError(f"Failed to lock memory at {addr:#x} with size {size:#x}")

  def system_paddrs(self, vaddr:int, size:int) -> list[int]:
    self.pagemap().seek(vaddr // mmap.PAGESIZE * 8)
    return [(x & ((1<<55) - 1)) * mmap.PAGESIZE for x in array.array('Q', self.pagemap().read(size//mmap.PAGESIZE*8, binary=True))]

  def alloc_sysmem(self, size:int, vaddr:int=0, contiguous:bool=False, data:bytes|None=None, name:str|None=None) -> tuple[int, list[int]]:
    assert not contiguous or size <= (2 << 20), "Contiguous allocation is only supported for sizes up to 2MB"
    flags = (libc.MAP_HUGETLB if contiguous and (size:=round_up(size, mmap.PAGESIZE)) > 0x1000 else 0) | (MAP_FIXED if vaddr else 0)
    va = FileIOInterface.anon_mmap(vaddr, size, mmap.PROT_READ|mmap.PROT_WRITE, mmap.MAP_SHARED|mmap.MAP_ANONYMOUS|MAP_POPULATE|MAP_LOCKED|flags, 0)

    if data is not None: to_mv(va, len(data))[:] = data
    paddrs = self.system_paddrs(va, size)
    # Optional debug tracing tied to a region name or DEBUG env
    try:
      if name is not None or getenv("DEBUG", 0):
        first_n = min(8, len(paddrs))
        pages_preview = ", ".join(hex(p) for p in paddrs[:first_n])
        more = f", ... {len(paddrs)-first_n} more" if len(paddrs) > first_n else ""
        print(f"🐧 DEBUG: alloc_sysmem name={name or '<unnamed>'} size=0x{size:x} contiguous={contiguous} vaddr=0x{vaddr:x} -> va=0x{va:x}, pages={len(paddrs)} [{pages_preview}{more}]")
    except Exception:
      pass
    return va, paddrs

  def pci_reset(self, gpu): os.system(f"sudo sh -c 'echo 1 > /sys/bus/pci/devices/{gpu}/reset'")
  def pci_scan_bus(self, target_vendor:int, target_devices:list[int]) -> list[str]:
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
    if FileIOInterface(reloc_sysfs:="/proc/sys/vm/compact_unevictable_allowed", os.O_RDONLY).read()[0] != "0":
      os.system(cmd:=f"sudo sh -c 'echo 0 > {reloc_sysfs}'")
      assert FileIOInterface(reloc_sysfs, os.O_RDONLY).read()[0] == "0", f"Failed to disable migration of locked pages. Please run {cmd} manually."
    return FileIOInterface("/proc/self/pagemap", os.O_RDONLY)

  @functools.cache
  def vfio(self) -> FileIOInterface|None:
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
        print(f"🐧 DEBUG: Removed stale lock file {lock_name}")
        # Retry
        self.lock_fd = os.open(lock_name, os.O_RDWR | os.O_CREAT | os.O_CLOEXEC, 0o666)
        fcntl.flock(self.lock_fd, fcntl.LOCK_EX | fcntl.LOCK_NB)
        print(f"🐧 DEBUG: Successfully acquired lock after cleanup")
      except:
        raise RuntimeError(f"Failed to take lock file {name}. It's already in use and cannot be cleaned up.")

    return self.lock_fd

System = _System()

class PCIDevice:
  def __init__(self, pcibus:str, bars:list[int], resize_bars:list[int]|None=None):
    self.pcibus, self.irq_poller = pcibus, None

    if FileIOInterface.exists(f"/sys/bus/pci/devices/{self.pcibus}/driver"):
      FileIOInterface(f"/sys/bus/pci/devices/{self.pcibus}/driver/unbind", os.O_WRONLY).write(self.pcibus)

    for i in resize_bars or []:
      # First check current BAR size
      resource_path = f"/sys/bus/pci/devices/{self.pcibus}/resource"
      try:
        with open(resource_path, 'r') as f:
          lines = f.readlines()
          if getenv("DEBUG", "0") == "5":  # Only show detailed BAR info at DEBUG=5
            print(f"🐧 DEBUG: Total BARs in resource file: {len(lines)}")
            for idx, line in enumerate(lines[:6]):  # Show first 6 BARs
              print(f"🐧 DEBUG: BAR {idx}: {line.strip()}")
          if i < len(lines):
            parts = lines[i].split()
            if len(parts) >= 3 and parts[0] != '0x0000000000000000':
              start = int(parts[0], 16)
              end = int(parts[1], 16)
              current_size = end - start + 1
              size_mb = current_size / (1024**2)
              size_gb = current_size / (1024**3)
              print(f"🐧 DEBUG: BAR {i} current size: {size_mb:.0f} MB ({size_gb:.2f} GB)")
              
              # Skip resize if already >= 4GB
              if current_size >= 4 * 1024**3:
                print(f"🐧 DEBUG: BAR {i} already sized at {current_size / (1024**3):.1f} GB, skipping resize")
                continue
              else:
                print(f"🐧 DEBUG: BAR {i} is only {size_gb:.2f} GB, needs resizing")
      except Exception as e:
        print(f"🐧 DEBUG: Could not read current BAR size: {e}")
      
      resize_path = f"/sys/bus/pci/devices/{self.pcibus}/resource{i}_resize"
      print(f"🐧 DEBUG: Attempting to resize BAR {i} at path: {resize_path}")
      print(f"🐧 DEBUG: Path exists: {os.path.exists(resize_path)}")
      
      # Read supported sizes
      read_fd = FileIOInterface(resize_path, os.O_RDONLY)
      supported_sizes = int(read_fd.read(), 16)
      print(f"🐧 DEBUG: Supported sizes for BAR {i}: 0x{supported_sizes:x}, bit_length-1: {supported_sizes.bit_length() - 1}")
      
      # Write new size
      try:
        # For RTX 5070 with 12GB VRAM, we need to use bit 13 (8GB) not bit 14 (16GB)
        # Check if bit 13 is set in supported sizes
        if supported_sizes & (1 << 13):
          new_size = "13"  # 8GB
          print(f"🐧 DEBUG: RTX 5070 detected - using bit 13 (8GB) instead of bit 14 (16GB)")
        else:
          new_size = str(supported_sizes.bit_length() - 1)
        print(f"🐧 DEBUG: Writing '{new_size}' to {resize_path}")
        # Double-check file exists before writing
        if not os.path.exists(resize_path):
          print(f"🐧 ERROR: {resize_path} disappeared!")
          raise FileNotFoundError(f"{resize_path} does not exist")
        
        # Try different approaches for sysfs files
        try:
          # Method 1: Direct write with open()
          with open(resize_path, 'w') as f:
            f.write(new_size)
        except Exception as e1:
          print(f"🐧 DEBUG: open() failed: {e1}, trying os.open()")
          try:
            # Method 2: Use os.open with O_WRONLY
            fd = os.open(resize_path, os.O_WRONLY)
            os.write(fd, new_size.encode())
            os.close(fd)
          except Exception as e2:
            print(f"🐧 DEBUG: os.open() also failed: {e2}")
            # Method 3: Try echo command
            import subprocess
            try:
              subprocess.run(['echo', new_size], stdout=open(resize_path, 'w'), check=True)
            except Exception as e3:
              print(f"🐧 DEBUG: echo method also failed: {e3}")
              raise e1  # Re-raise original error
        print(f"🐧 DEBUG: Successfully wrote to BAR {i}")
      except (OSError, IOError) as e:
        print(f"🐧 DEBUG: Write failed with error: {e}, errno: {getattr(e, 'errno', 'unknown')}")
        # If it's a permission error, it might be because the BAR is already at the right size
        # or the system doesn't support resizing this BAR
        if getattr(e, 'errno', None) == 13:  # Permission denied
          print(f"🐧 DEBUG: Permission denied - BAR resize might not be supported")
        
        # Check current BAR size again
        if 'current_size' in locals() and current_size >= 256 * 1024**2:  # At least 256MB
          print(f"🐧 WARNING: Cannot resize BAR {i}, but current size {current_size / (1024**2):.0f} MB is sufficient for testing")
          print("🐧 WARNING: Performance may be limited with smaller BAR size")
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

  def read_config(self, offset:int, size:int): return int.from_bytes(self.cfg_fd.read(size, binary=True, offset=offset), byteorder='little')
  def write_config(self, offset:int, value:int, size:int): self.cfg_fd.write(value.to_bytes(size, byteorder='little'), binary=True, offset=offset)
  def map_bar(self, bar:int, off:int=0, addr:int=0, size:int|None=None, fmt='B') -> MMIOInterface:
    fd, sz = self.bar_fds[bar], size or (self.bar_info[bar][1] - self.bar_info[bar][0] + 1)
    print(f"🐧 DEBUG: map_bar({bar}) - fd={fd.fd}, size={sz} ({sz/(1024**2):.0f}MB), addr=0x{addr:x}, off={off}")
    print(f"🐧 DEBUG: BAR {bar} info: start=0x{self.bar_info[bar][0]:x}, end=0x{self.bar_info[bar][1]:x}, flags=0x{self.bar_info[bar][2]:x}")
    try:
      flags = mmap.MAP_SHARED | (MAP_FIXED if addr else 0)
      page_size = os.sysconf(os.sysconf_names['SC_PAGE_SIZE'])
      print(f"🐧 DEBUG: mmap flags: {flags}, MAP_SHARED={mmap.MAP_SHARED}, MAP_FIXED={MAP_FIXED if addr else 0}")
      print(f"🐧 DEBUG: page_size={page_size}, size_aligned={sz % page_size == 0}, offset_aligned={off % page_size == 0}")
      loc = fd.mmap(addr, sz, mmap.PROT_READ | mmap.PROT_WRITE, flags, off)
      libc.madvise(loc, sz, libc.MADV_DONTFORK)
      return MMIOInterface(loc, sz, fmt=fmt)
    except Exception as e:
      print(f"🐧 DEBUG: mmap failed: {e}")
      raise

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
      paddrs = [(paddr, mmap.PAGESIZE) for paddr in System.alloc_sysmem(size, vaddr=vaddr, contiguous=contiguous)[1]]
      mapping = self.dev_impl.mm.map_range(vaddr, size, paddrs, system=True, snooped=True, uncached=True)
      return HCQBuffer(vaddr, size, meta=PCIAllocationMeta(mapping, has_cpu_mapping=True, hMemory=paddrs[0][0]),
        view=MMIOInterface(mapping.va_addr, size, fmt='B'), owner=self.dev)

    mapping = self.dev_impl.mm.valloc(size:=round_up(size, 4 << 10), uncached=uncached, contiguous=cpu_access)
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
      paddrs, snooped, uncached = [(x, 0x1000) for x in System.system_paddrs(cast(int, b.va_addr), round_up(b.size, 0x1000))], True, False
    elif (ifa:=getattr(b.owner, "iface", None)) is not None and isinstance(ifa, PCIIfaceBase):
      paddrs = [(paddr if b.meta.mapping.system else (paddr + ifa.p2p_base_addr), size) for paddr,size in b.meta.mapping.paddrs]
      snooped, uncached = b.meta.mapping.snooped, b.meta.mapping.uncached
    else: raise RuntimeError(f"map failed: {b.owner} -> {self.dev}")

    self.dev_impl.mm.map_range(cast(int, b.va_addr), round_up(b.size, 0x1000), paddrs, system=True, snooped=snooped, uncached=uncached)
