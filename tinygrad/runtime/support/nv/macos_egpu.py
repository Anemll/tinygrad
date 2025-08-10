from __future__ import annotations
import os, struct
from ctypes import CDLL, c_void_p, c_int, c_uint64, c_uint32, c_uint16, POINTER, Structure, byref, c_char_p, create_string_buffer, cast
from tinygrad.helpers import getenv, DEBUG
from tinygrad.runtime.support.hcq import MMIOInterface

NV_DEBUG = getenv("NV_DEBUG", 0)

# DMA on macOS (DriverKit/Apple‑silicon) quick notes:
# - Device DMA uses IOVA (IOMMU/DART). Do not program CPU virtual or host physical addresses into HW.
# - The driver returns IOVA segments. Use those for hardware, and use the returned virtual address only for memcpy.
# - max_pairs controls how many scatter/gather entries you request; actual returned count may be less.
# - segment_count is the number of IOVA segments returned; iova_4k_pages is total 4K pages across all segments.

# Optional path override via env
LIBEGPU_PATH = os.path.expanduser(os.environ.get('LIBEGPU_PATH', '~/SourceRelease/GITHUB/eGPU/eGPU/EGPUMapperDriver/libegpu_pcidevice.dylib'))

class _EgpuDeviceInfo(Structure):
  _fields_ = [
    ("vendor_id", c_uint16),
    ("device_id", c_uint16),
    ("bar_count", c_uint32),
    ("bar_sizes", c_uint64 * 6),
    ("bar_bases", c_uint64 * 6),
  ]

def _load_lib() -> CDLL|None:
  try:
    if NV_DEBUG >= 1: print(f"macos_egpu: loading {LIBEGPU_PATH}")
    return CDLL(LIBEGPU_PATH)
  except OSError:
    for alt in [
      "/usr/local/lib/libegpu_pcidevice.dylib",
      os.path.expanduser("~/lib/libegpu_pcidevice.dylib"),
      os.path.join(os.path.dirname(__file__), "libegpu_pcidevice.dylib"),
    ]:
      try:
        if NV_DEBUG >= 1: print(f"macos_egpu: trying {alt}")
        return CDLL(alt)
      except OSError:
        continue
  return None

libegpu = _load_lib()

if libegpu is not None:
  libegpu.egpu_device_create.restype = c_void_p
  libegpu.egpu_device_create.argtypes = [c_int, POINTER(c_int), c_int]
  libegpu.egpu_device_destroy.restype = None
  libegpu.egpu_device_destroy.argtypes = [c_void_p]
  libegpu.egpu_device_get_info.restype = POINTER(_EgpuDeviceInfo)
  libegpu.egpu_device_get_info.argtypes = [c_void_p]
  libegpu.egpu_device_map_bar.restype = c_void_p
  libegpu.egpu_device_map_bar.argtypes = [c_void_p, c_int]
  libegpu.egpu_mmio_read32.restype = c_uint32
  libegpu.egpu_mmio_read32.argtypes = [c_void_p, c_uint64]
  libegpu.egpu_mmio_write32.restype = None
  libegpu.egpu_mmio_write32.argtypes = [c_void_p, c_uint64, c_uint32]
  libegpu.egpu_mmio_read8.restype = c_uint32  # return promoted to 32
  libegpu.egpu_mmio_read8.argtypes = [c_void_p, c_uint64]
  libegpu.egpu_mmio_write8.restype = None
  libegpu.egpu_mmio_write8.argtypes = [c_void_p, c_uint64, c_uint32]
  libegpu.egpu_mmio_memory_barrier.restype = None
  libegpu.egpu_mmio_memory_barrier.argtypes = [c_void_p]
  libegpu.egpu_device_get_connection.restype = c_uint32
  libegpu.egpu_device_get_connection.argtypes = [c_void_p]
  # DMA APIs
  libegpu.egpu_device_allocate_dma_buffer.restype = c_uint64
  libegpu.egpu_device_allocate_dma_buffer.argtypes = [c_void_p, c_uint64, c_uint32, POINTER(c_uint64), POINTER(c_uint64)]
  # with-data variant (contiguous): returns handle (uint64), fills phys/virt via out ptrs
  try:
    libegpu.egpu_device_allocate_dma_buffer_with_data
    libegpu.egpu_device_allocate_dma_buffer_with_data.restype = c_uint64
    # args: device, size, direction, data_ptr, data_len, out_phys, out_virt
    libegpu.egpu_device_allocate_dma_buffer_with_data.argtypes = [
      c_void_p, c_uint64, c_uint32, c_void_p, c_uint64, POINTER(c_uint64), POINTER(c_uint64)
    ]
  except AttributeError:
    pass
  libegpu.egpu_device_destroy_dma_buffer.restype = c_int
  libegpu.egpu_device_destroy_dma_buffer.argtypes = [c_void_p, c_uint64]
  libegpu.egpu_device_get_memory_type_for_handle.restype = c_uint32
  libegpu.egpu_device_get_memory_type_for_handle.argtypes = [c_void_p, c_uint64]
  libegpu.egpu_device_copy_dma_buffers.restype = c_int
  libegpu.egpu_device_copy_dma_buffers.argtypes = [c_void_p, c_uint64, c_uint64, c_uint64, c_uint64]
  # Segmented DMA allocation API (returns count, handle, virtual, then pairs)
  # C++ signature:
  # bool allocate_dma_buffer_segmented(uint64_t size, uint32_t direction, uint32_t max_pairs,
  #                                    uint32_t* out_count, std::vector<pair<uint64_t,uint64_t>>& out_segments,
  #                                    uint64_t* handle, uint64_t* virtual_addr)
  try:
    libegpu.egpu_device_allocate_dma_buffer_segmented
    libegpu.egpu_device_allocate_dma_buffer_segmented.restype = c_int
    # args: device, size, direction, max_pairs, out_count, out_addrs, out_lens, out_handle, out_virtual
    libegpu.egpu_device_allocate_dma_buffer_segmented.argtypes = [c_void_p, c_uint64, c_uint32, c_uint32,
                                                                  POINTER(c_uint32), POINTER(c_uint64), POINTER(c_uint64),
                                                                  POINTER(c_uint64), POINTER(c_uint64)]
  except AttributeError:
    pass

  # With-data DMA allocation APIs (segmented): status return, fills outputs
  try:
    libegpu.egpu_device_allocate_dma_buffer_segmented_with_data
    libegpu.egpu_device_allocate_dma_buffer_segmented_with_data.restype = c_int
    # args: device, size, direction, data_ptr, data_len, max_pairs, out_count, out_addrs, out_lens, out_handle, out_virtual
    libegpu.egpu_device_allocate_dma_buffer_segmented_with_data.argtypes = [
      c_void_p, c_uint64, c_uint32, c_void_p, c_uint64, c_uint32,
      POINTER(c_uint32), POINTER(c_uint64), POINTER(c_uint64), POINTER(c_uint64), POINTER(c_uint64)
    ]
  except AttributeError:
    pass

class EGPUMMIOInterface(MMIOInterface):
  def __init__(self, device_handle: c_void_p, bar_index: int, fmt: str = 'I', base_offset: int = 0, size_bytes: int | None = None):
    if libegpu is None: raise RuntimeError("libegpu_pcidevice.dylib not available")
    self._fmt = fmt
    self._if = libegpu.egpu_device_map_bar(device_handle, c_int(bar_index))
    if not self._if: raise RuntimeError(f"Failed to map BAR{bar_index}")
    self._elt = struct.calcsize(fmt)
    self._base = base_offset
    self._size_bytes = size_bytes or (64 * 1024 * 1024)
    if DEBUG >= 2: print(f"macos_egpu: mapped BAR{bar_index} -> iface=0x{int(self._if):x}, base={self._base}, fmt={self._fmt}")

  def __len__(self):
    return self._size_bytes // self._elt

  def __getitem__(self, idx):
    if isinstance(idx, slice):
      start, stop, step = idx.indices(len(self))
      if step != 1: raise NotImplementedError("Step slicing not supported")
      length = stop - start
      if self._fmt == 'B':
        # Byte-granular read
        return bytes(int(libegpu.egpu_mmio_read8(self._if, c_uint64(self._base + start + off))) for off in range(length))
      elif self._fmt == 'I':
        # 32-bit reads, return list of uint32
        words = length // 4
        return [int(libegpu.egpu_mmio_read32(self._if, c_uint64(self._base + (start//4 + i) * 4))) for i in range(words)]
      else:
        raise NotImplementedError(self._fmt)
    # Single element
    off = self._base + idx * self._elt
    if self._fmt == 'I': return int(libegpu.egpu_mmio_read32(self._if, c_uint64(off)))
    if self._fmt == 'B': return int(libegpu.egpu_mmio_read8(self._if, c_uint64(off)))
    raise NotImplementedError(self._fmt)

  def __setitem__(self, idx, val):
    if isinstance(idx, slice):
      start, stop, step = idx.indices(len(self))
      if step != 1: raise NotImplementedError("Step slicing not supported")
      length = stop - start
      # Support bytes/bytearray for 'B' format
      if self._fmt == 'B':
        if isinstance(val, (bytes, bytearray)):
          for i in range(length):
            b = val[i] if i < len(val) else 0
            libegpu.egpu_mmio_write8(self._if, c_uint64(self._base + start + i), c_uint32(b))
        else:
          # Assume iterable of ints
          for i, v in enumerate(val):
            if i >= length: break
            libegpu.egpu_mmio_write8(self._if, c_uint64(self._base + start + i), c_uint32(int(v) & 0xff))
        libegpu.egpu_mmio_memory_barrier(self._if)
        return
      elif self._fmt == 'I':
        # Expect iterable of uint32 or bytes length multiple of 4
        if isinstance(val, (bytes, bytearray)):
          for i in range(0, length, 4):
            chunk = val[i:i+4]
            if len(chunk) < 4: chunk = chunk + b"\x00"*(4-len(chunk))
            word = int.from_bytes(chunk, 'little')
            libegpu.egpu_mmio_write32(self._if, c_uint64(self._base + start + i), c_uint32(word))
        else:
          for i, v in enumerate(val):
            if (i*4) >= length: break
            libegpu.egpu_mmio_write32(self._if, c_uint64(self._base + start + i*4), c_uint32(int(v)))
        libegpu.egpu_mmio_memory_barrier(self._if)
        return
      else:
        raise NotImplementedError(self._fmt)
    # Single element
    off = self._base + idx * self._elt
    if self._fmt == 'I': libegpu.egpu_mmio_write32(self._if, c_uint64(off), c_uint32(val))
    elif self._fmt == 'B': libegpu.egpu_mmio_write8(self._if, c_uint64(off), c_uint32(val & 0xff))
    else: raise NotImplementedError(self._fmt)
    libegpu.egpu_mmio_memory_barrier(self._if)

  def view(self, offset: int = 0, size: int | None = None, fmt: str | None = None):
    # Return a shallow wrapper with adjusted base/size/format
    return EGPUMMIOInterface.__new_with_existing(self._if, fmt or self._fmt, self._base + offset, size if size is not None else self._size_bytes - offset)

  @classmethod
  def __new_with_existing(cls, mmio_if: c_void_p, fmt: str, base_offset: int, size_bytes: int):
    obj = object.__new__(cls)
    obj._if = mmio_if
    obj._fmt = fmt
    obj._elt = struct.calcsize(fmt)
    obj._base = base_offset
    obj._size_bytes = size_bytes
    return obj

def open_device(device_id: int = 0):
  if libegpu is None:
    raise RuntimeError("libegpu_pcidevice.dylib not available")

  bars_req = (c_int * 3)(0, 2, 4)  # BAR0 regs, BAR2 VRAM, BAR4 inst (if present)
  dev = libegpu.egpu_device_create(device_id, bars_req, 3)
  if not dev: raise RuntimeError(f"Failed to open eGPU device {device_id}")

  info_ptr = libegpu.egpu_device_get_info(dev)
  if not info_ptr: raise RuntimeError("Failed to query eGPU device info")
  info = info_ptr.contents

  # Build interfaces
  mmio = EGPUMMIOInterface(dev, 0, fmt='I')
  vram = EGPUMMIOInterface(dev, 2, fmt='I')

  # Build bars dict as (start, end, flags) for NVDev
  bars: dict[int, tuple[int, int, int]] = {}
  for i in range(int(info.bar_count)):
    size = int(info.bar_sizes[i])
    base = int(info.bar_bases[i])
    if size > 0:
      bars[i] = (base, base + size - 1, 0)

  if DEBUG >= 1:
    human = {i: (hex(bars[i][0]), f"{(bars[i][1]-bars[i][0]+1)/(1024**3):.2f} GiB") for i in bars}
    print(f"macos_egpu: bars={human}")

  venid = int(info.vendor_id)
  subvenid = 0
  rev = 0
  return mmio, vram, venid, subvenid, rev, bars


class MacOSEGPUDevice:
  def __init__(self, device_id: int = 0):
    if libegpu is None: raise RuntimeError("libegpu_pcidevice.dylib not available")
    req = (c_int * 3)(0, 2, 4)
    self.handle = libegpu.egpu_device_create(device_id, req, 3)
    if not self.handle: raise RuntimeError(f"Failed to open eGPU device {device_id}")
    info_ptr = libegpu.egpu_device_get_info(self.handle)
    if not info_ptr: raise RuntimeError("Failed to query eGPU device info")
    self.info = info_ptr.contents
    # Precompute bars dict
    self.bars = {}
    for i in range(int(self.info.bar_count)):
      size = int(self.info.bar_sizes[i]); base = int(self.info.bar_bases[i])
      if size > 0: self.bars[i] = (base, base + size - 1, 0)

  def mmio_if(self, bar_index: int, fmt: str = 'I') -> EGPUMMIOInterface:
    return EGPUMMIOInterface(self.handle, bar_index, fmt)

  def get_connection(self) -> int:
    return int(libegpu.egpu_device_get_connection(self.handle))

  def allocate_dma_buffer(self, size: int, direction: int) -> dict:
    """
    Allocate a DMA buffer (single segment expected) and return:
      - handle: opaque buffer handle
      - size: requested size
      - physical_addr: device DMA address (IOVA) to program into hardware
      - virtual_addr: CPU pointer in the driver process (diagnostic); for memcpy in user space,
                      map the buffer with IOConnectMapMemory using handle/memory type.
    Notes:
      - On macOS DriverKit/Apple‑silicon, physical_addr here is an IOVA, not host physical or CPU virtual.
      - If a single segment cannot be guaranteed, use allocate_dma_buffer_segmented.
    """
    phys = c_uint64(0); virt = c_uint64(0)
    h = int(libegpu.egpu_device_allocate_dma_buffer(self.handle, c_uint64(size), c_uint32(direction), byref(phys), byref(virt)))
    return {'handle': h, 'size': size, 'physical_addr': int(phys.value), 'virtual_addr': int(virt.value)} if h != 0 else None

  def allocate_dma_buffer_with_data(self, data: bytes, direction: int) -> dict | None:
    """
    Allocate a DMA buffer sized to len(data), prefill it with the given bytes, and return the same
    fields as allocate_dma_buffer. Fallbacks are layered to try best-effort prefilling:
      1) Use direct with-data contiguous allocation if exported by the dylib.
      2) If unavailable, allocate a segmented-with-data staging buffer, allocate a final contiguous buffer,
         copy the bytes using the driver's copy API, and free the staging buffer.
      3) As a last resort, allocate without prefilling and return the buffer; caller must write the data.
    """
    size = len(data)

    # 1) Preferred: direct contiguous with-data API (current dylib signature)
    if hasattr(libegpu, 'egpu_device_allocate_dma_buffer_with_data'):
      phys = c_uint64(0)
      virt = c_uint64(0)
      # Prepare a stable buffer for the duration of the call and pass as void*
      buf = create_string_buffer(data)
      h = int(libegpu.egpu_device_allocate_dma_buffer_with_data(
        self.handle, c_uint64(size), c_uint32(direction), cast(buf, c_void_p), c_uint64(size), byref(phys), byref(virt)
     ))
      if h != 0:
        return {'handle': int(h), 'size': size, 'physical_addr': int(phys.value), 'virtual_addr': int(virt.value)}
      # Fall through to staged/last-resort if the call fails

    # 2) Fallback: staged segmented-with-data + copy into a final contiguous buffer
    if hasattr(libegpu, 'egpu_device_allocate_dma_buffer_segmented_with_data'):
      staged = self.allocate_dma_buffer_segmented_with_data(data, direction)
      if staged is not None:
        final = self.allocate_dma_buffer(size, direction)
        if final is None:
          # Cleanup staged before returning
          self.destroy_dma_buffer(staged['handle'])
          return None
        # Copy data and free staging buffer regardless of copy success
        copied = self.copy_dma_buffers(staged['handle'], final['handle'], 0, size)
        self.destroy_dma_buffer(staged['handle'])
        if not copied:
          self.destroy_dma_buffer(final['handle'])
          return None
        # Ensure size is set (allocate_dma_buffer already includes phys/virt)
        final['size'] = size
        return final

    # 3) Last resort: allocate without prefilling; caller must write the bytes
    return self.allocate_dma_buffer(size, direction)

  def allocate_dma_buffer_segmented(self, size: int, direction: int, max_pairs: int = 32):
    """
    Allocate a DMA buffer and return scatter/gather segments.
    - Responsibility: caller chooses max_pairs; this function allocates buffers for up to max_pairs
      and forwards the request to the driver. The driver returns up to that many segments (or fewer).
    - Returns None on failure or dict with:
        - handle: buffer handle (opaque)
        - virtual_addr: CPU pointer in the driver task (diagnostic). For memcpy in user space, map the buffer
                        via IOConnectMapMemory using the handle/memory type returned elsewhere in the API.
        - segments: list[(dma_iova, length)] for device programming
        - segment_count: number of IOVA segments returned (scatter/gather entries)
        - pages: alias of segment_count for compatibility with code that equates "pages" with segments
        - iova_4k_pages: total 4K pages across all segments (ceil(sum(length)/4096))
    Notes:
      - Program the device with segment dma_iova values (IOVA), not the CPU pointer.
    """
    # Fallback if symbol not present
    if not hasattr(libegpu, 'egpu_device_allocate_dma_buffer_segmented'):
      single = self.allocate_dma_buffer(size, direction)
      if not single: return None
      return {'handle': single['handle'], 'virtual_addr': single['virtual_addr'], 'segments': [(single['physical_addr'], size)]}

    out_count = c_uint32(0)
    handle = c_uint64(0)
    virt = c_uint64(0)
    addrs = (c_uint64 * max_pairs)()
    lens  = (c_uint64 * max_pairs)()

    ok = int(libegpu.egpu_device_allocate_dma_buffer_segmented(self.handle, c_uint64(size), c_uint32(direction), c_uint32(max_pairs),
                                                               byref(out_count), addrs, lens, byref(handle), byref(virt)))
    if ok != 1: return None

    segs = [(int(addrs[i]), int(lens[i])) for i in range(int(out_count.value))]
    total_len = sum(l for _, l in segs)
    iova_4k_pages = (total_len + 4095) // 4096
    segment_count = len(segs)
    return {
      'handle': int(handle.value),
      'virtual_addr': int(virt.value),
      'segments': segs,
      'segment_count': segment_count,
      'pages': segment_count,                # compatibility: some code treats "pages" == segments
      'iova_4k_pages': int(iova_4k_pages),   # actual total 4K pages across segments
    }

  def allocate_dma_buffer_segmented_with_data(self, data: bytes, direction: int, max_pairs: int = 32):
    if not hasattr(libegpu, 'egpu_device_allocate_dma_buffer_segmented_with_data'):
      return None
    out_count = c_uint32(0)
    handle = c_uint64(0)
    virt = c_uint64(0)
    addrs = (c_uint64 * max_pairs)()
    lens  = (c_uint64 * max_pairs)()
    buf = create_string_buffer(data)
    ok = int(libegpu.egpu_device_allocate_dma_buffer_segmented_with_data(
      self.handle, c_uint64(len(data)), c_uint32(direction), cast(buf, c_void_p), c_uint64(len(data)), c_uint32(max_pairs),
      byref(out_count), addrs, lens, byref(handle), byref(virt)
    ))
    if ok != 1: return None
    segs = [(int(addrs[i]), int(lens[i])) for i in range(int(out_count.value))]
    total_len = sum(l for _, l in segs)
    iova_4k_pages = (total_len + 4095) // 4096
    return {
      'handle': int(handle.value),
      'virtual_addr': int(virt.value),
      'segments': segs,
      'segment_count': len(segs),
      'iova_4k_pages': int(iova_4k_pages),
    }

  # (note) second duplicate definition removed; the implementation above is the authoritative version

  def destroy_dma_buffer(self, handle: int) -> bool:
    return int(libegpu.egpu_device_destroy_dma_buffer(self.handle, c_uint64(handle))) == 1

  def get_memory_type_for_handle(self, handle: int) -> int:
    return int(libegpu.egpu_device_get_memory_type_for_handle(self.handle, c_uint64(handle)))

  def copy_dma_buffers(self, src_handle: int, dst_handle: int, offset: int, size: int) -> bool:
    return int(libegpu.egpu_device_copy_dma_buffers(self.handle, c_uint64(src_handle), c_uint64(dst_handle), c_uint64(offset), c_uint64(size))) == 1


_device_singleton: MacOSEGPUDevice|None = None

def acquire_device(device_id: int = 0) -> MacOSEGPUDevice:
  global _device_singleton
  if _device_singleton is None:
    _device_singleton = MacOSEGPUDevice(device_id)
  return _device_singleton


