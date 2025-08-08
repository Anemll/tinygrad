from __future__ import annotations
import os, struct
from ctypes import CDLL, c_void_p, c_int, c_uint64, c_uint32, c_uint16, POINTER, Structure, byref
from tinygrad.helpers import getenv, DEBUG
from tinygrad.runtime.support.hcq import MMIOInterface

NV_DEBUG = getenv("NV_DEBUG", 0)

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

class EGPUMMIOInterface(MMIOInterface):
  def __init__(self, device_handle: c_void_p, bar_index: int, fmt: str = 'I'):
    if libegpu is None: raise RuntimeError("libegpu_pcidevice.dylib not available")
    self._fmt = fmt
    self._if = libegpu.egpu_device_map_bar(device_handle, c_int(bar_index))
    if not self._if: raise RuntimeError(f"Failed to map BAR{bar_index}")
    self._elt = struct.calcsize(fmt)
    self._size_bytes = 64 * 1024 * 1024
    if DEBUG >= 2: print(f"macos_egpu: mapped BAR{bar_index} -> iface=0x{int(self._if):x}")

  def __len__(self):
    return self._size_bytes // self._elt

  def __getitem__(self, idx: int):
    off = idx * self._elt
    if self._fmt == 'I': return int(libegpu.egpu_mmio_read32(self._if, c_uint64(off)))
    if self._fmt == 'B': return int(libegpu.egpu_mmio_read8(self._if, c_uint64(off)))
    raise NotImplementedError(self._fmt)

  def __setitem__(self, idx: int, val: int):
    off = idx * self._elt
    if self._fmt == 'I': libegpu.egpu_mmio_write32(self._if, c_uint64(off), c_uint32(val))
    elif self._fmt == 'B': libegpu.egpu_mmio_write8(self._if, c_uint64(off), c_uint32(val & 0xff))
    else: raise NotImplementedError(self._fmt)
    libegpu.egpu_mmio_memory_barrier(self._if)

  def view(self, offset: int = 0, size: int | None = None, fmt: str | None = None):
    # For now, return self; higher layers don't rely on view semantics for MMIO
    return self

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


