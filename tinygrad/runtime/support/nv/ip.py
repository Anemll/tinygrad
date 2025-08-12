from __future__ import annotations
import ctypes, time, array, struct, itertools, dataclasses, functools, types
from typing import cast, Any
from tinygrad.runtime.autogen.nv import nv
from tinygrad.helpers import to_mv, lo32, hi32, DEBUG, round_up, round_down, mv_address, fetch, wait_cond

# --- Debug tracing utility ---
def _trace_call(func):
  @functools.wraps(func)
  def wrapper(*args, **kwargs):
    if DEBUG > 1:
      try:
        print(func.__qualname__)
      except Exception:
        print(getattr(func, '__name__', 'function'))
    return func(*args, **kwargs)
  return wrapper

def trace_methods(cls):
  for name, attr in list(cls.__dict__.items()):
    if name.startswith('__') and name.endswith('__'): continue
    if isinstance(attr, staticmethod):
      setattr(cls, name, staticmethod(_trace_call(attr.__func__)))
    elif isinstance(attr, classmethod):
      setattr(cls, name, classmethod(_trace_call(attr.__func__)))
    elif callable(attr):
      setattr(cls, name, _trace_call(attr))
  return cls
from tinygrad.runtime.support.system import System
from tinygrad.runtime.support.elf import elf_loader
from tinygrad.runtime.autogen import nv_gpu

@dataclasses.dataclass(frozen=True)
class GRBufDesc: size:int; virt:bool; phys:bool; local:bool=False # noqa: E702

@trace_methods
class NV_IP:
  def __init__(self, nvdev): self.nvdev = nvdev
  def init_sw(self): pass # Prepare sw/allocations for this IP
  def init_hw(self): pass # Initialize hw for this IP
  def fini_hw(self): pass # Finalize hw for this IP

@trace_methods
class NVRpcQueue:
  def __init__(self, gsp:NV_GSP, va:int, completion_q_va:int|None=None):
    self.tx = nv.msgqTxHeader.from_address(va)
    wait_cond(lambda: self.tx.entryOff, value=0x1000, msg="RPC queue not initialized")

    if completion_q_va is not None: self.rx = nv.msgqRxHeader.from_address(completion_q_va + nv.msgqTxHeader.from_address(completion_q_va).rxHdrOff)

    self.gsp, self.va, self.queue_va, self.seq = gsp, va, va + self.tx.entryOff, 0
    self.queue_mv = to_mv(self.queue_va, self.tx.msgSize * self.tx.msgCount)

  def _checksum(self, data:bytes):
    if (pad_len:=(-len(data)) % 8): data += b'\x00' * pad_len
    checksum = 0
    for offset in range(0, len(data), 8): checksum ^= struct.unpack_from('Q', data, offset)[0]
    return hi32(checksum) ^ lo32(checksum)

  def send_rpc(self, func:int, msg:bytes, wait=False):
    header = nv.rpc_message_header_v(signature=nv.NV_VGPU_MSG_SIGNATURE_VALID, rpc_result=nv.NV_VGPU_MSG_RESULT_RPC_PENDING,
      rpc_result_private=nv.NV_VGPU_MSG_RESULT_RPC_PENDING, header_version=(3<<24), function=func, length=len(msg) + 0x20)

    msg = bytes(header) + msg
    phdr = nv.GSP_MSG_QUEUE_ELEMENT(elemCount=round_up(len(msg), self.tx.msgSize) // self.tx.msgSize, seqNum=self.seq)
    phdr.checkSum = self._checksum(bytes(phdr) + msg)
    msg = bytes(phdr) + msg

    off = self.tx.writePtr * self.tx.msgSize
    self.queue_mv[off:off+len(msg)] = msg
    self.tx.writePtr = (self.tx.writePtr + round_up(len(msg), self.tx.msgSize) // self.tx.msgSize) % self.tx.msgCount
    System.memory_barrier()

    self.seq += 1
    self.gsp.nvdev.NV_PGSP_QUEUE_HEAD[0].write(0x0)

  def wait_resp(self, cmd:int) -> memoryview:
    while True:
      System.memory_barrier()
      if self.rx.readPtr == self.tx.writePtr: continue

      off = self.rx.readPtr * self.tx.msgSize
      hdr = nv.rpc_message_header_v.from_address(self.queue_va + off + 0x30)
      msg = self.queue_mv[off + 0x50 : off + 0x50 + hdr.length]

      # Handling special functions
      if hdr.function == nv.NV_VGPU_MSG_EVENT_GSP_RUN_CPU_SEQUENCER: self.gsp.run_cpu_seq(msg)
      elif hdr.function == nv.NV_VGPU_MSG_EVENT_OS_ERROR_LOG:
        print(f"nv {self.gsp.nvdev.devfmt}: GSP LOG: {msg[12:].tobytes().rstrip(bytes([0])).decode('utf-8')}")

      # Update the read pointer
      self.rx.readPtr = (self.rx.readPtr + round_up(hdr.length, self.tx.msgSize) // self.tx.msgSize) % self.tx.msgCount
      System.memory_barrier()

      if DEBUG >= 3:
        rpc_names = {**nv.c__Ea_NV_VGPU_MSG_FUNCTION_NOP__enumvalues, **nv.c__Ea_NV_VGPU_MSG_EVENT_FIRST_EVENT__enumvalues}
        print(f"nv {self.gsp.nvdev.devfmt}: in RPC: {rpc_names.get(hdr.function, f'ev:{hdr.function:x}')}, res:{hdr.rpc_result:#x}")

      # Debug register snapshot on every response
      self.gsp.dump_gsp_regs()

      if hdr.rpc_result != 0: raise RuntimeError(f"RPC call {hdr.function} failed with result {hdr.rpc_result}")
      if hdr.rpc_result != 0:
        # Extra dump on error for parity with macOS logging
        self.gsp.dump_gsp_regs()
      if hdr.function == cmd: return msg

@trace_methods
class NV_FLCN(NV_IP):
  def init_sw(self):
    self.nvdev.include("src/common/inc/swref/published/ampere/ga102/dev_gsp.h")
    self.nvdev.include("src/common/inc/swref/published/ampere/ga102/dev_falcon_v4.h")
    self.nvdev.include("src/common/inc/swref/published/ampere/ga102/dev_falcon_v4_addendum.h")
    self.nvdev.include("src/common/inc/swref/published/ampere/ga102/dev_riscv_pri.h")
    self.nvdev.include("src/common/inc/swref/published/ampere/ga102/dev_fbif_v4.h")
    self.nvdev.include("src/common/inc/swref/published/ampere/ga102/dev_falcon_second_pri.h")
    self.nvdev.include("src/common/inc/swref/published/ampere/ga102/dev_sec_pri.h")
    self.nvdev.include("src/common/inc/swref/published/turing/tu102/dev_bus.h")

    self.prep_ucode()
    self.prep_booter()

  def prep_ucode(self):
    expansion_rom_off, bit_addr = {"GA": 0x16600, "AD": 0x14e00, "GB": 0x14e00}[self.nvdev.chip_name[:2]], 0x1b0
    vbios_bytes = bytes(array.array('I', self.nvdev.mmio[0x00300000//4:(0x00300000+0x98e00)//4]))

    bit_header = nv.BIT_HEADER_V1_00.from_buffer_copy(vbios_bytes[bit_addr:bit_addr + ctypes.sizeof(nv.BIT_HEADER_V1_00)])
    assert bit_header.Signature == 0x00544942, f"Invalid BIT header signature {hex(bit_header.Signature)}"

    for i in range(bit_header.TokenEntries):
      bit = nv.BIT_TOKEN_V1_00.from_buffer_copy(vbios_bytes[bit_addr + bit_header.HeaderSize + i * bit_header.TokenSize:])
      if bit.TokenId != nv.BIT_TOKEN_FALCON_DATA or bit.DataVersion != 2 or bit.DataSize < nv.BIT_DATA_FALCON_DATA_V2_SIZE_4: continue

      falcon_data = nv.BIT_DATA_FALCON_DATA_V2.from_buffer_copy(vbios_bytes[bit.DataPtr & 0xffff:])
      ucode_hdr = nv.FALCON_UCODE_TABLE_HDR_V1.from_buffer_copy(vbios_bytes[(table_ptr:=expansion_rom_off + falcon_data.FalconUcodeTablePtr):])
      for j in range(ucode_hdr.EntryCount):
        ucode_entry = nv.FALCON_UCODE_TABLE_ENTRY_V1.from_buffer_copy(vbios_bytes[table_ptr + ucode_hdr.HeaderSize + j * ucode_hdr.EntrySize:])
        if ucode_entry.ApplicationID != nv.FALCON_UCODE_ENTRY_APPID_FWSEC_PROD: continue

        ucode_desc_hdr = nv.FALCON_UCODE_DESC_HEADER.from_buffer_copy(vbios_bytes[expansion_rom_off + ucode_entry.DescPtr:])
        ucode_desc_off = expansion_rom_off + ucode_entry.DescPtr
        ucode_desc_size = ucode_desc_hdr.vDesc >> 16

    self.desc_v3 = nv.FALCON_UCODE_DESC_V3.from_buffer_copy(vbios_bytes[ucode_desc_off:ucode_desc_off + ucode_desc_size])

    sig_total_size = ucode_desc_size - nv.FALCON_UCODE_DESC_V3_SIZE_44
    signature = vbios_bytes[ucode_desc_off + nv.FALCON_UCODE_DESC_V3_SIZE_44:][:sig_total_size]
    image = vbios_bytes[ucode_desc_off + ucode_desc_size:][:round_up(self.desc_v3.StoredSize, 256)]

    self.frts_offset = self.nvdev.vram_size - 0x100000 - 0x100000
    read_vbios_desc = nv.FWSECLIC_READ_VBIOS_DESC(version=0x1, size=ctypes.sizeof(nv.FWSECLIC_READ_VBIOS_DESC), flags=2)
    frst_reg_desc = nv.FWSECLIC_FRTS_REGION_DESC(version=0x1, size=ctypes.sizeof(nv.FWSECLIC_FRTS_REGION_DESC),
      frtsRegionOffset4K=self.frts_offset >> 12, frtsRegionSize=0x100, frtsRegionMediaType=2)
    frts_cmd = nv.FWSECLIC_FRTS_CMD(readVbiosDesc=read_vbios_desc, frtsRegionDesc=frst_reg_desc)

    def __patch(cmd_id, cmd):
      patched_image = bytearray(image)

      hdr = nv.FALCON_APPLICATION_INTERFACE_HEADER_V1.from_buffer_copy(image[(app_hdr_off:=self.desc_v3.IMEMLoadSize+self.desc_v3.InterfaceOffset):])
      ents = (nv.FALCON_APPLICATION_INTERFACE_ENTRY_V1 * hdr.entryCount).from_buffer_copy(image[app_hdr_off + ctypes.sizeof(hdr):])
      for i in range(hdr.entryCount):
        if ents[i].id == nv.FALCON_APPLICATION_INTERFACE_ENTRY_ID_DMEMMAPPER: dmem_offset = ents[i].dmemOffset

      # Patch image
      dmem = nv.FALCON_APPLICATION_INTERFACE_DMEM_MAPPER_V3.from_buffer_copy(image[(dmem_mapper_offset:=self.desc_v3.IMEMLoadSize+dmem_offset):])
      dmem.init_cmd = cmd_id
      patched_image[dmem_mapper_offset : dmem_mapper_offset+len(bytes(dmem))] = bytes(dmem)
      patched_image[(cmd_off:=self.desc_v3.IMEMLoadSize+dmem.cmd_in_buffer_offset) : cmd_off+len(cmd)] = cmd
      patched_image[(sig_off:=self.desc_v3.IMEMLoadSize+self.desc_v3.PKCDataOffset) : sig_off+0x180] = signature[-0x180:]

      return System.alloc_sysmem(len(patched_image), contiguous=True, data=patched_image)

    self.frts_image_va, self.frts_image_sysmem = __patch(0x15, bytes(frts_cmd))

  def prep_booter(self):
    image = self.nvdev.extract_fw("kgspBinArchiveBooterLoadUcode", "image_prod_data")
    sig = self.nvdev.extract_fw("kgspBinArchiveBooterLoadUcode", "sig_prod_data")
    header = self.nvdev.extract_fw("kgspBinArchiveBooterLoadUcode", "header_prod_data")
    patch_loc = int.from_bytes(self.nvdev.extract_fw("kgspBinArchiveBooterLoadUcode", "patch_loc_data"), 'little')
    sig_len = len(sig) // int.from_bytes(self.nvdev.extract_fw("kgspBinArchiveBooterLoadUcode", "num_sigs_data"), 'little')

    patched_image = bytearray(image)
    patched_image[patch_loc:patch_loc+sig_len] = sig[:sig_len]
    self.booter_image_va, self.booter_image_sysmem = System.alloc_sysmem(len(patched_image), contiguous=True, data=patched_image, name="BOOTER_UCODE_IMAGE")
    _, _, self.booter_data_off, self.booter_data_sz, _, self.booter_code_off, self.booter_code_sz, _, _ = struct.unpack("9I", header)

  def init_hw(self):
    self.falcon, self.sec2 = 0x00110000, 0x00840000

    self.reset(self.falcon)
    self.execute_hs(self.falcon, self.frts_image_sysmem[0], code_off=0x0, data_off=self.desc_v3.IMEMLoadSize,
      imemPa=self.desc_v3.IMEMPhysBase, imemVa=self.desc_v3.IMEMVirtBase, imemSz=self.desc_v3.IMEMLoadSize,
      dmemPa=self.desc_v3.DMEMPhysBase, dmemVa=0x0, dmemSz=self.desc_v3.DMEMLoadSize,
      pkc_off=self.desc_v3.PKCDataOffset, engid=self.desc_v3.EngineIdMask, ucodeid=self.desc_v3.UcodeId)
    assert self.nvdev.NV_PFB_PRI_MMU_WPR2_ADDR_HI.read() != 0, "WPR2 is not initialized"

    self.reset(self.falcon, riscv=True)

    # set up the mailbox
    self.nvdev.NV_PGSP_FALCON_MAILBOX0.write(lo32(self.nvdev.gsp.libos_args_sysmem[0]))
    self.nvdev.NV_PGSP_FALCON_MAILBOX1.write(hi32(self.nvdev.gsp.libos_args_sysmem[0]))

    # booter
    self.reset(self.sec2)
    mbx = self.execute_hs(self.sec2, self.booter_image_sysmem[0], code_off=self.booter_code_off, data_off=self.booter_data_off,
      imemPa=0x0, imemVa=self.booter_code_off, imemSz=self.booter_code_sz, dmemPa=0x0, dmemVa=0x0, dmemSz=self.booter_data_sz,
      pkc_off=0x10, engid=1, ucodeid=3, mailbox=self.nvdev.gsp.wpr_meta_sysmem)
    assert mbx[0] == 0x0, f"Booter failed to execute, mailbox is {mbx[0]:08x}, {mbx[1]:08x}"

    self.nvdev.NV_PFALCON_FALCON_OS.with_base(self.falcon).write(0x0)
    assert self.nvdev.NV_PRISCV_RISCV_CPUCTL.with_base(self.falcon).read_bitfields()['active_stat'] == 1, "GSP Core is not active"

  def execute_dma(self, base:int, cmd:int, dest:int, mem_off:int, sysmem:int, size:int):
    wait_cond(lambda: self.nvdev.NV_PFALCON_FALCON_DMATRFCMD.with_base(base).read_bitfields()['full'], value=0, msg="DMA does not progress")

    self.nvdev.NV_PFALCON_FALCON_DMATRFBASE.with_base(base).write(lo32(sysmem >> 8))
    self.nvdev.NV_PFALCON_FALCON_DMATRFBASE1.with_base(base).write(hi32(sysmem >> 8) & 0x1ff)

    xfered = 0
    while xfered < size:
      wait_cond(lambda: self.nvdev.NV_PFALCON_FALCON_DMATRFCMD.with_base(base).read_bitfields()['full'], value=0, msg="DMA does not progress")

      self.nvdev.NV_PFALCON_FALCON_DMATRFMOFFS.with_base(base).write(dest + xfered)
      self.nvdev.NV_PFALCON_FALCON_DMATRFFBOFFS.with_base(base).write(mem_off + xfered)
      self.nvdev.NV_PFALCON_FALCON_DMATRFCMD.with_base(base).write(cmd)
      xfered += 256

    wait_cond(lambda: self.nvdev.NV_PFALCON_FALCON_DMATRFCMD.with_base(base).read_bitfields()['idle'], msg="DMA does not complete")

  def start_cpu(self, base:int):
    if self.nvdev.NV_PFALCON_FALCON_CPUCTL.with_base(base).read_bitfields()['alias_en'] == 1:
      self.nvdev.wreg(base + self.nvdev.NV_PFALCON_FALCON_CPUCTL_ALIAS, 0x2)
    else: self.nvdev.NV_PFALCON_FALCON_CPUCTL.with_base(base).write(startcpu=1)

  def wait_cpu_halted(self, base): wait_cond(lambda: self.nvdev.NV_PFALCON_FALCON_CPUCTL.with_base(base).read_bitfields()['halted'], msg="not halted")

  def execute_hs(self, base, img_sysmem, code_off, data_off, imemPa, imemVa, imemSz, dmemPa, dmemVa, dmemSz, pkc_off, engid, ucodeid, mailbox=None):
    self.disable_ctx_req(base)

    self.nvdev.NV_PFALCON_FBIF_TRANSCFG.with_base(base)[ctx_dma:=0].update(target=self.nvdev.NV_PFALCON_FBIF_TRANSCFG_TARGET_COHERENT_SYSMEM,
      mem_type=self.nvdev.NV_PFALCON_FBIF_TRANSCFG_MEM_TYPE_PHYSICAL)

    cmd = self.nvdev.NV_PFALCON_FALCON_DMATRFCMD.with_base(base).encode(write=0, size=self.nvdev.NV_PFALCON_FALCON_DMATRFCMD_SIZE_256B,
      ctxdma=ctx_dma, imem=1, sec=1)
    self.execute_dma(base, cmd, dest=imemPa, mem_off=imemVa, sysmem=img_sysmem+code_off-imemVa, size=imemSz)

    cmd = self.nvdev.NV_PFALCON_FALCON_DMATRFCMD.with_base(base).encode(write=0, size=self.nvdev.NV_PFALCON_FALCON_DMATRFCMD_SIZE_256B,
      ctxdma=ctx_dma, imem=0, sec=0)
    self.execute_dma(base, cmd, dest=dmemPa, mem_off=dmemVa, sysmem=img_sysmem+data_off-dmemVa, size=dmemSz)

    self.nvdev.NV_PFALCON2_FALCON_BROM_PARAADDR.with_base(base)[0].write(pkc_off)
    self.nvdev.NV_PFALCON2_FALCON_BROM_ENGIDMASK.with_base(base).write(engid)
    self.nvdev.NV_PFALCON2_FALCON_BROM_CURR_UCODE_ID.with_base(base).write(val=ucodeid)
    self.nvdev.NV_PFALCON2_FALCON_MOD_SEL.with_base(base).write(algo=self.nvdev.NV_PFALCON2_FALCON_MOD_SEL_ALGO_RSA3K)

    self.nvdev.NV_PFALCON_FALCON_BOOTVEC.with_base(base).write(imemVa)

    if mailbox is not None:
      self.nvdev.NV_PFALCON_FALCON_MAILBOX0.with_base(base).write(lo32(mailbox))
      self.nvdev.NV_PFALCON_FALCON_MAILBOX1.with_base(base).write(hi32(mailbox))

    self.start_cpu(base)
    self.wait_cpu_halted(base)

    if mailbox is not None:
      return self.nvdev.NV_PFALCON_FALCON_MAILBOX0.with_base(base).read(), self.nvdev.NV_PFALCON_FALCON_MAILBOX1.with_base(base).read()

  def disable_ctx_req(self, base:int):
    self.nvdev.NV_PFALCON_FBIF_CTL.with_base(base).update(allow_phys_no_ctx=1)
    self.nvdev.NV_PFALCON_FALCON_DMACTL.with_base(base).write(0x0)

  def reset(self, base:int, riscv=False):
    engine_reg = self.nvdev.NV_PGSP_FALCON_ENGINE if base == self.falcon else self.nvdev.NV_PSEC_FALCON_ENGINE
    engine_reg.write(reset=1)
    time.sleep(0.1)
    engine_reg.write(reset=0)

    wait_cond(lambda: self.nvdev.NV_PFALCON_FALCON_HWCFG2.with_base(base).read_bitfields()['mem_scrubbing'], value=0, msg="Scrubbing not completed")

    if riscv: self.nvdev.NV_PRISCV_RISCV_BCR_CTRL.with_base(base).write(core_select=1, valid=0, brfetch=1)
    elif self.nvdev.NV_PFALCON_FALCON_HWCFG2.with_base(base).read_bitfields()['riscv'] == 1:
      self.nvdev.NV_PRISCV_RISCV_BCR_CTRL.with_base(base).write(core_select=0)
      wait_cond(lambda: self.nvdev.NV_PRISCV_RISCV_BCR_CTRL.with_base(base).read_bitfields()['valid'], msg="RISCV core not booted")
      self.nvdev.NV_PFALCON_FALCON_RM.with_base(base).write(self.nvdev.chip_id)

@trace_methods
class NV_FLCN_COT(NV_IP):
  def init_sw(self):
    self.nvdev.include("src/common/inc/swref/published/ampere/ga102/dev_gsp.h")
    self.nvdev.include("src/common/inc/swref/published/hopper/gh100/dev_falcon_v4.h")
    self.nvdev.include("src/common/inc/swref/published/hopper/gh100/dev_vm.h")
    self.nvdev.include("src/common/inc/swref/published/hopper/gh100/dev_fsp_pri.h")
    self.nvdev.include("src/common/inc/swref/published/turing/tu102/dev_bus.h")
    self.nvdev.include("src/nvidia/arch/nvalloc/common/inc/fsp/fsp_mctp_format.h")
    self.nvdev.include("src/nvidia/arch/nvalloc/common/inc/fsp/fsp_emem_channels.h")

    self.fmc_boot_args, self.fmc_boot_args_sysmem = self.nvdev._alloc_boot_struct(nv.GSP_FMC_BOOT_PARAMS())
    self.init_fmc_image()

  def init_fmc_image(self):
    self.fmc_booter_image = self.nvdev.extract_fw("kgspBinArchiveGspRmFmcGfwProdSigned", "ucode_image_data")
    self.fmc_booter_hash = memoryview(self.nvdev.extract_fw("kgspBinArchiveGspRmFmcGfwProdSigned", "ucode_hash_data")).cast('I')
    self.fmc_booter_sig = memoryview(self.nvdev.extract_fw("kgspBinArchiveGspRmFmcGfwProdSigned", "ucode_sig_data")).cast('I')
    self.fmc_booter_pkey = memoryview(self.nvdev.extract_fw("kgspBinArchiveGspRmFmcGfwProdSigned", "ucode_pkey_data") + b'\x00\x00\x00').cast('I')
    _, self.fmc_booter_sysmem = System.alloc_sysmem(len(self.fmc_booter_image), contiguous=True, data=self.fmc_booter_image, name="FMC_BOOTER_UCODE_IMAGE")

  def init_hw(self):
    self.falcon = 0x00110000

    self.fmc_boot_args.bootGspRmParams = nv.GSP_ACR_BOOT_GSP_RM_PARAMS(gspRmDescOffset=self.nvdev.gsp.wpr_meta_sysmem,
      gspRmDescSize=ctypes.sizeof(nv.GspFwWprMeta), target=nv.GSP_DMA_TARGET_COHERENT_SYSTEM, bIsGspRmBoot=True)
    self.fmc_boot_args.gspRmParams = nv.GSP_RM_PARAMS(bootArgsOffset=self.nvdev.gsp.libos_args_sysmem[0], target=nv.GSP_DMA_TARGET_COHERENT_SYSTEM)

    cot_payload = nv.NVDM_PAYLOAD_COT(version=0x2, size=ctypes.sizeof(nv.NVDM_PAYLOAD_COT), frtsVidmemOffset=0x1c00000, frtsVidmemSize=0x100000,
      gspBootArgsSysmemOffset=self.fmc_boot_args_sysmem, gspFmcSysmemOffset=self.fmc_booter_sysmem[0])
    for i,x in enumerate(self.fmc_booter_hash): cot_payload.hash384[i] = x
    for i,x in enumerate(self.fmc_booter_sig): cot_payload.signature[i] = x
    for i,x in enumerate(self.fmc_booter_pkey): cot_payload.publicKey[i] = x

    self.kfsp_send_msg(nv.NVDM_TYPE_COT, bytes(cot_payload))
    wait_cond(lambda: self.nvdev.NV_PFALCON_FALCON_HWCFG2.with_base(self.falcon).read_bitfields()['riscv_br_priv_lockdown'], value=0)

  def dump_gsp_regs(self):
    """Dump GSP/FSP registers for debugging"""
    try:
      regs: dict[str, str] = {}

      # PGSP mailbox/queue
      try:
        regs["PGSP_MAILBOX0"] = hex(self.nvdev.NV_PGSP_FALCON_MAILBOX0.read())
        regs["PGSP_MAILBOX1"] = hex(self.nvdev.NV_PGSP_FALCON_MAILBOX1.read())
      except Exception:
        pass
      try:
        regs["PGSP_QUEUE_HEAD"] = hex(self.nvdev.NV_PGSP_QUEUE_HEAD[0].read())
      except Exception:
        pass
      try:
        regs["PGSP_QUEUE_TAIL"] = hex(self.nvdev.NV_PGSP_QUEUE_TAIL[0].read())
      except Exception:
        pass

      # PFSP queue/msgq
      try:
        regs["PFSP_QUEUE_HEAD"] = hex(self.nvdev.NV_PFSP_QUEUE_HEAD[0].read())
      except Exception:
        pass
      try:
        regs["PFSP_QUEUE_TAIL"] = hex(self.nvdev.NV_PFSP_QUEUE_TAIL[0].read())
      except Exception:
        pass
      try:
        regs["PFSP_MSGQ_HEAD"] = hex(self.nvdev.NV_PFSP_MSGQ_HEAD[0].read())
      except Exception:
        pass
      try:
        regs["PFSP_MSGQ_TAIL"] = hex(self.nvdev.NV_PFSP_MSGQ_TAIL[0].read())
      except Exception:
        pass

      # PMC boot and falcon cfg
      try:
        regs["PMC_BOOT_0"] = hex(self.nvdev.NV_PMC_BOOT_0.read())
      except Exception:
        pass
      try:
        if hasattr(self, 'falcon') and self.falcon:
          regs["PFALCON_HWCFG2_FSP"] = hex(self.nvdev.NV_PFALCON_FALCON_HWCFG2.with_base(self.falcon).read())
      except Exception:
        pass

      print(f"🐧🔍 GSP/FSP Reg Dump: {regs}")
    except Exception as e:
      print(f"🐧🔍 Reg Dump Error: {e}")

  def kfsp_send_msg(self, nvmd:int, buf:bytes):
    # All single-packets go to seid 0
    headers = int.to_bytes((1 << 31) | (1 << 30), 4, 'little') + int.to_bytes((0x7e << 0) | (0x10de << 8) | (nvmd << 24), 4, 'little')
    buf = headers + buf + (4 - (len(buf) % 4)) * b'\x00'
    assert len(buf) < 0x400, f"FSP message too long: {len(buf)} bytes, max 1024 bytes"

    self.nvdev.NV_PFSP_EMEMC[0].write(offs=0, blk=0, aincw=1, aincr=0)
    for i in range(0, len(buf), 4): self.nvdev.NV_PFSP_EMEMD[0].write(int.from_bytes(buf[i:i+4], 'little'))

    self.nvdev.NV_PFSP_QUEUE_TAIL[0].write(len(buf) - 4)
    self.nvdev.NV_PFSP_QUEUE_HEAD[0].write(0)

    # Waiting for a response
    wait_cond(lambda: self.nvdev.NV_PFSP_MSGQ_HEAD[0].read() != self.nvdev.NV_PFSP_MSGQ_TAIL[0].read(), msg="FSP didn't respond to message")

    self.nvdev.NV_PFSP_EMEMC[0].write(offs=0, blk=0, aincw=0, aincr=1)
    self.nvdev.NV_PFSP_MSGQ_TAIL[0].write(self.nvdev.NV_PFSP_MSGQ_HEAD[0].read())

    # Snapshot registers for debugging
    self.dump_gsp_regs()

@trace_methods
class NV_GSP(NV_IP):
  def init_sw(self):
    self.handle_gen = itertools.count(0xcf000000)
    self.init_rm_args()
    self.init_libos_args()
    self.init_wpr_meta()

    # Prefill cmd queue with info for gsp to start.
    self.rpc_set_gsp_system_info()
    self.rpc_set_registry_table()

    self.gpfifo_class, self.compute_class, self.dma_class = nv_gpu.AMPERE_CHANNEL_GPFIFO_A, nv_gpu.AMPERE_COMPUTE_B, nv_gpu.AMPERE_DMA_COPY_B
    match self.nvdev.chip_name[:2]:
      case "AD": self.compute_class = nv_gpu.ADA_COMPUTE_A
      case "GB":
        self.gpfifo_class,self.compute_class,self.dma_class=nv_gpu.BLACKWELL_CHANNEL_GPFIFO_A,nv_gpu.BLACKWELL_COMPUTE_B,nv_gpu.BLACKWELL_DMA_COPY_B

  def init_rm_args(self, queue_size=0x40000):
    # Alloc queues
    pte_cnt = ((queue_pte_cnt:=(queue_size * 2) // 0x1000)) + round_up(queue_pte_cnt * 8, 0x1000) // 0x1000
    pt_size = round_up(pte_cnt * 8, 0x1000)
    queues_va, queues_sysmem = System.alloc_sysmem(pt_size + queue_size * 2, contiguous=False, name="LIBOS_LOG_BUFFER_REGION")
    if DEBUG >= 1:
      print(f"MSGQ PT: pte_cnt={pte_cnt}, pt_size={pt_size}, total_pages={len(queues_sysmem)}")
      for i in range(min(8, len(queues_sysmem))):
        print(f"  PTE[{i}] = 0x{queues_sysmem[i]:x}")
      if len(queues_sysmem) > 8:
        print(f"  ... {len(queues_sysmem)-8} more PTEs")

    # Fill up ptes
    for i, sysmem in enumerate(queues_sysmem): to_mv(queues_va + i * 0x8, 0x8).cast('Q')[0] = sysmem

    # Fill up arguments
    queue_args = nv.MESSAGE_QUEUE_INIT_ARGUMENTS(sharedMemPhysAddr=queues_sysmem[0], pageTableEntryCount=pte_cnt, cmdQueueOffset=pt_size,
      statQueueOffset=pt_size + queue_size)
    rm_args, self.rm_args_sysmem = self.nvdev._alloc_boot_struct(nv.GSP_ARGUMENTS_CACHED(bDmemStack=True, messageQueueInitArguments=queue_args))

    # Build command queue header
    self.cmd_q_va, self.stat_q_va = queues_va + pt_size, queues_va + pt_size + queue_size

    cmd_q_tx = nv.msgqTxHeader(version=0, size=queue_size, entryOff=0x1000, msgSize=0x1000, msgCount=(queue_size - 0x1000) // 0x1000,
      writePtr=0, flags=1, rxHdrOff=ctypes.sizeof(nv.msgqTxHeader))
    to_mv(self.cmd_q_va, ctypes.sizeof(nv.msgqTxHeader))[:] = bytes(cmd_q_tx)

    self.cmd_q = NVRpcQueue(self, self.cmd_q_va, None)

  def init_libos_args(self):
    logbuf_va, logbuf_sysmem = System.alloc_sysmem((2 << 20), contiguous=True, name="LIBOS_LOG_BUFFER_REGION")
    libos_args_va, self.libos_args_sysmem = System.alloc_sysmem(0x1000, contiguous=True, name="LIBOS_ARGS_BLOCK")

    libos_structs = (nv.LibosMemoryRegionInitArgument * 6).from_address(libos_args_va)
    for i, name in enumerate(["INIT", "INTR", "RM", "MNOC", "KRNL"]):
      libos_structs[i] = nv.LibosMemoryRegionInitArgument(kind=nv.LIBOS_MEMORY_REGION_CONTIGUOUS, loc=nv.LIBOS_MEMORY_REGION_LOC_SYSMEM, size=0x10000,
        id8=int.from_bytes(bytes(f"LOG{name}", 'utf-8'), 'big'), pa=logbuf_sysmem[0] + 0x10000 * i)

    libos_structs[5] = nv.LibosMemoryRegionInitArgument(kind=nv.LIBOS_MEMORY_REGION_CONTIGUOUS, loc=nv.LIBOS_MEMORY_REGION_LOC_SYSMEM, size=0x1000,
        id8=int.from_bytes(bytes("RMARGS", 'utf-8'), 'big'), pa=self.rm_args_sysmem)

    # Track mapped view for Linux log dumping (2MB buffer)
    self.libos_log_sysmem = logbuf_sysmem
    self.libos_log_mv = to_mv(logbuf_va, (2 << 20))

  def init_gsp_image(self):
    fw = fetch("https://github.com/NVIDIA/linux-firmware/raw/refs/heads/nvidia-staging/nvidia/ga102/gsp/gsp-570.144.bin", subdir="fw").read_bytes()

    _, sections, _ = elf_loader(fw)
    self.gsp_image = next((sh.content for sh in sections if sh.name == ".fwimage"))
    signature = next((sh.content for sh in sections if sh.name == (f".fwsignature_{self.nvdev.chip_name[:4].lower()}x")))

    # Build radix3
    npages = [0, 0, 0, round_up(len(self.gsp_image), 0x1000) // 0x1000]
    for i in range(3, 0, -1): npages[i-1] = ((npages[i] - 1) >> (nv.LIBOS_MEMORY_REGION_RADIX_PAGE_LOG2 - 3)) + 1

    offsets = [sum(npages[:i]) * 0x1000 for i in range(4)]
    radix_va, self.gsp_radix3_sysmem = System.alloc_sysmem(offsets[-1] + len(self.gsp_image), contiguous=False, name="GSP_RADIX3_IMAGE_AND_PT")

    # Copy image
    to_mv(radix_va + offsets[-1], len(self.gsp_image))[:] = self.gsp_image

    # Copy level and image pages.
    for i in range(0, 3):
      cur_offset = sum(npages[:i+1])
      to_mv(radix_va + offsets[i], npages[i+1] * 8).cast('Q')[:] = array.array('Q', self.gsp_radix3_sysmem[cur_offset:cur_offset+npages[i+1]])

    # Copy signature
    self.gsp_signature_va, self.gsp_signature_sysmem = System.alloc_sysmem(len(signature), contiguous=True, data=signature, name="GSP_IMAGE_SIGNATURE")

  def init_boot_binary_image(self):
    self.booter_image = self.nvdev.extract_fw("kgspBinArchiveGspRmBoot", "ucode_image_prod_data")
    self.booter_desc = nv.RM_RISCV_UCODE_DESC.from_buffer_copy(self.nvdev.extract_fw("kgspBinArchiveGspRmBoot", "ucode_desc_prod_data"))
    _, self.booter_sysmem = System.alloc_sysmem(len(self.booter_image), contiguous=True, data=self.booter_image, name="GSP_BOOTER_IMAGE")

  def init_wpr_meta(self):
    self.init_gsp_image()
    self.init_boot_binary_image()

    common = {'sizeOfBootloader':(boot_sz:=len(self.booter_image)), 'sysmemAddrOfBootloader':self.booter_sysmem[0],
      'sizeOfRadix3Elf':(radix3_sz:=len(self.gsp_image)), 'sysmemAddrOfRadix3Elf': self.gsp_radix3_sysmem[0],
      'sizeOfSignature': 0x1000, 'sysmemAddrOfSignature': self.gsp_signature_sysmem[0],
      'bootloaderCodeOffset': self.booter_desc.monitorCodeOffset, 'bootloaderDataOffset': self.booter_desc.monitorDataOffset,
      'bootloaderManifestOffset': self.booter_desc.manifestOffset, 'revision':nv.GSP_FW_WPR_META_REVISION, 'magic':nv.GSP_FW_WPR_META_MAGIC}

    if self.nvdev.fmc_boot:
      m = nv.GspFwWprMeta(**common, vgaWorkspaceSize=0x20000, pmuReservedSize=0x1820000, nonWprHeapSize=0x220000, gspFwHeapSize=0x8700000,
        frtsSize=0x100000)
    else:
      m = nv.GspFwWprMeta(**common, vgaWorkspaceSize=(vga_sz:=0x100000), vgaWorkspaceOffset=(vga_off:=self.nvdev.vram_size-vga_sz),
        gspFwWprEnd=vga_off, frtsSize=(frts_sz:=0x100000), frtsOffset=(frts_off:=vga_off-frts_sz), bootBinOffset=(boot_off:=frts_off-boot_sz),
        gspFwOffset=(gsp_off:=round_down(boot_off-radix3_sz, 0x10000)), gspFwHeapSize=(gsp_heap_sz:=0x8100000), fbSize=self.nvdev.vram_size,
        gspFwHeapOffset=(gsp_heap_off:=round_down(gsp_off-gsp_heap_sz, 0x100000)), gspFwWprStart=(wpr_st:=round_down(gsp_heap_off-0x1000, 0x100000)),
        nonWprHeapSize=(non_wpr_sz:=0x100000), nonWprHeapOffset=(non_wpr_off:=round_down(wpr_st-non_wpr_sz, 0x100000)), gspFwRsvdStart=non_wpr_off)
      assert self.nvdev.flcn.frts_offset == m.frtsOffset, f"FRTS mismatch: {self.nvdev.flcn.frts_offset} != {m.frtsOffset}"
    self.wpr_meta, self.wpr_meta_sysmem = self.nvdev._alloc_boot_struct(m)

  def promote_ctx(self, client:int, subdevice:int, obj:int, ctxbufs:dict[int, GRBufDesc], bufs=None, virt=None, phys=None):
    res, prom = {}, nv_gpu.NV2080_CTRL_GPU_PROMOTE_CTX_PARAMS(entryCount=len(ctxbufs), engineType=0x1, hChanClient=client, hObject=obj)
    for i,(buf,desc) in enumerate(ctxbufs.items()):
      use_v, use_p = (desc.virt if virt is None else virt), (desc.phys if phys is None else phys)
      x = (bufs or {}).get(buf, self.nvdev.mm.valloc(desc.size, contiguous=True)) # allocate buffers
      prom.promoteEntry[i] = nv_gpu.NV2080_CTRL_GPU_PROMOTE_CTX_BUFFER_ENTRY(bufferId=buf, gpuVirtAddr=x.va_addr if use_v else 0, bInitialize=use_p,
        gpuPhysAddr=x.paddrs[0][0] if use_p else 0, size=desc.size if use_p else 0, physAttr=0x4 if use_p else 0, bNonmapped=(use_p and not use_v))
      res[buf] = x
    self.rpc_rm_control(hObject=subdevice, cmd=nv_gpu.NV2080_CTRL_CMD_GPU_PROMOTE_CTX, params=prom, client=client)
    return res

  def init_golden_image(self):
    self.rpc_rm_alloc(hParent=0x0, hClass=0x0, params=nv_gpu.NV0000_ALLOC_PARAMETERS())
    dev = self.rpc_rm_alloc(hParent=self.priv_root, hClass=nv_gpu.NV01_DEVICE_0, params=nv_gpu.NV0080_ALLOC_PARAMETERS(hClientShare=self.priv_root))
    subdev = self.rpc_rm_alloc(hParent=dev, hClass=nv_gpu.NV20_SUBDEVICE_0, params=nv_gpu.NV2080_ALLOC_PARAMETERS())
    vaspace = self.rpc_rm_alloc(hParent=dev, hClass=nv_gpu.FERMI_VASPACE_A, params=nv_gpu.NV_VASPACE_ALLOCATION_PARAMETERS())

    # reserve 512MB for the reserved PDES
    res_va = self.nvdev.mm.alloc_vaddr(res_sz:=(512 << 20))

    bufs_p = nv_gpu.struct_NV90F1_CTRL_VASPACE_COPY_SERVER_RESERVED_PDES_PARAMS(pageSize=res_sz, numLevelsToCopy=3,
      virtAddrLo=res_va, virtAddrHi=res_va + res_sz - 1)
    for i,pt in enumerate(self.nvdev.mm.page_tables(res_va, size=res_sz)):
      bufs_p.levels[i] = nv_gpu.struct_NV90F1_CTRL_VASPACE_COPY_SERVER_RESERVED_PDES_PARAMS_0(physAddress=pt.paddr,
        size=self.nvdev.mm.pte_cnt[0] * 8 if i == 0 else 0x1000, pageShift=self.nvdev.mm.pte_covers[i].bit_length() - 1, aperture=1)
    self.rpc_rm_control(hObject=vaspace, cmd=nv_gpu.NV90F1_CTRL_CMD_VASPACE_COPY_SERVER_RESERVED_PDES, params=bufs_p)

    gpfifo_area = self.nvdev.mm.valloc(4 << 10, contiguous=True)
    userd = nv_gpu.NV_MEMORY_DESC_PARAMS(base=gpfifo_area.paddrs[0][0] + 0x20 * 8, size=0x20, addressSpace=2, cacheAttrib=0)
    gg_params = nv_gpu.NV_CHANNELGPFIFO_ALLOCATION_PARAMETERS(gpFifoOffset=gpfifo_area.va_addr, gpFifoEntries=32, engineType=0x1, cid=3,
      hVASpace=vaspace, userdOffset=(ctypes.c_uint64*8)(0x20 * 8), userdMem=userd, internalFlags=0x1a, flags=0x200320)
    ch_gpfifo = self.rpc_rm_alloc(hParent=dev, hClass=self.gpfifo_class, params=gg_params)

    gr_ctx_bufs_info = self.rpc_rm_control(hObject=subdev, cmd=nv_gpu.NV2080_CTRL_CMD_INTERNAL_STATIC_KGR_GET_CONTEXT_BUFFERS_INFO,
      params=nv_gpu.NV2080_CTRL_INTERNAL_STATIC_KGR_GET_CONTEXT_BUFFERS_INFO_PARAMS()).engineContextBuffersInfo[0]
    def _ctx_info(idx, add=0, align=None): return round_up(gr_ctx_bufs_info.engine[idx].size + add, align or gr_ctx_bufs_info.engine[idx].alignment)

    # Setup graphics context
    gr_size = _ctx_info(nv_gpu.NV0080_CTRL_FIFO_GET_ENGINE_CONTEXT_PROPERTIES_ENGINE_ID_GRAPHICS, add=0x40000)
    patch_size = _ctx_info(nv_gpu.NV0080_CTRL_FIFO_GET_ENGINE_CONTEXT_PROPERTIES_ENGINE_ID_GRAPHICS_PATCH)
    cfgs_sizes = {x: _ctx_info(x + 14, align=(2 << 20) if x == 5 else None) for x in range(3, 11)} # indices 3–10 are mapped to 17–24
    self.grctx_bufs = {0: GRBufDesc(gr_size, phys=True, virt=True), 1: GRBufDesc(patch_size, phys=True, virt=True, local=True),
      2: GRBufDesc(patch_size, phys=True, virt=True), **{x: GRBufDesc(cfgs_sizes[x], phys=False, virt=True) for x in range(3, 7)},
      9: GRBufDesc(cfgs_sizes[9], phys=True, virt=True), 10: GRBufDesc(cfgs_sizes[10], phys=True, virt=False),
      11: GRBufDesc(cfgs_sizes[10], phys=True, virt=True)} # NOTE: 11 reuses cfgs_sizes[10]
    self.promote_ctx(self.priv_root, subdev, ch_gpfifo, {k:v for k, v in self.grctx_bufs.items() if not v.local})

    self.rpc_rm_alloc(hParent=ch_gpfifo, hClass=self.compute_class, params=None)
    self.rpc_rm_alloc(hParent=ch_gpfifo, hClass=self.dma_class, params=None)

  def init_hw(self):
    self.stat_q = NVRpcQueue(self, self.stat_q_va, self.cmd_q_va)
    self.cmd_q.rx = nv.msgqRxHeader.from_address(self.stat_q.va + self.stat_q.tx.rxHdrOff)

    self.dump_gsp_regs()
    if DEBUG:
      print("--------------------------------")
      print("GSP INIT DONE...")
      print("--------------------------------")
    self.stat_q.wait_resp(nv.NV_VGPU_MSG_EVENT_GSP_INIT_DONE)

    # Early reg snapshot after init done
    self.dump_gsp_regs()

    self.nvdev.NV_PBUS_BAR1_BLOCK.write(mode=0, target=0, ptr=0)
    if self.nvdev.fmc_boot: self.nvdev.NV_VIRTUAL_FUNCTION_PRIV_FUNC_BAR1_BLOCK_LOW_ADDR.write(mode=0, target=0, ptr=0)

    self.priv_root = 0xc1e00004
    self.init_golden_image()

    # Post-golden-image reg snapshot and LIBOS baseline dump
    self.dump_gsp_regs()
    self.dump_log_buffer()

  def fini_hw(self): self.rpc_unloading_guest_driver()

  ### RPCs

  def rpc_rm_alloc(self, hParent:int, hClass:int, params:Any, client=None) -> int:
    if hClass == self.gpfifo_class:
      ramfc_alloc = self.nvdev.mm.valloc(0x1000, contiguous=True)
      params.ramfcMem = nv_gpu.NV_MEMORY_DESC_PARAMS(base=ramfc_alloc.paddrs[0][0], size=0x200, addressSpace=2, cacheAttrib=0)
      params.instanceMem = nv_gpu.NV_MEMORY_DESC_PARAMS(base=ramfc_alloc.paddrs[0][0], size=0x1000, addressSpace=2, cacheAttrib=0)

      method_va, method_sysmem = System.alloc_sysmem(0x5000, contiguous=True, name="NV_METHOD_BUFFER")
      params.mthdbufMem = nv_gpu.NV_MEMORY_DESC_PARAMS(base=method_sysmem[0], size=0x5000, addressSpace=1, cacheAttrib=0)

      if client is not None and client != self.priv_root and params.hObjectError != 0:
        params.errorNotifierMem = nv_gpu.NV_MEMORY_DESC_PARAMS(base=0, size=0xecc, addressSpace=0, cacheAttrib=0)
        params.userdMem = nv_gpu.NV_MEMORY_DESC_PARAMS(base=params.hUserdMemory[0] + params.userdOffset[0], size=0x400, addressSpace=2, cacheAttrib=0)

    alloc_args = nv.rpc_gsp_rm_alloc_v(hClient=(client:=client or self.priv_root), hParent=hParent, hObject=(obj:=next(self.handle_gen)),
      hClass=hClass, flags=0x0, paramsSize=ctypes.sizeof(params) if params is not None else 0x0)
    self.cmd_q.send_rpc(nv.NV_VGPU_MSG_FUNCTION_GSP_RM_ALLOC, bytes(alloc_args) + (bytes(params) if params is not None else b''))
    self.stat_q.wait_resp(nv.NV_VGPU_MSG_FUNCTION_GSP_RM_ALLOC)

    if hClass == nv_gpu.FERMI_VASPACE_A and client != self.priv_root:
      self.rpc_set_page_directory(device=hParent, hVASpace=obj, pdir_paddr=self.nvdev.mm.root_page_table.paddr, client=client)
    if hClass == nv_gpu.NV20_SUBDEVICE_0: self.subdevice = obj # save subdevice handle
    if hClass == self.compute_class and client != self.priv_root:
      phys_gr_ctx = self.promote_ctx(client, self.subdevice, hParent, {k:v for k,v in self.grctx_bufs.items() if k in [0, 1, 2]}, virt=False)
      self.promote_ctx(client, self.subdevice, hParent, {k:v for k,v in self.grctx_bufs.items() if k in [0, 1, 2]}, phys_gr_ctx, phys=False)
    return obj if hClass != nv_gpu.NV1_ROOT else client

  def rpc_rm_control(self, hObject:int, cmd:int, params:Any, client=None):
    control_args = nv.rpc_gsp_rm_control_v(hClient=(client:=client or self.priv_root), hObject=hObject, cmd=cmd, flags=0x0,
      paramsSize=ctypes.sizeof(params) if params is not None else 0x0)
    self.cmd_q.send_rpc(nv.NV_VGPU_MSG_FUNCTION_GSP_RM_CONTROL, bytes(control_args) + (bytes(params) if params is not None else b''))
    res = self.stat_q.wait_resp(nv.NV_VGPU_MSG_FUNCTION_GSP_RM_CONTROL)
    st = type(params).from_buffer_copy(res[len(bytes(control_args)):]) if params is not None else None

    # NOTE: gb20x requires the enable bit for token submission. Patch workSubmitToken here to maintain userspace compatibility.
    if self.nvdev.chip_name.startswith("GB2") and cmd == nv_gpu.NVC36F_CTRL_CMD_GPFIFO_GET_WORK_SUBMIT_TOKEN:
      cast(nv_gpu.NVC36F_CTRL_CMD_GPFIFO_GET_WORK_SUBMIT_TOKEN_PARAMS, st).workSubmitToken |= (1 << 30)
    return st

  def rpc_set_page_directory(self, device:int, hVASpace:int, pdir_paddr:int, client=None, pasid=0xffffffff):
    params = nv.struct_NV0080_CTRL_DMA_SET_PAGE_DIRECTORY_PARAMS_v1E_05(physAddress=pdir_paddr,
      numEntries=self.nvdev.mm.pte_cnt[0], flags=0x8, hVASpace=hVASpace, pasid=pasid, subDeviceId=1, chId=0) # flags field is all channels.
    alloc_args = nv.rpc_set_page_directory_v(hClient=client or self.priv_root, hDevice=device, pasid=pasid, params=params)
    self.cmd_q.send_rpc(nv.NV_VGPU_MSG_FUNCTION_SET_PAGE_DIRECTORY, bytes(alloc_args))
    self.stat_q.wait_resp(nv.NV_VGPU_MSG_FUNCTION_SET_PAGE_DIRECTORY)

  def rpc_set_gsp_system_info(self):
    def bdf_as_int(s): return (int(s[5:7],16)<<8) | (int(s[8:10],16)<<3) | int(s[-1],16)

    data = nv.GspSystemInfo(gpuPhysAddr=self.nvdev.bars[0][0], gpuPhysFbAddr=self.nvdev.bars[1][0], gpuPhysInstAddr=self.nvdev.bars[3][0],
      pciConfigMirrorBase=[0x88000, 0x92000][self.nvdev.fmc_boot], pciConfigMirrorSize=0x1000, nvDomainBusDeviceFunc=bdf_as_int(self.nvdev.devfmt),
      bIsPassthru=1, PCIDeviceID=self.nvdev.venid, PCISubDeviceID=self.nvdev.subvenid, PCIRevisionID=self.nvdev.rev, maxUserVa=0x7ffffffff000)
    self.cmd_q.send_rpc(nv.NV_VGPU_MSG_FUNCTION_GSP_SET_SYSTEM_INFO, bytes(data))

  def rpc_unloading_guest_driver(self):
    data = nv.rpc_unloading_guest_driver_v(bInPMTransition=0, bGc6Entering=0, newLevel=(__GPU_STATE_FLAGS_FAST_UNLOAD:=1 << 6))
    self.cmd_q.send_rpc(nv.NV_VGPU_MSG_FUNCTION_UNLOADING_GUEST_DRIVER, bytes(data))
    self.stat_q.wait_resp(nv.NV_VGPU_MSG_FUNCTION_UNLOADING_GUEST_DRIVER)

  def rpc_set_registry_table(self):
    table = {
      'RMForcePcieConfigSave': 0x1,
      'RMSecBusResetEnable': 0x1,
      # Spoof OS as Linux for firmware logic consistency with Linux runs
      'gspOsType': 1,
    }
    entries_bytes, data_bytes = bytes(), bytes()
    hdr_size, entries_size = ctypes.sizeof(nv.PACKED_REGISTRY_TABLE), ctypes.sizeof(nv.PACKED_REGISTRY_ENTRY) * len(table)

    for k,v in table.items():
      entries_bytes += bytes(nv.PACKED_REGISTRY_ENTRY(nameOffset=hdr_size + entries_size + len(data_bytes),
                                                      type=nv.REGISTRY_TABLE_ENTRY_TYPE_DWORD, data=v, length=4))
      data_bytes += k.encode('utf-8') + b'\x00'

    header = nv.PACKED_REGISTRY_TABLE(size=hdr_size + len(entries_bytes) + len(data_bytes), numEntries=len(table))
    self.cmd_q.send_rpc(nv.NV_VGPU_MSG_FUNCTION_SET_REGISTRY, bytes(header) + entries_bytes + data_bytes)

  def dump_gsp_regs(self):
    """Dump GSP/FSP registers for debugging"""
    try:
      regs: dict[str, str] = {}

      # GSP mailbox registers
      try:
        regs["MAILBOX0"] = hex(self.nvdev.NV_PGSP_FALCON_MAILBOX0.read())
        regs["MAILBOX1"] = hex(self.nvdev.NV_PGSP_FALCON_MAILBOX1.read())
      except Exception:
        pass

      # GSP queue registers
      try:
        regs["QUEUE_HEAD"] = hex(self.nvdev.NV_PGSP_QUEUE_HEAD[0].read())
      except Exception:
        pass
      try:
        regs["QUEUE_TAIL"] = hex(self.nvdev.NV_PGSP_QUEUE_TAIL[0].read())
      except Exception:
        pass

      # PMC boot register
      try:
        regs["PMC_BOOT_0"] = hex(self.nvdev.NV_PMC_BOOT_0.read())
      except Exception:
        pass

      # Falcon HWCFG2 at GSP base
      try:
        gsp_base = self.nvdev.flcn.falcon if hasattr(self.nvdev.flcn, 'falcon') else 0
        if gsp_base:
          regs["FALCON_HWCFG2"] = hex(self.nvdev.NV_PFALCON_FALCON_HWCFG2.with_base(gsp_base).read())
      except Exception:
        pass

      print(f"🐧🔍 GSP Reg Dump: {regs}")
    except Exception as e:
      print(f"🐧🔍 Reg Dump Error: {e}")

  def dump_log_buffer(self):
    """Dump and decode LIBOS log buffer for Linux runs"""
    try:
      if not (hasattr(self, 'libos_log_sysmem') and hasattr(self, 'libos_log_mv')):
        return

      # Ensure CPU sees latest writes
      try:
        from tinygrad.runtime.autogen import libc
        libc.msync(ctypes.c_void_p(mv_address(self.libos_log_mv)), ctypes.c_size_t(len(self.libos_log_mv)), libc.MS_SYNC)
      except Exception:
        pass

      log_mv = self.libos_log_mv
      log_data = log_mv[:0x10000].tobytes()

      # First, dump as hex for analysis
      print("\n📋 LibOS Log Buffer Hex Dump:")
      print("=" * 80)

      # Find the last non-zero byte to avoid dumping tons of zeros
      last_nonzero = 0
      for i in range(len(log_data) - 1, -1, -1):
        if log_data[i] != 0:
          last_nonzero = i
          break

      if last_nonzero > 0:
        # Round up to next 16-byte boundary
        dump_size = ((last_nonzero + 16) // 16) * 16
        dump_size = min(dump_size, len(log_data))

        print(f"Dumping 0x{dump_size:x} bytes (last non-zero at 0x{last_nonzero:x})")
        print("Offset    00 01 02 03 04 05 06 07  08 09 0A 0B 0C 0D 0E 0F  |ASCII...........|")
        print("-" * 80)

        for offset in range(0, dump_size, 16):
          # Hex part
          hex_part = f"{offset:08x}  "
          ascii_part = "|"

          for j in range(16):
            if offset + j < len(log_data):
              byte = log_data[offset + j]
              hex_part += f"{byte:02x} "
              # ASCII representation
              ascii_part += chr(byte) if 32 <= byte <= 126 else "."
            else:
              hex_part += "   "
              ascii_part += " "

            # Add extra space in the middle
            if j == 7:
              hex_part += " "

          ascii_part += "|"
          print(hex_part + " " + ascii_part)
      else:
        print("📋 LibOS Log Buffer is empty (all zeros)")

      print("=" * 80)

      # Decode LibOS ring buffer: scan backwards from put pointer for recent entries
      print("\n📋 Extracting Recent LibOS Log Entries:")
      print("-" * 80)

      put_pointer = int.from_bytes(log_mv[0:8], 'little')
      recent_messages = self._parse_libos_basic(log_mv.tobytes(), put_pointer)

      if recent_messages:
        print(f"📝 Found {len(recent_messages)} recent log entries (scanning backwards from put pointer):")
        for i, msg in enumerate(recent_messages[:40]):  # Show up to 40 recent entries
          # Highlight critical messages
          msg_lower = msg.lower()
          if any(keyword in msg_lower for keyword in ['error', 'fail', 'fatal', 'assert', 'panic']):
            print(f"  🚨 [{i:3d}] {msg}")
          elif any(keyword in msg_lower for keyword in ['init', 'sysmem', 'bar', 'dma', 'rpc']):
            print(f"  ⚠️  [{i:3d}] {msg}")
          else:
            print(f"     [{i:3d}] {msg}")

        if len(recent_messages) > 40:
          print(f"     ... {len(recent_messages) - 40} more entries")
      else:
        print("  ⚠️  No readable recent log entries found")
        print("  ℹ️  Ring buffer may be:")
        print("     - Empty (GSP didn't log before failing)")  
        print("     - Using compressed/binary format only")
        print("     - Circular buffer wrapped multiple times")

      # Also try the debug helper for verification
      print(f"\n📋 Debug Helper Verification (raw metadata parsing):")
      debug_entries = self._dump_meta_words_debug(log_mv.tobytes(), put_pointer//8, 10)
      if debug_entries:
        print(f"📝 Debug helper found {len(debug_entries)} valid entries:")
        for entry in debug_entries:
          print(f"     {entry}")
      else:
        print("📝 Debug helper found no valid entries")
      
      print("=" * 80)
          
    except Exception as e:
      print(f"❌ Failed to dump LibOS logs: {e}")

  def _dump_meta_words_debug(self, buf: bytes, start_idx: int, count: int = 10) -> list[str]:
    """Debug helper to dump metadata words correctly (as provided by user)"""
    import struct
    messages = []
    
    try:
      words = struct.unpack("<" + "Q"*(len(buf)//8), buf)
      log_entries = len(words) - 1
      slot = lambda x: 1 + (x % log_entries)
      
      # Use the ring put value as-is (element index), NOT bytes
      i = start_idx  # should already be in element units
      previous_put = max(0, i - log_entries)
      
      entries_found = 0
      while entries_found < count:
        # Adjust previous_put if put wrapped past our window
        if previous_put + log_entries < i:
            previous_put = i - log_entries

        if i == previous_put:
            break
            
        i -= 1
        ts = words[slot(i)]
        i -= 1
        meta = words[slot(i)]
        total_args = (meta >> 48) & 0xff
        task_id    = (meta >> 56) & 0xff
        arg_count  = total_args - 2
        meta_va    = meta & ((1<<48)-1)
        
        # Accept only plausible v2 packed records
        if 2 <= total_args <= 20 and 0 <= arg_count <= 18 and 1 <= task_id <= 9:
          task_name = {
            1: "LIBOS", 2: "GPU_SCHED", 3: "GSP_RM", 4: "SEC2", 
            5: "NVDEC", 6: "NVENC", 7: "NVJPG", 8: "NVLINK", 9: "FSP"
          }.get(task_id, f"T{task_id}")
          
          messages.append(f"[@{i:04x}] ts=0x{ts:016x} task={task_name}({task_id}) argc={arg_count} meta=0x{meta:016x} (va=0x{meta_va:012x})")
          entries_found += 1
          
    except Exception as e:
      messages.append(f"Debug helper failed: {e}")
      
    return messages

  def _parse_libos_basic(self, log_data: bytes, put_pointer: int) -> list[str]:
    """Basic LibOS parser using correct metadata format"""
    import struct
    messages = []
    
    try:
      if len(log_data) % 8 != 0:
        return ["Buffer not 64-bit aligned"]
        
      words = struct.unpack("<" + "Q" * (len(log_data) // 8), log_data)
      log_entries = len(words) - 1
      slot = lambda x: 1 + (x % log_entries)

      # Convert put_pointer from bytes to words if needed
      # The put_pointer at offset 0 is in bytes, we need words
      i = put_pointer // 8  # Convert bytes to word index
      previous_put = max(0, i - log_entries)  # decode full ring if you don't track prev

      # Debug: show what we're working with
      messages.append(f"Debug: put_pointer={put_pointer} (0x{put_pointer:x}), log_entries={log_entries}, buffer_words={len(words)}")
      messages.append(f"Debug: Starting scan from i={i}, previous_put={previous_put}")
      
      # Show some raw words around the put_pointer area
      if len(words) > 10:
        start_word = max(0, min(i - 5, len(words) - 10))
        messages.append(f"Debug: Raw words around put area (starting at word {start_word}):")
        for j in range(10):
          if start_word + j < len(words):
            word_val = words[start_word + j]
            messages.append(f"  word[{start_word + j:3d}] = 0x{word_val:016x}")
      
      # Check if put_pointer is reasonable
      if i >= len(words):
        messages.append(f"Debug: put_pointer {put_pointer} >= buffer words {len(words)}, using buffer end")
        i = len(words) - 1
      elif i <= 0:
        messages.append(f"Debug: put_pointer {put_pointer} <= 0, using small value")
        i = min(10, len(words) - 1)

      entries_found = 0
      scanned_count = 0
      while entries_found < 10 and scanned_count < 20:  # Add scan limit to prevent infinite loops
          # Adjust previous_put if put wrapped past our window (matches C logic)
          if previous_put + log_entries < i:
              previous_put = i - log_entries

          if i == previous_put:
              break

          i -= 1
          ts = words[slot(i)]
          i -= 1
          meta = words[slot(i)]

          total_args = (meta >> 48) & 0xFF
          task_id    = (meta >> 56) & 0xFF
          arg_count  = total_args - 2
          meta_va    = meta & ((1 << 48) - 1)

          # Debug: show what we found regardless of validity
          scanned_count += 1
          messages.append(f"Debug scan #{scanned_count}: i={i}, ts=0x{ts:016x}, meta=0x{meta:016x}")
          messages.append(f"  -> total_args={total_args}, task_id={task_id}, arg_count={arg_count}, meta_va=0x{meta_va:012x}")

          # Accept only plausible v2 packed records
          if 2 <= total_args <= 20 and 0 <= arg_count <= 18 and 1 <= task_id <= 9:
              # Task name mapping from libos_log.h
              task_names = {
                1: "GSP_KERNEL", 2: "GSP_INIT", 3: "GSP_RM", 4: "GSP_INTR", 
                5: "GSP_VGPU", 6: "GSP_MNOC", 7: "GSP_DEBUG", 8: "GSP_ROOT", 9: "GSP_MAX"
              }
              task_name = task_names.get(task_id, f"T{task_id}")
              
              messages.append(f"[@0x{i:04x}] ts=0x{ts:016x} task={task_name}({task_id}) argc={arg_count} meta=0x{meta:016x} (va=0x{meta_va:012x})")
              entries_found += 1
          else:
              # Show why it was rejected
              reasons = []
              if not (2 <= total_args <= 20): reasons.append(f"total_args={total_args} not in [2,20]")
              if not (0 <= arg_count <= 18): reasons.append(f"arg_count={arg_count} not in [0,18]")  
              if not (1 <= task_id <= 9): reasons.append(f"task_id={task_id} not in [1,9]")
              messages.append(f"  -> REJECTED: {'; '.join(reasons)}")
              # likely unpacked metadata (older/flag not set) → cannot get arg_count without ELF; skip
          
    except Exception as e:
      messages.append(f"Parser failed: {e}")
    
    if not messages or entries_found == 0:
      # Try parsing as NOCAT/RATS format instead of LibOS v2
      messages.append("No valid LibOS v2 records found - trying NOCAT/RATS format...")
      nocat_messages = self._parse_nocat_format(log_data, put_pointer)
      messages.extend(nocat_messages)
    
    return messages

  def _parse_nocat_format(self, log_data: bytes, put_pointer: int) -> list[str]:
    """Parse NOCAT/RATS log format based on actual buffer content"""
    import struct
    messages = []
    
    try:
      # Convert to 64-bit words
      words = struct.unpack("<" + "Q" * (len(log_data) // 8), log_data)
      
      messages.append(f"Parsing NOCAT format: {len(log_data)} bytes, {len(words)} words")
      
      # Look for the magic pattern 0x20003008 or 0x20003XXX in the words
      magic_base = 0x20003000
      magic_mask = 0xFFFFF000
      
      # Also look for known error codes
      known_errors = {
        0x019f59d0: "0x59 NV_ERR_OPERATING_SYSTEM related",
        0x01a7cd82: "GSP error code",
        0x01b30b62: "GSP error code",
        0x019e31fa: "GSP error code",
        0x0141f954: "GSP error code",
        0x0141f782: "GSP error code",
        0x01a7cd8e: "GSP error code",
        0x013a4194: "GSP error code"
      }
      
      entries_found = 0
      max_entries = 30
      
      # Scan through words looking for patterns
      for i in range(len(words) - 2):
        if entries_found >= max_entries:
          break
          
        word = words[i]
        
        # Check if this looks like a NOCAT entry marker (0x20003XXX pattern)
        if (word & magic_mask) == magic_base:
          # This could be a NOCAT entry
          # Next word should be timestamp or error code
          if i + 1 < len(words):
            next_word = words[i + 1]
            
            # Check if next word is a timestamp (large value > 0x0000000100000000)
            if next_word > 0x0000000100000000:
              messages.append(f"NOCAT[{entries_found:2d}] word[{i:4d}]: marker=0x{word:016x}, timestamp=0x{next_word:016x}")
              entries_found += 1
            # Or check if previous word is a known error code
            elif i > 0:
              prev_word = words[i - 1]
              if prev_word in known_errors:
                messages.append(f"NOCAT[{entries_found:2d}] word[{i:4d}]: error=0x{prev_word:08x} ({known_errors[prev_word]}), marker=0x{word:016x}")
                entries_found += 1
              elif prev_word > 0 and prev_word < 0x10000000:
                # Could be an unknown error code
                messages.append(f"NOCAT[{entries_found:2d}] word[{i:4d}]: code=0x{prev_word:08x}, marker=0x{word:016x}")
        
        # Also check if this word itself is a known error
        elif word in known_errors:
          messages.append(f"Error[{entries_found:2d}] word[{i:4d}]: 0x{word:08x} = {known_errors[word]}")
          entries_found += 1
      
      if entries_found == 0:
        messages.append("No NOCAT patterns found - showing raw word dump instead:")
        # Show some interesting words
        for i in range(min(20, len(words))):
          if words[i] != 0:
            messages.append(f"  word[{i:3d}] = 0x{words[i]:016x}")
        
    except Exception as e:
      messages.append(f"NOCAT parser failed: {e}")
      
    return messages

  def run_cpu_seq(self, seq_buf:memoryview):
    hdr = nv.rpc_run_cpu_sequencer_v17_00.from_address(mv_address(seq_buf))
    cmd_iter = iter(seq_buf[ctypes.sizeof(nv.rpc_run_cpu_sequencer_v17_00):].cast('I')[:hdr.cmdIndex])

    for op in cmd_iter:
      if op == 0x0: self.nvdev.wreg(next(cmd_iter), next(cmd_iter)) # reg write
      elif op == 0x1: # reg modify
        addr, val, mask = next(cmd_iter), next(cmd_iter), next(cmd_iter)
        self.nvdev.wreg(addr, (self.nvdev.rreg(addr) & ~mask) | (val & mask))
      elif op == 0x2: # reg poll
        addr, mask, val, _, _ = next(cmd_iter), next(cmd_iter), next(cmd_iter), next(cmd_iter), next(cmd_iter)
        wait_cond(lambda: (self.nvdev.rreg(addr) & mask), value=val, msg=f"Register {addr:#x} not equal to {val:#x} after polling")
      elif op == 0x3: time.sleep(next(cmd_iter) / 1e6) # delay us
      elif op == 0x4: # save reg
        addr, index = next(cmd_iter), next(cmd_iter)
        hdr.regSaveArea[index] = self.nvdev.rreg(addr)
      elif op == 0x5: # core reset
        self.nvdev.flcn.reset(self.nvdev.flcn.falcon)
        self.nvdev.flcn.disable_ctx_req(self.nvdev.flcn.falcon)
      elif op == 0x6: self.nvdev.flcn.start_cpu(self.nvdev.flcn.falcon)
      elif op == 0x7: self.nvdev.flcn.wait_cpu_halted(self.nvdev.flcn.falcon)
      elif op == 0x8: # core resume
        self.nvdev.flcn.reset(self.nvdev.flcn.falcon, riscv=True)

        self.nvdev.NV_PGSP_FALCON_MAILBOX0.write(lo32(self.libos_args_sysmem[0]))
        self.nvdev.NV_PGSP_FALCON_MAILBOX1.write(hi32(self.libos_args_sysmem[0]))

        self.nvdev.flcn.start_cpu(self.nvdev.flcn.sec2)
        wait_cond(lambda: self.nvdev.NV_PGC6_BSI_SECURE_SCRATCH_14.read_bitfields()['boot_stage_3_handoff'], msg="SEC2 didn't hand off")

        mailbox = self.nvdev.NV_PFALCON_FALCON_MAILBOX0.with_base(self.nvdev.flcn.sec2).read()
        assert mailbox == 0x0, f"Falcon SEC2 failed to execute, mailbox is {mailbox:08x}"
      else: raise ValueError(f"Unknown op code {op} in run_cpu_seq")
