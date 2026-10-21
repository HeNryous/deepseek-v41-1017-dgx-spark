# SPDX-License-Identifier: Apache-2.0
"""Display-backed KV cache backing for GB10 (patch 0053, VLLM_DSV41_DISPLAY_KV_MIB=<MiB>, default off).

The NVIDIA DRM driver of a DGX Spark keeps a display reservation (1792 MiB per node) that the OS does not count as RAM.
With nvidia_drm loaded as modeset=1 fbdev=0 and no monitor scanning out, a DRM dumb buffer of that size can be mapped
into the process and registered with CUDA as I/O memory; GPU kernels then read and write it like device memory.
Technique as in Gilesco's GLM recipe (patch 0072, Apache-2.0) and 0rand's TensorFold fork (Apache-2.0); this is our own
implementation for vLLM.

vLLM keeps the whole KV cache in ONE backing allocation (vllm/v1/worker/utils.py allocate_kv_cache: every layer is a
strided view into it). So the backing here is one contiguous device range: [display buffer | ordinary device memory],
the tail allocated with the CUDA virtual-memory API (cuMemCreate + cuMemMap) right behind the display mapping. The
display part adds capacity without ordinary memory: with --kv-cache-memory-bytes raised by the display size, the
ordinary-memory share of the KV cache stays what it was. Cache rows are only touched by GPU kernels (vLLM does not copy
KV blocks with copy engines in this setup); the backing is zeroed by a kernel.
"""
from __future__ import annotations

import ctypes
import fcntl
import mmap
import os
import struct

ENV = "VLLM_DSV41_DISPLAY_KV_MIB"
CARD_ENV = "VLLM_DSV41_DISPLAY_KV_CARD"
MIB = 1 << 20
MAX_MIB = 1792
GRAN = 2 * MIB

# Linux DRM uapi (include/uapi/drm/drm.h, drm_mode.h)
_IOCTL_MODE_CREATE_DUMB = 0xC02064B2   # struct drm_mode_create_dumb {u32 h, w, bpp, flags, handle, pitch; u64 size}
_IOCTL_MODE_MAP_DUMB = 0xC01064B3      # struct drm_mode_map_dumb {u32 handle, pad; u64 offset}
_IOCTL_GEM_CLOSE = 0x40086409          # struct drm_gem_close {u32 handle, pad}
_PITCH = 16384                         # 4096 px x 32 bpp
# mman.h (Python's mmap module has no PROT_NONE / MAP_FIXED)
_PROT_NONE, _PROT_RW = 0, mmap.PROT_READ | mmap.PROT_WRITE
_MAP_SHARED, _MAP_PRIVATE, _MAP_FIXED, _MAP_ANON = mmap.MAP_SHARED, mmap.MAP_PRIVATE, 0x10, mmap.MAP_ANONYMOUS

# CUDA driver API constants
_CU_MEMHOSTREGISTER_DEVICEMAP = 0x02
_CU_MEMHOSTREGISTER_IOMEMORY = 0x04
_CU_MEM_ALLOCATION_TYPE_PINNED = 1
_CU_MEM_LOCATION_TYPE_DEVICE = 1
_CU_MEM_ACCESS_FLAGS_PROT_READWRITE = 3


class _CUmemLocation(ctypes.Structure):
    _fields_ = [("type", ctypes.c_int), ("id", ctypes.c_int)]


class _CUmemAllocFlags(ctypes.Structure):
    _fields_ = [("compressionType", ctypes.c_ubyte), ("gpuDirectRDMACapable", ctypes.c_ubyte),
                ("usage", ctypes.c_ushort), ("reserved", ctypes.c_ubyte * 4)]


class _CUmemAllocationProp(ctypes.Structure):
    _fields_ = [("type", ctypes.c_int), ("requestedHandleTypes", ctypes.c_int), ("location", _CUmemLocation),
                ("win32HandleMetaData", ctypes.c_void_p), ("allocFlags", _CUmemAllocFlags)]


class _CUmemAccessDesc(ctypes.Structure):
    _fields_ = [("location", _CUmemLocation), ("flags", ctypes.c_int)]


def requested_bytes(environ=None) -> int:
    raw = ((os.environ if environ is None else environ).get(ENV) or "").strip()
    if not raw or raw == "0":
        return 0
    if not raw.isdecimal():
        raise ValueError(f"{ENV}={raw!r}: expected MiB")
    mib = int(raw)
    if mib % 2 or mib > MAX_MIB:
        raise ValueError(f"{ENV}={mib}: expected an even number of MiB up to {MAX_MIB}")
    return mib * MIB


def nvidia_card(environ=None, sysfs="/sys/class/drm") -> str:
    environ = os.environ if environ is None else environ
    want = (environ.get(CARD_ENV) or "").strip()
    cards = [want] if want else ["/dev/dri/" + c for c in sorted(os.listdir(sysfs))
                                 if c.startswith("card") and c[4:].isdigit()]
    for card in cards:
        name = os.path.basename(card)
        try:
            vendor = open(os.path.join(sysfs, name, "device", "vendor")).read().strip().lower()
        except OSError:
            continue
        if vendor == "0x10de":
            # a connected output with an enabled encoder would be scanning out of this memory
            for entry in os.listdir(sysfs):
                if entry.startswith(name + "-"):
                    base = os.path.join(sysfs, entry)
                    try:
                        if open(base + "/status").read().strip() == "connected" and \
                                open(base + "/enabled").read().strip() == "enabled":
                            raise RuntimeError(f"{ENV}: {entry} is connected and enabled")
                    except OSError:
                        pass
            return card
    raise RuntimeError(f"{ENV}: no NVIDIA DRM card (is /dev/dri passed into the container?)")


class _Cuda:
    def __init__(self):
        self.lib = ctypes.CDLL("libcuda.so.1")
        L = self.lib
        L.cuMemHostRegister_v2.argtypes = [ctypes.c_void_p, ctypes.c_size_t, ctypes.c_uint]
        L.cuMemHostGetDevicePointer_v2.argtypes = [ctypes.POINTER(ctypes.c_uint64), ctypes.c_void_p, ctypes.c_uint]
        L.cuMemHostUnregister.argtypes = [ctypes.c_void_p]
        L.cuMemAddressReserve.argtypes = [ctypes.POINTER(ctypes.c_uint64), ctypes.c_size_t, ctypes.c_size_t,
                                          ctypes.c_uint64, ctypes.c_ulonglong]
        L.cuMemAddressFree.argtypes = [ctypes.c_uint64, ctypes.c_size_t]
        L.cuMemCreate.argtypes = [ctypes.POINTER(ctypes.c_ulonglong), ctypes.c_size_t,
                                  ctypes.POINTER(_CUmemAllocationProp), ctypes.c_ulonglong]
        L.cuMemMap.argtypes = [ctypes.c_uint64, ctypes.c_size_t, ctypes.c_size_t, ctypes.c_ulonglong,
                               ctypes.c_ulonglong]
        L.cuMemSetAccess.argtypes = [ctypes.c_uint64, ctypes.c_size_t, ctypes.POINTER(_CUmemAccessDesc),
                                     ctypes.c_size_t]
        L.cuMemUnmap.argtypes = [ctypes.c_uint64, ctypes.c_size_t]
        L.cuMemRelease.argtypes = [ctypes.c_ulonglong]
        L.cuGetErrorName.argtypes = [ctypes.c_int, ctypes.POINTER(ctypes.c_char_p)]

    def check(self, rc, what):
        if rc:
            name = ctypes.c_char_p()
            self.lib.cuGetErrorName(rc, ctypes.byref(name))
            raise RuntimeError(f"{ENV}: {what} failed: CUDA error {rc} ({(name.value or b'?').decode()})")


_KEEP = []   # owners stay alive for the process lifetime (the KV cache does)


class Backing:
    """[display | ordinary] contiguous device range; ``__cuda_array_interface__`` for torch.as_tensor."""

    def __init__(self, total_bytes: int, display_bytes: int, device_index: int):
        if display_bytes <= 0 or display_bytes % GRAN:
            raise ValueError(f"{ENV}: display bytes must be a positive multiple of 2 MiB")
        tail = max(0, -(-(total_bytes - display_bytes) // GRAN) * GRAN)
        self.size, self.display, self.tail = display_bytes + tail, display_bytes, tail
        self.nbytes = min(total_bytes, self.size)   # what the tensor exposes (exact request, no rounding tail)
        import torch

        torch.empty(1, device=f"cuda:{device_index}")   # makes the primary context current for the driver API below
        torch.cuda.synchronize(device_index)
        cu = self.cu = _Cuda()
        libc = ctypes.CDLL("libc.so.6", use_errno=True)
        libc.mmap.restype = ctypes.c_void_p
        libc.mmap.argtypes = [ctypes.c_void_p, ctypes.c_size_t, ctypes.c_int, ctypes.c_int, ctypes.c_int,
                              ctypes.c_long]
        libc.munmap.argtypes = [ctypes.c_void_p, ctypes.c_size_t]
        test_anon = os.environ.get("DISPLAY_KV_TEST_ANON") == "1"   # probe only: ordinary memory stands in for DRM
        self.card = "anon" if test_anon else nvidia_card()
        self.fd = -1 if test_anon else os.open(self.card, os.O_RDWR | os.O_CLOEXEC)
        try:
            # 1) display dumb buffer
            if test_anon:
                offset = 0
            else:
                req = struct.pack("=6IQ", -(-display_bytes // _PITCH), 4096, 32, 0, 0, 0, 0)
                out = struct.unpack("=6IQ", fcntl.ioctl(self.fd, _IOCTL_MODE_CREATE_DUMB, req))
                self.handle, got = out[4], out[6]
                if got < display_bytes:
                    raise RuntimeError(f"{ENV}: display buffer is {got} B, short of {display_bytes} B")
                offset = struct.unpack("=2IQ", fcntl.ioctl(self.fd, _IOCTL_MODE_MAP_DUMB,
                                                           struct.pack("=2IQ", self.handle, 0, 0)))[2]
            # 2) a 2 MiB-aligned CPU VA window for both parts, display mapped at its start
            win = self.size + GRAN
            base = libc.mmap(None, win, _PROT_NONE, _MAP_PRIVATE | _MAP_ANON, -1, 0)
            if base in (None, ctypes.c_void_p(-1).value):
                raise OSError(ctypes.get_errno(), "mmap window")
            start = -(-base // GRAN) * GRAN
            if start > base:
                libc.munmap(ctypes.c_void_p(base), start - base)
            end = start + self.size
            if base + win > end:
                libc.munmap(ctypes.c_void_p(end), base + win - end)
            if test_anon:
                got = libc.mmap(ctypes.c_void_p(start), display_bytes, _PROT_RW, _MAP_PRIVATE | _MAP_ANON | _MAP_FIXED,
                                -1, 0)
            else:
                got = libc.mmap(ctypes.c_void_p(start), display_bytes, _PROT_RW, _MAP_SHARED | _MAP_FIXED,
                                self.fd, offset)
            if got != start:
                raise OSError(ctypes.get_errno(), "mmap display buffer")
            self.addr = start
            flags = _CU_MEMHOSTREGISTER_DEVICEMAP | (0 if test_anon else _CU_MEMHOSTREGISTER_IOMEMORY)
            cu.check(cu.lib.cuMemHostRegister_v2(ctypes.c_void_p(start), display_bytes, flags),
                     "cuMemHostRegister(display)")
            dptr = ctypes.c_uint64()
            cu.check(cu.lib.cuMemHostGetDevicePointer_v2(ctypes.byref(dptr), ctypes.c_void_p(start), 0),
                     "cuMemHostGetDevicePointer")
            if dptr.value != start:
                raise RuntimeError(f"{ENV}: device pointer {dptr.value:#x} differs from the mapping {start:#x}")
            # 3) ordinary memory right behind it. Default "pinned" (20.10., third attempt): anonymous memory mapped at
            #    the fixed address and registered like the display part - both device pointers equal their host
            #    addresses, so the range is contiguous for the GPU too. "vmm" (cuMemAddressReserve at the address)
            #    failed on GB10: CUDA ignored the requested address.
            self.mem_handle = None
            mode = os.environ.get("VLLM_DSV41_DISPLAY_KV_TAIL", "pinned")
            if tail and mode == "pinned":
                tstart = start + display_bytes
                got = libc.mmap(ctypes.c_void_p(tstart), tail, _PROT_RW, _MAP_PRIVATE | _MAP_ANON | _MAP_FIXED, -1, 0)
                if got != tstart:
                    raise OSError(ctypes.get_errno(), "mmap pinned tail")
                cu.check(cu.lib.cuMemHostRegister_v2(ctypes.c_void_p(tstart), tail, _CU_MEMHOSTREGISTER_DEVICEMAP),
                         "cuMemHostRegister(tail)")
                tptr = ctypes.c_uint64()
                cu.check(cu.lib.cuMemHostGetDevicePointer_v2(ctypes.byref(tptr), ctypes.c_void_p(tstart), 0),
                         "cuMemHostGetDevicePointer(tail)")
                if tptr.value != tstart:
                    raise RuntimeError(f"{ENV}: tail device pointer {tptr.value:#x} differs from {tstart:#x}")
                self.tail_mode = "pinned"
            elif tail:
                self.tail_mode = "vmm"
                libc.munmap(ctypes.c_void_p(start + display_bytes), tail)   # free the CPU VA for CUDA's reservation
                va = ctypes.c_uint64()
                cu.check(cu.lib.cuMemAddressReserve(ctypes.byref(va), tail, GRAN, start + display_bytes, 0),
                         "cuMemAddressReserve")
                if va.value != start + display_bytes:
                    cu.lib.cuMemAddressFree(va.value, tail)
                    raise RuntimeError(f"{ENV}: VA reservation landed at {va.value:#x}, not right behind the display")
                prop = _CUmemAllocationProp()
                prop.type = _CU_MEM_ALLOCATION_TYPE_PINNED
                prop.location.type = _CU_MEM_LOCATION_TYPE_DEVICE
                prop.location.id = device_index
                h = ctypes.c_ulonglong()
                cu.check(cu.lib.cuMemCreate(ctypes.byref(h), tail, ctypes.byref(prop), 0), "cuMemCreate")
                self.mem_handle = h.value
                cu.check(cu.lib.cuMemMap(va.value, tail, 0, h.value, 0), "cuMemMap")
                acc = _CUmemAccessDesc()
                acc.location.type = _CU_MEM_LOCATION_TYPE_DEVICE
                acc.location.id = device_index
                acc.flags = _CU_MEM_ACCESS_FLAGS_PROT_READWRITE
                cu.check(cu.lib.cuMemSetAccess(va.value, tail, ctypes.byref(acc), 1), "cuMemSetAccess")
        except BaseException:
            if self.fd >= 0:
                os.close(self.fd)
            raise
        _KEEP.append(self)

    @property
    def __cuda_array_interface__(self):
        return {"shape": (self.nbytes,), "typestr": "|i1", "data": (self.addr, False), "version": 3}


def kv_backing(buf_size: int, device):
    """The KV backing tensor (int8, ``buf_size`` bytes, zeroed), or None when display KV is off."""
    disp = requested_bytes()
    if not disp:
        return None
    import torch

    if buf_size <= disp:
        disp = -(-buf_size // GRAN) * GRAN
    index = device.index if getattr(device, "index", None) is not None else torch.cuda.current_device()
    b = Backing(buf_size, disp, index)
    t = torch.as_tensor(b, device=device)
    if t.data_ptr() != b.addr or t.numel() != buf_size:
        raise RuntimeError(f"{ENV}: torch did not keep the backing pointer/size")
    t.zero_()
    torch.cuda.synchronize()
    print(f"[dsv41] display-backed KV: {buf_size / 2**30:.2f} GiB backing = {disp / 2**30:.2f} GiB display "
          f"({b.card}) + {b.tail / 2**30:.2f} GiB ordinary memory ({getattr(b, 'tail_mode', '-')})", flush=True)
    return t


# ---------------------------------------------------------------------------------------------------------------------
# Patch 0054 (VLLM_DSV41_DISPLAY_TENSORS=1, default off): rarely and contiguously read model tensors into the display
# reservation instead of the KV cache. Probe 20.10.: display memory reads 162 GB/s sequential (cudaMalloc 244) but only
# 12 GB/s for scattered 584-byte records (cudaMalloc 79) - wrong for paged KV, fine for tensors read a few whole rows at
# a time: the token embedding (vocab-parallel, a 10 KB row per input token) and the DSpark Markov-bias cache (a 129 KB
# row per draft position). The freed ordinary memory goes to the KV pool (raise --kv-cache-memory-bytes accordingly).

def _kernel_copy(src, dst) -> None:
    """Bit-exact copy by a GPU kernel (no copy engine on the display mapping)."""
    import torch

    s = src.contiguous().view(torch.uint8).view(-1)
    d = dst.view(torch.uint8).view(-1)
    torch.bitwise_or(s, 0, out=d)


def move_to_display(target_model, draft_model, device) -> int:
    """Move the target's token embedding and the drafter's Markov cache into one display mapping; returns the bytes
    moved (0 when nothing was found). Same values, other memory: results are bit-identical."""
    import torch

    from vllm.model_executor.layers.vocab_parallel_embedding import ParallelLMHead, VocabParallelEmbedding

    items = []   # (label, getter, setter)
    seen = set()
    for model in (target_model, draft_model):
        if model is None:
            continue
        for name, mod in model.named_modules():
            if isinstance(mod, VocabParallelEmbedding) and not isinstance(mod, ParallelLMHead):
                w = getattr(mod, "weight", None)
                if isinstance(w, torch.Tensor) and w.is_cuda and id(w) not in seen and w.dtype.is_floating_point:
                    seen.add(id(w))
                    items.append((f"embedding {name}", w))
    if draft_model is not None:
        for attr in ("_markov_cache_local", "_markov_cache"):
            c = getattr(draft_model, attr, None)
            if c is not None and isinstance(getattr(c, "cache", None), torch.Tensor) and id(c.cache) not in seen:
                seen.add(id(c.cache))
                items.append((f"markov {attr}", c))
    if not items:
        return 0

    def tensor_of(obj):
        return obj.cache if not isinstance(obj, torch.Tensor) else obj

    need = 0
    for _, obj in items:
        t = tensor_of(obj)
        need = (need + 255) // 256 * 256 + t.numel() * t.element_size()
    disp = -(-need // GRAN) * GRAN
    if disp > MAX_MIB * MIB:
        raise RuntimeError(f"{ENV}: {need} B do not fit the display reservation")
    index = device.index if getattr(device, "index", None) is not None else torch.cuda.current_device()
    b = Backing(disp, disp, index)
    flat = torch.as_tensor(b, device=device)
    if flat.data_ptr() != b.addr:
        raise RuntimeError(f"{ENV}: torch did not keep the display pointer")
    off, moved, report = 0, 0, []
    for label, obj in items:
        t = tensor_of(obj)
        off = (off + 255) // 256 * 256
        nb = t.numel() * t.element_size()
        view = flat.narrow(0, off, nb).view(t.dtype).view(t.shape)
        _kernel_copy(t, view)
        torch.cuda.synchronize()
        if not torch.equal(view.view(torch.uint8), t.contiguous().view(torch.uint8)):
            raise RuntimeError(f"{ENV}: copy of {label} into display memory differs")
        if isinstance(obj, torch.nn.Parameter):
            obj.data = view
        elif isinstance(obj, torch.Tensor):
            obj.set_(view)
        else:
            obj.cache = view
        off += nb
        moved += nb
        report.append(f"{label} {nb / MIB:.0f} MiB")
    del t
    torch.cuda.empty_cache()
    print(f"[dsv41] display tensors ({b.card}, {disp / MIB:.0f} MiB mapped): " + ", ".join(report), flush=True)
    return moved
