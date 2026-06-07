"""GPU detection, auto-configuration, and enablement guidance for the reduction tiers.

Detection is layered and side-effect-free: an NVIDIA driver via ``nvidia-smi`` (no Python
dependency), then ``cupy`` plus a usable CUDA device, then the device specifications. From
that the GPU tier auto-configures -- enabled only when a device is actually usable, with a
VRAM budget derived from the card -- and, when the hardware is present but the CUDA Python
stack is not, it produces concrete install steps. Nothing about any particular card is
assumed: name, compute capability, and memory are all queried at runtime.
"""
from __future__ import annotations

import dataclasses
import re
import shutil
import subprocess

_VRAM_BUDGET_FRACTION = 0.8         # leave headroom; the router won't fill the whole card


@dataclasses.dataclass(frozen=True)
class GpuDevice:
    name: str
    compute_capability: str | None
    memory_total_mib: int | None
    memory_free_mib: int | None
    multiprocessors: int | None = None


@dataclasses.dataclass(frozen=True)
class GpuInfo:
    driver_present: bool                # nvidia-smi found and responded
    driver_version: str | None
    driver_cuda_version: str | None     # max CUDA the driver supports (nvidia-smi header)
    cupy_installed: bool
    device_usable: bool                 # cupy can actually run on a device
    devices: tuple[GpuDevice, ...]
    detail: str | None                  # why unusable, when applicable


@dataclasses.dataclass(frozen=True)
class GpuConfig:
    enabled: bool
    device_id: int | None
    vram_budget_mib: int | None
    compute_capability: str | None


def _run(cmd) -> str | None:
    try:
        out = subprocess.run(cmd, capture_output=True, text=True, timeout=10)
    except Exception:
        return None
    return out.stdout if out.returncode == 0 else None


def _to_int(s):
    try:
        return int(float(s))
    except (TypeError, ValueError):
        return None


def _probe_nvidia_smi():
    """``(devices, driver_version, driver_cuda_version)`` from nvidia-smi, or None when the
    driver tool is absent. Works without cupy, so it can spot a GPU whose software stack
    is incomplete."""
    if not shutil.which("nvidia-smi"):
        return None
    devices: list[GpuDevice] = []
    driver_version = None
    out = _run(["nvidia-smi", "--query-gpu=name,memory.total,memory.free,driver_version",
                "--format=csv,noheader,nounits"])
    caps = _run(["nvidia-smi", "--query-gpu=compute_cap", "--format=csv,noheader"])
    cap_list = [c.strip() for c in caps.strip().splitlines()] if caps else []
    if out:
        for idx, line in enumerate(out.strip().splitlines()):
            parts = [p.strip() for p in line.split(",")]
            if len(parts) < 4:
                continue
            name, mtot, mfree, drv = parts[:4]
            driver_version = drv or driver_version
            devices.append(GpuDevice(
                name=name,
                compute_capability=cap_list[idx] if idx < len(cap_list) else None,
                memory_total_mib=_to_int(mtot),
                memory_free_mib=_to_int(mfree),
            ))
    cuda_version = None
    header = _run(["nvidia-smi"])
    if header:
        m = re.search(r"CUDA Version:\s*([0-9]+\.[0-9]+)", header)
        if m:
            cuda_version = m.group(1)
    return devices, driver_version, cuda_version


def _probe_cupy():
    """``(cupy_installed, device_usable, detail, devices)``."""
    try:
        import cupy
    except ImportError:
        return False, False, None, []
    try:
        count = cupy.cuda.runtime.getDeviceCount()
    except Exception as exc:
        return True, False, f"cupy is installed but the CUDA runtime is unusable ({exc})", []
    if count == 0:
        return True, False, "cupy is installed but no CUDA device is visible", []
    devices = []
    for i in range(count):
        props = cupy.cuda.runtime.getDeviceProperties(i)
        name = props["name"]
        if isinstance(name, bytes):
            name = name.decode(errors="replace")
        free, total = cupy.cuda.Device(i).mem_info
        devices.append(GpuDevice(
            name=name,
            compute_capability=f"{props['major']}.{props['minor']}",
            memory_total_mib=total // (1024 * 1024),
            memory_free_mib=free // (1024 * 1024),
            multiprocessors=props.get("multiProcessorCount"),
        ))
    return True, True, None, devices


def detect_gpu() -> GpuInfo:
    """Probe driver, cupy, and device specs into a single GpuInfo. Never raises."""
    smi = _probe_nvidia_smi()
    smi_devices, driver_version, driver_cuda = smi if smi else ([], None, None)
    cupy_installed, device_usable, detail, cupy_devices = _probe_cupy()
    devices = tuple(cupy_devices) if device_usable else tuple(smi_devices)
    return GpuInfo(
        driver_present=smi is not None,
        driver_version=driver_version,
        driver_cuda_version=driver_cuda,
        cupy_installed=cupy_installed,
        device_usable=device_usable,
        devices=devices,
        detail=detail,
    )


def usable_cupy():
    """The cupy module iff a CUDA device is actually usable, else None -- the single
    availability check the GPU tier relies on."""
    try:
        import cupy
    except ImportError:
        return None
    try:
        if cupy.cuda.runtime.getDeviceCount() > 0:
            return cupy
    except Exception:
        return None
    return None


def gpu_config(info: GpuInfo | None = None) -> GpuConfig:
    """Auto-configuration from the detected hardware -- enabled only when usable, with the
    device of most free memory chosen and a VRAM budget derived from it. No model assumed."""
    if info is None:
        info = detect_gpu()
    if not info.device_usable or not info.devices:
        return GpuConfig(enabled=False, device_id=None, vram_budget_mib=None,
                         compute_capability=None)
    best = max(range(len(info.devices)),
               key=lambda i: info.devices[i].memory_free_mib or 0)
    dev = info.devices[best]
    budget = int((dev.memory_free_mib or 0) * _VRAM_BUDGET_FRACTION) or None
    return GpuConfig(enabled=True, device_id=best, vram_budget_mib=budget,
                     compute_capability=dev.compute_capability)


def _cupy_wheel(cuda_version: str | None) -> str | None:
    if not cuda_version:
        return None
    return {"12": "cupy-cuda12x", "11": "cupy-cuda11x"}.get(cuda_version.split(".")[0])


def enablement_instructions(info: GpuInfo | None = None) -> str | None:
    """Actionable steps to turn the GPU tier on, or None when it is already usable or no
    NVIDIA hardware is present (CPU tiers cover that case)."""
    if info is None:
        info = detect_gpu()
    if info.device_usable:
        return None
    if not info.driver_present and not info.devices:
        return None
    lines = []
    if info.cupy_installed:
        lines.append(f"cupy is installed but cannot use a device ({info.detail}).")
        lines.append("Make sure the cupy build matches your CUDA version and the driver "
                     "is current, then re-run.")
    elif info.driver_present:
        card = info.devices[0].name if info.devices else "an NVIDIA GPU"
        lines.append(f"{card} with a working driver was found, but cupy (the CUDA Python "
                     "stack) is not installed, so the GPU tier is off.")
        wheel = _cupy_wheel(info.driver_cuda_version)
        if wheel:
            lines.append(f"Enable it:  pip install {wheel}   "
                         f"(matches your driver's CUDA {info.driver_cuda_version})")
        else:
            lines.append("Enable it:  check your CUDA version with nvidia-smi (top-right, "
                         "'CUDA Version'), then install the matching wheel -- cupy-cuda12x "
                         "for CUDA 12.x or cupy-cuda11x for 11.x.")
        lines.append("Re-run afterwards; the GPU tier is detected automatically.")
    else:
        lines.append("An NVIDIA GPU appears present but no driver was detected.")
        lines.append("Install the NVIDIA driver + CUDA, then cupy (cupy-cuda12x or "
                     "cupy-cuda11x to match), and re-run.")
    return "\n".join(lines)


def format_report(info: GpuInfo | None = None) -> str:
    """Human-readable detection summary for the CLI."""
    if info is None:
        info = detect_gpu()
    if not info.driver_present and not info.devices:
        return "GPU detection:\n  no NVIDIA GPU/driver detected -- CPU tiers only"
    lines = ["GPU detection:"]
    drv = f" (version {info.driver_version})" if info.driver_version else ""
    lines.append(f"  driver present:  {info.driver_present}{drv}")
    if info.driver_cuda_version:
        lines.append(f"  driver CUDA max: {info.driver_cuda_version}")
    lines.append(f"  cupy installed:  {info.cupy_installed}")
    lines.append(f"  GPU tier usable: {info.device_usable}")
    for i, d in enumerate(info.devices):
        spec = f"  device {i}: {d.name}"
        if d.compute_capability:
            spec += f", sm_{d.compute_capability.replace('.', '')}"
        if d.memory_total_mib:
            spec += f", {d.memory_total_mib} MiB"
            if d.memory_free_mib is not None:
                spec += f" ({d.memory_free_mib} free)"
        lines.append(spec)
    cfg = gpu_config(info)
    if cfg.enabled:
        lines.append(f"  auto-config:     device {cfg.device_id}, "
                     f"VRAM budget {cfg.vram_budget_mib} MiB")
    return "\n".join(lines)
