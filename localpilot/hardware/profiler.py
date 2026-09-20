from __future__ import annotations

import importlib.util
import os
import platform
import shutil
import subprocess
from typing import Any, Dict, List, Tuple

from localpilot.schemas import DeviceInfo, HardwareProfile
from localpilot.utils import stable_hash


def _native_machine() -> str:
    try:
        result = subprocess.run(
            ["uname", "-m"],
            check=True,
            capture_output=True,
            text=True,
            timeout=2,
        )
        if result.stdout.strip():
            return result.stdout.strip()
    except (OSError, subprocess.SubprocessError):
        pass
    return platform.machine() or "unknown"


def _cpu_model(native_machine: str) -> str:
    if platform.system() == "Darwin":
        try:
            result = subprocess.run(
                ["sysctl", "-n", "machdep.cpu.brand_string"],
                check=True,
                capture_output=True,
                text=True,
                timeout=2,
            )
            if result.stdout.strip():
                return result.stdout.strip()
        except (OSError, subprocess.SubprocessError):
            pass
        if native_machine == "arm64":
            return "Apple Silicon (arm64)"
    model = platform.processor().strip()
    if model and model.lower() not in {"i386", "unknown"}:
        return model
    if platform.system() == "Linux":
        try:
            with open("/proc/cpuinfo", encoding="utf-8") as handle:
                for line in handle:
                    if line.lower().startswith("model name"):
                        return line.split(":", 1)[1].strip()
        except OSError:
            pass
    return platform.machine() or "unknown"


def _memory() -> Dict[str, Any]:
    try:
        import psutil

        memory = psutil.virtual_memory()
        return {
            "total_gb": round(memory.total / (1024**3), 2),
            "available_gb": round(memory.available / (1024**3), 2),
            "used_percent": float(memory.percent),
        }
    except ImportError:
        total = None
        if hasattr(os, "sysconf"):
            try:
                total = os.sysconf("SC_PAGE_SIZE") * os.sysconf("SC_PHYS_PAGES")
            except (ValueError, OSError):
                total = None
        return {
            "total_gb": round(total / (1024**3), 2) if total else None,
            "available_gb": None,
            "used_percent": None,
        }


def _cpu_workload() -> Dict[str, Any]:
    try:
        import psutil

        return {"usage_percent": float(psutil.cpu_percent(interval=0.05))}
    except ImportError:
        try:
            one, five, fifteen = os.getloadavg()
            return {"load_average": [one, five, fifteen], "usage_percent": None}
        except (AttributeError, OSError):
            return {"usage_percent": None}


def _openvino_devices() -> Tuple[Dict[str, Any], List[DeviceInfo]]:
    if importlib.util.find_spec("openvino") is None:
        return (
            {"installed": False, "version": None, "available_devices": []},
            [],
        )

    try:
        import openvino

        core = openvino.Core()
        available = list(core.available_devices)
        devices = []
        for device_id in available:
            try:
                name = str(core.get_property(device_id, "FULL_DEVICE_NAME"))
            except Exception:
                name = device_id
            kind = device_id.split(".", 1)[0].lower()
            devices.append(
                DeviceInfo(
                    id=device_id,
                    kind=kind,
                    name=name,
                    available=True,
                    properties={"utilization_percent": None},
                )
            )
        version = getattr(openvino, "__version__", None)
        return (
            {
                "installed": True,
                "version": version,
                "available_devices": available,
            },
            devices,
        )
    except Exception as exc:
        return (
            {
                "installed": True,
                "version": None,
                "available_devices": [],
                "error": f"{type(exc).__name__}: {exc}",
            },
            [],
        )


class HardwareProfiler:
    def profile(self, simulate: bool = False) -> HardwareProfile:
        native_machine = _native_machine()
        cpu_model = _cpu_model(native_machine)
        memory = _memory()
        disk_usage = shutil.disk_usage(os.getcwd())
        openvino_info, devices = _openvino_devices()
        notes = []

        if simulate:
            devices = [
                DeviceInfo("CPU", "cpu", "Simulated Intel CPU", True),
                DeviceInfo("GPU", "gpu", "Simulated Intel GPU", True),
                DeviceInfo("NPU", "npu", "Simulated Intel NPU", True),
            ]
            available_devices = ["CPU", "GPU", "NPU"]
            notes.append(
                "Development simulation only; results are not hardware measurements."
            )
        else:
            available_devices = [device.id for device in devices]

        cpu_is_intel = "intel" in cpu_model.lower()
        has_ov_device = bool(available_devices)
        real_ready = (not simulate) and cpu_is_intel and has_ov_device
        if not cpu_is_intel:
            notes.append("Intel target hardware was not detected.")
        if not openvino_info.get("installed"):
            notes.append("OpenVINO is not installed.")

        gpu_devices = [d for d in devices if d.kind == "gpu"]
        npu_devices = [d for d in devices if d.kind == "npu"]
        fingerprint_input = {
            "os": platform.system(),
            "os_release": platform.release(),
            "machine": native_machine,
            "cpu_model": cpu_model,
            "memory_total_gb": memory.get("total_gb"),
            "devices": [d.name for d in devices],
            "openvino_version": openvino_info.get("version"),
            "simulated": simulate,
        }

        return HardwareProfile(
            os={
                "name": platform.system(),
                "release": platform.release(),
                "version": platform.version(),
                "machine": native_machine,
            },
            cpu={
                "model": cpu_model,
                "architecture": native_machine,
                "logical_cores": os.cpu_count(),
                "workload": _cpu_workload(),
                "intel": cpu_is_intel,
            },
            gpu={
                "available": bool(gpu_devices),
                "devices": [d.to_dict() for d in gpu_devices],
                "workload": {"usage_percent": None, "status": "unsupported"},
            },
            npu={
                "available": bool(npu_devices),
                "devices": [d.to_dict() for d in npu_devices],
                "workload": {"usage_percent": None, "status": "unsupported"},
            },
            memory=memory,
            disk={
                "total_gb": round(disk_usage.total / (1024**3), 2),
                "free_gb": round(disk_usage.free / (1024**3), 2),
            },
            openvino=openvino_info,
            available_devices=available_devices,
            devices=devices,
            fingerprint=stable_hash(fingerprint_input),
            simulated=simulate,
            real_execution_ready=real_ready,
            notes=notes,
        )
