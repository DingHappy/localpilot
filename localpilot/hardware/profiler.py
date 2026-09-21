from __future__ import annotations

import importlib.util
import os
import platform
import shutil
import subprocess
from pathlib import Path
from typing import Any, Dict, List, Optional, Tuple

from localpilot.engines.registry import EngineRegistry
from localpilot.schemas import DeviceInfo, HardwareProfile
from localpilot.utils import config_file, load_data_file, stable_hash


def _run(command: List[str], timeout: float = 4.0) -> Optional[str]:
    try:
        result = subprocess.run(
            command,
            check=True,
            capture_output=True,
            text=True,
            timeout=timeout,
        )
    except (OSError, subprocess.SubprocessError):
        return None
    output = result.stdout.strip()
    return output or None


def _native_machine() -> str:
    return _run(["uname", "-m"]) or platform.machine() or "unknown"


def _cpu_model(native_machine: str) -> str:
    if platform.system() == "Darwin":
        brand = _run(["sysctl", "-n", "machdep.cpu.brand_string"])
        if brand:
            return brand
        if native_machine == "arm64":
            return "Apple Silicon (arm64)"
    model = platform.processor().strip()
    if model and model.lower() not in {"i386", "unknown"}:
        return model
    if platform.system() == "Linux":
        try:
            with open("/proc/cpuinfo", encoding="utf-8") as handle:
                for line in handle:
                    lowered = line.lower()
                    if lowered.startswith("model name"):
                        return line.split(":", 1)[1].strip()
                    if lowered.startswith("cpu part"):
                        return f"ARM CPU part {line.split(':', 1)[1].strip()}"
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


def _device_tree_model() -> Optional[str]:
    path = Path("/proc/device-tree/model")
    try:
        return path.read_text(errors="ignore").replace("\x00", "").strip() or None
    except OSError:
        return None


def _cuda_via_pynvml() -> Tuple[Optional[Dict[str, Any]], List[DeviceInfo]]:
    if importlib.util.find_spec("pynvml") is None:
        return None, []
    try:
        import pynvml

        pynvml.nvmlInit()
    except Exception:
        return None, []

    try:
        driver = pynvml.nvmlSystemGetDriverVersion()
        if isinstance(driver, bytes):
            driver = driver.decode()
        count = pynvml.nvmlDeviceGetCount()
        devices: List[DeviceInfo] = []
        total_memory_gb = 0.0
        names = []
        for index in range(count):
            handle = pynvml.nvmlDeviceGetHandleByIndex(index)
            name = pynvml.nvmlDeviceGetName(handle)
            if isinstance(name, bytes):
                name = name.decode()
            info = pynvml.nvmlDeviceGetMemoryInfo(handle)
            memory_gb = round(info.total / (1024**3), 2)
            total_memory_gb += memory_gb
            names.append(name)
            try:
                major, minor = pynvml.nvmlDeviceGetCudaComputeCapability(handle)
                capability = f"{major}.{minor}"
            except Exception:
                capability = None
            try:
                utilization = pynvml.nvmlDeviceGetUtilizationRates(handle).gpu
            except Exception:
                utilization = None
            devices.append(
                DeviceInfo(
                    id=f"CUDA.{index}" if count > 1 else "CUDA",
                    kind="gpu",
                    name=name,
                    available=True,
                    properties={
                        "index": index,
                        "memory_total_gb": memory_gb,
                        "memory_free_gb": round(info.free / (1024**3), 2),
                        "compute_capability": capability,
                        "utilization_percent": utilization,
                    },
                )
            )
        return (
            {
                "detected": bool(devices),
                "source": "pynvml",
                "driver_version": driver,
                "device_count": count,
                "names": names,
                "memory_total_gb": round(total_memory_gb, 2) or None,
            },
            devices,
        )
    except Exception as exc:
        return (
            {
                "detected": False,
                "source": "pynvml",
                "error": f"{type(exc).__name__}: {exc}",
            },
            [],
        )
    finally:
        try:
            import pynvml

            pynvml.nvmlShutdown()
        except Exception:
            pass


def _cuda_via_smi() -> Tuple[Optional[Dict[str, Any]], List[DeviceInfo]]:
    if shutil.which("nvidia-smi") is None:
        return None, []
    output = _run(
        [
            "nvidia-smi",
            "--query-gpu=index,name,memory.total,memory.free,"
            "compute_cap,utilization.gpu,driver_version",
            "--format=csv,noheader,nounits",
        ],
        timeout=10.0,
    )
    if not output:
        return None, []

    devices: List[DeviceInfo] = []
    names: List[str] = []
    total_memory_gb = 0.0
    driver = None
    rows = [line for line in output.splitlines() if line.strip()]
    for line in rows:
        parts = [part.strip() for part in line.split(",")]
        if len(parts) < 7:
            continue
        index, name, total_mib, free_mib, capability, utilization, driver = parts[:7]
        try:
            memory_gb = round(float(total_mib) / 1024, 2)
            free_gb = round(float(free_mib) / 1024, 2)
        except ValueError:
            memory_gb, free_gb = 0.0, 0.0
        total_memory_gb += memory_gb
        names.append(name)
        devices.append(
            DeviceInfo(
                id=f"CUDA.{index}" if len(rows) > 1 else "CUDA",
                kind="gpu",
                name=name,
                available=True,
                properties={
                    "index": int(index) if index.isdigit() else index,
                    "memory_total_gb": memory_gb,
                    "memory_free_gb": free_gb,
                    "compute_capability": capability,
                    "utilization_percent": (
                        float(utilization) if utilization.replace(".", "").isdigit()
                        else None
                    ),
                },
            )
        )
    return (
        {
            "detected": bool(devices),
            "source": "nvidia-smi",
            "driver_version": driver,
            "device_count": len(devices),
            "names": names,
            "memory_total_gb": round(total_memory_gb, 2) or None,
        },
        devices,
    )


def _detect_cuda() -> Tuple[Dict[str, Any], List[DeviceInfo]]:
    for probe in (_cuda_via_pynvml, _cuda_via_smi):
        accelerator, devices = probe()
        if accelerator and accelerator.get("detected"):
            return accelerator, devices
    return {"detected": False, "source": None, "device_count": 0, "names": []}, []


def _cuda_toolkit() -> Dict[str, Any]:
    version = None
    output = _run(["nvcc", "--version"])
    if output:
        for token in output.replace(",", " ").split():
            if token.startswith("V") and token[1:2].isdigit():
                version = token[1:]
                break
    torch_cuda = None
    if importlib.util.find_spec("torch") is not None:
        torch_cuda = "installed"
    return {
        "nvcc_version": version,
        "nvcc_path": shutil.which("nvcc"),
        "torch": torch_cuda,
        "container_runtime": shutil.which("docker") or shutil.which("podman"),
    }


class HardwareProfiler:
    def __init__(
        self,
        engine_registry: EngineRegistry = None,
        devices_path: Path = None,
    ) -> None:
        self.engines = engine_registry or EngineRegistry()
        self.devices_path = devices_path or config_file("devices.yaml")

    def _device_config(self) -> Dict[str, Any]:
        try:
            return load_data_file(self.devices_path)
        except (OSError, ValueError, RuntimeError):
            return {}

    def _identify_platform(
        self,
        accelerator: Dict[str, Any],
        native_machine: str,
        config: Dict[str, Any],
    ) -> Tuple[str, Dict[str, Any], Dict[str, Any]]:
        """Identifies the platform, and reports what it decided on.

        Which platform this is changes the memory budget from board VRAM to
        system memory, so a silent misidentification quietly rewrites every
        gating decision. The evidence travels with the answer so `doctor`
        can show why it landed where it did.

        Only the GPU name and the device tree count as evidence. An aarch64
        CPU does not: Jetson is also aarch64 with CUDA, and its memory
        behaves differently.
        """
        platforms = config.get("platforms", {})
        observed_names = accelerator.get("names") or []
        gpu_names = " ".join(observed_names).upper()
        tree_model = _device_tree_model() or ""

        signals = {
            "gpu_names": observed_names,
            "device_tree_model": tree_model or None,
            "architecture": native_machine,
            "matched_on": None,
        }

        for platform_id, spec in platforms.items():
            detect = spec.get("detect") or {}
            if not detect:
                continue
            for token in detect.get("gpu_name_contains", []):
                if token.upper() in gpu_names:
                    signals["matched_on"] = f"gpu_name contains '{token}'"
                    return platform_id, spec, signals
            for token in detect.get("device_tree_model_contains", []):
                if token.upper() in tree_model.upper():
                    signals["matched_on"] = f"device_tree contains '{token}'"
                    return platform_id, spec, signals

        if accelerator.get("detected"):
            signals["matched_on"] = "fallback: a CUDA device with no platform match"
            return "cuda_discrete", platforms.get("cuda_discrete", {}), signals
        if platform.system() == "Darwin" and native_machine == "arm64":
            signals["matched_on"] = "Darwin arm64 with no CUDA device"
            return "apple_silicon", platforms.get("apple_silicon", {}), signals
        signals["matched_on"] = "no CUDA device detected"
        return "unknown", {}, signals

    def profile(self, simulate: bool = False) -> HardwareProfile:
        config = self._device_config()
        native_machine = _native_machine()
        cpu_model = _cpu_model(native_machine)
        memory = _memory()
        disk_usage = shutil.disk_usage(os.getcwd())
        engine_reports = self.engines.probe_all()
        notes: List[str] = []

        if simulate:
            platform_id = config.get("mock_platform", "dgx_spark")
            platform_spec = (config.get("platforms", {}) or {}).get(
                platform_id, {}
            )
            accelerator = {
                "detected": True,
                "source": "simulated",
                "driver_version": None,
                "device_count": 1,
                "names": [platform_spec.get("display_name", "Simulated NVIDIA GPU")],
                "memory_total_gb": 128.0,
                "compute_capability": platform_spec.get("compute_capability"),
            }
            devices = [
                DeviceInfo(
                    id="CUDA",
                    kind="gpu",
                    name=f"SIMULATED {platform_spec.get('display_name', 'CUDA device')}",
                    available=True,
                    properties={
                        "memory_total_gb": 128.0,
                        "compute_capability": platform_spec.get(
                            "compute_capability"
                        ),
                        "utilization_percent": None,
                    },
                ),
                DeviceInfo(
                    id="CPU",
                    kind="cpu",
                    name="SIMULATED Grace CPU",
                    available=True,
                ),
            ]
            available_devices = list(
                config.get("mock_available_devices", ["CUDA", "CPU"])
            )
            memory = dict(memory)
            memory["total_gb"] = 128.0
            memory["available_gb"] = 120.0
            memory["simulated"] = True
            unified_memory = bool(platform_spec.get("unified_memory", True))
            bandwidth = platform_spec.get("memory_bandwidth_gbps")
            engines_view = {
                engine_id: {**report, "available": True, "simulated": True}
                for engine_id, report in engine_reports.items()
            }
            notes.append(
                "Development simulation only; every number is modelled, not measured."
            )
            notes.append(
                f"Simulating platform '{platform_id}' with "
                f"{memory['total_gb']} GB unified memory at "
                f"{bandwidth} GB/s."
            )
            real_ready = False
            stack = {
                "cuda": {"nvcc_version": None, "simulated": True},
                "platform": platform_id,
                "engines_available": sorted(engines_view),
            }
        else:
            accelerator, devices = _detect_cuda()
            platform_id, platform_spec, detection = self._identify_platform(
                accelerator, native_machine, config
            )
            unified_memory = bool(platform_spec.get("unified_memory", False))
            bandwidth = platform_spec.get("memory_bandwidth_gbps")
            if platform_id == "apple_silicon":
                accelerator = {
                    "detected": True,
                    "source": "metal",
                    "driver_version": None,
                    "device_count": 1,
                    "names": [cpu_model],
                    "memory_total_gb": memory.get("total_gb"),
                    "unified": True,
                }
                devices = [
                    DeviceInfo(
                        id="METAL",
                        kind="gpu",
                        name=f"{cpu_model} GPU",
                        available=True,
                        properties={"memory_total_gb": memory.get("total_gb")},
                    ),
                    DeviceInfo(id="CPU", kind="cpu", name=cpu_model, available=True),
                ]
                available_devices = ["METAL", "CPU"]
            elif devices:
                devices = devices + [
                    DeviceInfo(
                        id="CPU", kind="cpu", name=cpu_model, available=True
                    )
                ]
                available_devices = [device.id for device in devices]
            else:
                devices = [
                    DeviceInfo(
                        id="CPU", kind="cpu", name=cpu_model, available=True
                    )
                ]
                available_devices = ["CPU"]
            engines_view = engine_reports
            stack = {
                "cuda": _cuda_toolkit(),
                "metal": {
                    "available": platform_id == "apple_silicon",
                    "unified_memory": platform_id == "apple_silicon",
                },
                "platform": platform_id,
                "platform_detection": detection,
                "engines_available": sorted(
                    engine_id
                    for engine_id, report in engine_reports.items()
                    if report["available"]
                ),
            }

            servable = [
                engine_id
                for engine_id in stack["engines_available"]
                if self.engines.servable(engine_id)
            ]
            real_ready = bool(accelerator.get("detected")) and bool(servable)

            if platform_id == "apple_silicon":
                notes.append(
                    "Apple Silicon detected: local inference uses the Metal "
                    "unified-memory path and a servable local engine such as Ollama."
                )
            elif not accelerator.get("detected"):
                notes.append(
                    "No CUDA device detected. Install the driver or run with "
                    "--mode mock for orchestration development."
                )
            if not servable:
                notes.append(
                    "No servable inference engine found. Install one of vLLM, "
                    "TensorRT-LLM, SGLang or NIM before a real benchmark."
                )
            if platform_id == "dgx_spark":
                notes.append(
                    "DGX Spark detected: capacity is large and bandwidth is "
                    "narrow, so decode is bandwidth-bound. Sparse models and "
                    "speculative decoding are ranked accordingly."
                )
            elif platform_id == "cuda_discrete":
                observed = ", ".join(accelerator.get("names") or []) or "unknown"
                notes.append(
                    "Discrete CUDA GPU: the memory ceiling is board VRAM, not "
                    "host RAM. If this machine really has unified memory the "
                    "budget is now wrong -- the GPU reported itself as "
                    f"'{observed}', so add a matching token to "
                    "config/devices.yaml under the right platform's "
                    "gpu_name_contains."
                )

        if unified_memory and memory.get("total_gb"):
            accelerator = dict(accelerator)
            accelerator["memory_total_gb"] = memory["total_gb"]
            accelerator["unified"] = True

        fingerprint_input = {
            "os": platform.system(),
            "os_release": platform.release(),
            "machine": native_machine,
            "cpu_model": cpu_model,
            "memory_total_gb": memory.get("total_gb"),
            "devices": [device.name for device in devices],
            "platform_id": stack.get("platform"),
            "driver": accelerator.get("driver_version"),
            "engines": stack.get("engines_available"),
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
            },
            accelerator=accelerator,
            memory=memory,
            disk={
                "total_gb": round(disk_usage.total / (1024**3), 2),
                "free_gb": round(disk_usage.free / (1024**3), 2),
            },
            stack=stack,
            engines=engines_view,
            available_devices=available_devices,
            devices=devices,
            fingerprint=stable_hash(fingerprint_input),
            platform_id=stack.get("platform", "unknown"),
            unified_memory=unified_memory,
            memory_bandwidth_gbps=bandwidth,
            simulated=simulate,
            real_execution_ready=real_ready,
            notes=notes,
        )
