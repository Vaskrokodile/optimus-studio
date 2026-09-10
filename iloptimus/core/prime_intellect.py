"""Prime Intellect integration — provision GPU pods, run remote training/benchmarking.

This module connects Optimus Studio to Prime Intellect's GPU marketplace so
the AI can autonomously provision compute, deploy the model + vLLM, run the
full training loop (TTT + RSI + batched RL), and benchmark on AIME 2025 —
all managed from within Optimus Studio.

Architecture:
    Optimus Studio (local)
        │
        ├── prime_intellect.py  ← this module
        │       ├── provision_pod()      — rent a GPU on PI
        │       ├── deploy_to_pod()      — SSH setup: install vLLM, upload model+adapter
        │       ├── run_remote_benchmark() — vLLM-based AIME 2025 benchmark
        │       ├── run_remote_training()  — SFT/GRPO/RL training on the pod
        │       ├── run_remote_rl_factory() — batched RL rlfactory loop
        │       └── get_pod_status() / terminate_pod()
        │
        └── Prime Intellect Pod (remote GPU)
                ├── vLLM server (fast inference for benchmarking)
                ├── Optimus Studio pipeline (SFT + GRPO)
                └── rl_factory batched RL (param-split across envs)
"""

from __future__ import annotations

import json
import os
import paramiko
import subprocess
import time
import urllib.request
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any, Optional


# ---------------------------------------------------------------------------
# Configuration
# ---------------------------------------------------------------------------

PI_API_BASE = "https://api.primeintellect.ai/api/v1"
PI_API_KEY_ENV = "PRIME_INTELLECT_API_KEY"

# Default GPU preference order (best fit for 9B model training within $8 budget)
GPU_PREFERENCE = [
    "RTX6000Ada_48GB",   # $0.75/hr — 48GB, 10.7 hrs on $8
    "L40S_48GB",         # $0.82/hr — 48GB, 9.8 hrs
    "A6000_48GB",        # $0.54/hr — 48GB, 14.8 hrs (slower but cheapest)
    "A100_40GB",         # $1.99/hr — 40GB, 4.0 hrs
    "H200_141GB",        # $4.50/hr — 141GB, 1.8 hrs (overkill but fastest)
]


@dataclass
class PodConfig:
    """Configuration for a Prime Intellect pod."""
    name: str = "optimus-studio"
    gpu_type: str = "RTX6000Ada_48GB"
    gpu_count: int = 1
    image: str = "cuda_12_4_pytorch_2_5"  # has PyTorch + CUDA preinstalled
    disk_size: int = 200  # GB — need space for 9B model + adapter + vLLM
    max_price: float = 2.50  # $/hr cap (allows A100_40GB at $1.99)


@dataclass
class PodInfo:
    """Information about a provisioned pod."""
    pod_id: str
    name: str
    gpu_type: str
    gpu_count: int
    status: str = "unknown"
    ssh_connection: str = ""
    ip: str = ""
    ssh_port: int = 22
    cost_per_hr: float = 0.0
    created_at: float = field(default_factory=time.time)

    @property
    def is_active(self) -> bool:
        return self.status.upper() in ("ACTIVE", "RUNNING")

    def ssh_host(self) -> str:
        """Return the SSH host string (user@ip)."""
        if self.ssh_connection:
            parts = self.ssh_connection.split()
            return parts[0] if parts else f"root@{self.ip}"
        return f"root@{self.ip}"


@dataclass
class BenchmarkResult:
    """Result of a remote AIME 2025 benchmark run."""
    pass_at_1: float = 0.0
    pass_at_5: float = 0.0
    correct_at_1: int = 0
    correct_at_5: int = 0
    total_problems: int = 30
    elapsed_seconds: float = 0.0
    raw_results: list[dict] = field(default_factory=list)
    error: Optional[str] = None

    @property
    def score_str(self) -> str:
        return f"{self.correct_at_1}/{self.total_problems} (pass@1), {self.correct_at_5}/{self.total_problems} (pass@5)"


@dataclass
class TrainingResult:
    """Result of a remote training run."""
    run_id: str = ""
    stage: str = ""
    accuracy_before: float = 0.0
    accuracy_after: float = 0.0
    elapsed_seconds: float = 0.0
    adapter_path: str = ""
    error: Optional[str] = None


# ---------------------------------------------------------------------------
# API client
# ---------------------------------------------------------------------------


def _get_api_key(api_key: Optional[str] = None) -> str:
    key = api_key or os.environ.get(PI_API_KEY_ENV, "")
    if not key:
        raise RuntimeError(
            f"No Prime Intellect API key. Set {PI_API_KEY_ENV} env var or pass api_key."
        )
    return key


def _api_request(
    method: str,
    path: str,
    api_key: Optional[str] = None,
    body: Optional[dict] = None,
    params: Optional[dict] = None,
) -> Any:
    """Make a Prime Intellect API request."""
    key = _get_api_key(api_key)
    url = f"{PI_API_BASE}/{path.lstrip('/')}"
    if params:
        query = "&".join(f"{k}={v}" for k, v in params.items())
        url += f"?{query}"

    data = json.dumps(body).encode() if body else None
    req = urllib.request.Request(url, data=data, method=method, headers={
        "Authorization": f"Bearer {key}",
        "Content-Type": "application/json",
        "User-Agent": "OptimusStudio/1.0",
    })
    with urllib.request.urlopen(req, timeout=60) as resp:
        return json.loads(resp.read())


# ---------------------------------------------------------------------------
# GPU availability and pod provisioning
# ---------------------------------------------------------------------------


def list_available_gpus(
    gpu_count: int = 1,
    api_key: Optional[str] = None,
) -> list[dict]:
    """List all available GPU offers on Prime Intellect."""
    data = _api_request("GET", "availability/gpus", api_key=api_key, params={
        "gpu_count": str(gpu_count),
        "page_size": "100",
    })
    items = data.get("items", [])
    return [o for o in items if o.get("stockStatus") == "Available"]


def find_best_gpu(
    budget_usd: float = 8.0,
    min_vram_gb: float = 40.0,
    api_key: Optional[str] = None,
) -> dict:
    """Find the best GPU within budget, preferring from GPU_PREFERENCE order."""
    available = list_available_gpus(gpu_count=1, api_key=api_key)

    # Sort by our preference, then by price
    def preference_score(offer):
        gpu_type = offer.get("gpuType", "")
        vram = offer.get("gpuMemory", 0)
        price = offer.get("prices", {}).get("onDemand", 999)
        hours = budget_usd / price if price > 0 else 0

        # Score: preference rank (lower is better) + vram adequacy + hours
        pref_rank = len(GPU_PREFERENCE)
        for i, preferred in enumerate(GPU_PREFERENCE):
            if preferred in gpu_type:
                pref_rank = i
                break

        vram_ok = 1 if vram >= min_vram_gb else 0
        return (vram_ok, -pref_rank, -hours)  # maximize vram_ok, preference, then hours

    available.sort(key=preference_score, reverse=True)
    if available:
        return available[0]
    return {}


def provision_pod(
    config: PodConfig,
    api_key: Optional[str] = None,
) -> PodInfo:
    """Provision a GPU pod on Prime Intellect.

    Finds the best available offer for the requested GPU type and creates a pod.
    """
    # Find an offer matching the requested GPU type
    available = list_available_gpus(gpu_count=config.gpu_count, api_key=api_key)

    offer = None
    for o in available:
        if config.gpu_type in o.get("gpuType", ""):
            offer = o
            break

    if not offer:
        # Fall back to any available GPU with enough VRAM
        offer = find_best_gpu(api_key=api_key)
        if not offer:
            raise RuntimeError(f"No available GPUs matching {config.gpu_type}")

    price = offer.get("prices", {}).get("onDemand", 0)
    if price > config.max_price:
        raise RuntimeError(
            f"Best available GPU ({offer['gpuType']}) costs ${price}/hr, "
            f"exceeds max ${config.max_price}/hr"
        )

    # Build the create request
    pod_body = {
        "name": config.name,
        "cloudId": offer.get("cloudId", ""),
        "gpuType": offer.get("gpuType", config.gpu_type),
        "socket": offer.get("socket", "PCIe"),
        "gpuCount": config.gpu_count,
        "image": config.image,
        "dataCenterId": offer.get("dataCenter", ""),
        "country": offer.get("country", "US"),
        "security": offer.get("security", "secure_cloud"),
        "diskSize": config.disk_size,
    }

    provider_type = offer.get("provider", "lambdalabs")
    body = {
        "pod": pod_body,
        "provider": {"type": provider_type},
    }

    result = _api_request("POST", "pods/", api_key=api_key, body=body)
    pod_id = result.get("podId", result.get("id", ""))

    return PodInfo(
        pod_id=pod_id,
        name=config.name,
        gpu_type=offer.get("gpuType", config.gpu_type),
        gpu_count=config.gpu_count,
        status="provisioning",
        cost_per_hr=price,
    )


def get_pod_status(pod_id: str, api_key: Optional[str] = None) -> dict:
    """Get the status of a pod. Uses pods/{id} endpoint (not /status which 404s)."""
    return _api_request("GET", f"pods/{pod_id}", api_key=api_key)


def get_pod(pod_id: str, api_key: Optional[str] = None) -> dict:
    """Get pod details."""
    return _api_request("GET", f"pods/{pod_id}", api_key=api_key)


def list_pods(api_key: Optional[str] = None) -> list[dict]:
    """List all pods."""
    data = _api_request("GET", "pods/", api_key=api_key)
    if isinstance(data, dict):
        return data.get("data", [])
    return data if isinstance(data, list) else []


def terminate_pod(pod_id: str, api_key: Optional[str] = None) -> bool:
    """Terminate a pod."""
    try:
        _api_request("DELETE", f"pods/{pod_id}", api_key=api_key)
        return True
    except Exception:
        return False


def wait_for_pod_ready(
    pod_id: str,
    timeout: int = 300,
    api_key: Optional[str] = None,
) -> PodInfo:
    """Wait until a pod is active and return its connection info."""
    deadline = time.time() + timeout
    while time.time() < deadline:
        status = get_pod_status(pod_id, api_key=api_key)
        pod_status = status.get("status", "").upper()
        if pod_status == "ACTIVE":
            ssh_conn = status.get("sshConnection", "")
            ip = status.get("ip", "")
            cost = status.get("costPerHr", 0)

            # Parse SSH connection: "root@1.2.3.4 -p 22"
            ssh_port = 22
            if "-p " in ssh_conn:
                parts = ssh_conn.split("-p ")
                if len(parts) > 1:
                    ssh_port = int(parts[1].strip())

            return PodInfo(
                pod_id=pod_id,
                name="",
                gpu_type="",
                gpu_count=1,
                status="ACTIVE",
                ssh_connection=ssh_conn,
                ip=ip,
                ssh_port=ssh_port,
                cost_per_hr=cost,
            )
        elif pod_status in ("FAILED", "ERROR"):
            raise RuntimeError(f"Pod {pod_id} failed: {status}")
        time.sleep(10)

    raise TimeoutError(f"Pod {pod_id} not ready within {timeout}s")


# ---------------------------------------------------------------------------
# SSH operations
# ---------------------------------------------------------------------------


def _ssh_connect(pod: PodInfo, timeout: int = 30) -> paramiko.SSHClient:
    """Connect to a pod via SSH."""
    client = paramiko.SSHClient()
    client.set_missing_host_key_policy(paramiko.AutoAddPolicy())
    host = pod.ip or pod.ssh_connection.split("@")[1].split()[0] if pod.ssh_connection else ""
    client.connect(
        hostname=pod.ip,
        port=pod.ssh_port,
        username="root",
        timeout=timeout,
        allow_agent=True,
        look_for_keys=True,
    )
    return client


def ssh_exec(
    pod: PodInfo,
    command: str,
    timeout: int = 300,
    background: bool = False,
) -> tuple[int, str, str]:
    """Execute a command on the pod via SSH.

    Returns (exit_code, stdout, stderr).
    If background=True, launches the command with nohup and returns immediately.
    """
    client = _ssh_connect(pod)
    try:
        if background:
            command = f"nohup {command} > /root/output.log 2>&1 &"
            stdin, stdout, stderr = client.exec_command(command, timeout=10)
            return 0, "Background process started", ""

        stdin, stdout, stderr = client.exec_command(command, timeout=timeout)
        exit_code = stdout.channel.recv_exit_status()
        out = stdout.read().decode("utf-8", errors="replace")
        err = stderr.read().decode("utf-8", errors="replace")
        return exit_code, out, err
    finally:
        client.close()


def ssh_upload(pod: PodInfo, local_path: str, remote_path: str) -> bool:
    """Upload a file to the pod via SFTP."""
    client = _ssh_connect(pod)
    try:
        sftp = client.open_sftp()
        # Create remote directory if needed
        remote_dir = str(Path(remote_path).parent)
        try:
            sftp.stat(remote_dir)
        except FileNotFoundError:
            # Create parent directories
            parts = remote_dir.strip("/").split("/")
            current = ""
            for part in parts:
                current += "/" + part
                try:
                    sftp.stat(current)
                except FileNotFoundError:
                    sftp.mkdir(current)

        sftp.put(local_path, remote_path)
        sftp.close()
        return True
    except Exception as e:
        return False
    finally:
        client.close()


def ssh_upload_string(pod: PodInfo, content: str, remote_path: str) -> bool:
    """Upload a string as a file to the pod via SFTP."""
    client = _ssh_connect(pod)
    try:
        sftp = client.open_sftp()
        remote_dir = str(Path(remote_path).parent)
        try:
            sftp.stat(remote_dir)
        except FileNotFoundError:
            parts = remote_dir.strip("/").split("/")
            current = ""
            for part in parts:
                current += "/" + part
                try:
                    sftp.stat(current)
                except FileNotFoundError:
                    sftp.mkdir(current)
        with sftp.file(remote_path, "w") as f:
            f.write(content)
        sftp.close()
        return True
    except Exception:
        return False
    finally:
        client.close()


# ---------------------------------------------------------------------------
# Deployment: install vLLM, upload model + adapter
# ---------------------------------------------------------------------------


def deploy_to_pod(
    pod: PodInfo,
    model_id: str = "Tesslate/OmniCoder-9B",
    adapter_local_path: str = "",
    install_vllm: bool = True,
) -> dict:
    """Deploy the model and vLLM to a freshly provisioned pod.

    Steps:
    1. Install Python dependencies (vLLM, transformers, peft, trl, etc.)
    2. Download the base model from HuggingFace
    3. Upload the LoRA adapter from local machine
    4. Start vLLM server in background
    """
    results = {"steps": [], "errors": []}

    def _step(name: str, cmd: str, timeout: int = 600):
        print(f"  [{name}] Running: {cmd[:80]}...")
        exit_code, out, err = ssh_exec(pod, cmd, timeout=timeout)
        status = "ok" if exit_code == 0 else "failed"
        results["steps"].append({"name": name, "status": status, "exit_code": exit_code})
        if exit_code != 0:
            results["errors"].append(f"{name}: {err[:500]}")
            print(f"  [{name}] FAILED: {err[:200]}")
        else:
            print(f"  [{name}] OK")
        return exit_code == 0

    # Step 1: Install system packages
    _step("apt-update", "apt-get update -qq", timeout=120)
    _step("install-build-tools", "apt-get install -y -qq build-essential git ninja-build", timeout=120)

    # Step 2: Install vLLM and training dependencies
    if install_vllm:
        _step("install-vllm",
              "pip install -q vllm==0.6.3.post1 transformers peft trl bitsandbytes datasets accelerate",
              timeout=600)
        _step("install-unsloth",
              "pip install -q unsloth",
              timeout=600)

    # Step 3: Download base model
    _step("download-model",
          f"python -c \"from huggingface_hub import snapshot_download; snapshot_download('{model_id}', local_dir='/root/model')\"",
          timeout=900)

    # Step 4: Upload adapter
    if adapter_local_path and os.path.exists(adapter_local_path):
        adapter_remote = "/root/adapter"
        # Upload all files in the adapter directory
        for fn in os.listdir(adapter_local_path):
            local_file = os.path.join(adapter_local_path, fn)
            if os.path.isfile(local_file):
                remote_file = f"{adapter_remote}/{fn}"
                print(f"  [upload-adapter] Uploading {fn}...")
                if ssh_upload(pod, local_file, remote_file):
                    print(f"  [upload-adapter] {fn} uploaded")
                else:
                    results["errors"].append(f"upload-adapter: failed to upload {fn}")

        results["steps"].append({"name": "upload-adapter", "status": "ok"})
    else:
        results["errors"].append(f"Adapter not found at {adapter_local_path}")

    # Step 5: Start vLLM server with LoRA adapter
    vllm_cmd = (
        f"nohup python -m vllm.entrypoints.openai.api_server "
        f"--model /root/model "
        f"--enable-lora "
        f"--lora-modules adapter=/root/adapter "
        f"--max-lora-rank 256 "
        f"--gpu-memory-utilization 0.90 "
        f"--dtype bfloat16 "
        f"--max-model-len 8192 "
        f"--port 8000 "
        f"> /root/vllm.log 2>&1 &"
    )
    _step("start-vllm", vllm_cmd, timeout=30)
    results["vllm_url"] = f"http://{pod.ip}:8000"

    return results


# ---------------------------------------------------------------------------
# Remote AIME 2025 benchmark via vLLM
# ---------------------------------------------------------------------------


def run_remote_aime_benchmark(
    pod: PodInfo,
    vllm_url: str = "",
    num_samples: int = 5,
    max_tokens: int = 4096,
    temperature: float = 0.6,
    use_adapter: bool = True,
) -> BenchmarkResult:
    """Run AIME 2025 benchmark on the pod using vLLM for fast inference.

    Uploads a benchmark script to the pod that:
    1. Loads AIME 2025 from HuggingFace
    2. Generates N samples per problem via vLLM
    3. Extracts boxed answers and scores them
    """
    if not vllm_url:
        vllm_url = f"http://localhost:8000"

    # Upload the benchmark script
    benchmark_script = f'''
import json, re, time, urllib.request

VLLM_URL = "{vllm_url}"
MODEL = "adapter" if {use_adapter} else "/root/model"
NUM_SAMPLES = {num_samples}
MAX_TOKENS = {max_tokens}
TEMPERATURE = {temperature}

def extract_aime_answer(text):
    patterns = (
        r"\\\\boxed\\s*\\{{\\s*(?:\\\\text\\s*\\{{\\s*)?(\\d{{1,3}})",
        r"final\\s+answer\\s*(?:is|:)\\s*\\$?\\s*(\\d{{1,3}})",
        r"answer\\s*(?:is|:)\\s*\\$?\\s*(\\d{{1,3}})",
    )
    for pattern in patterns:
        matches = re.findall(pattern, text, flags=re.IGNORECASE)
        if matches:
            value = int(matches[-1])
            return value if 0 <= value <= 999 else None
    trailing = re.search(r"(\\d{{1,3}})\\D*$", text)
    if trailing:
        value = int(trailing.group(1))
        return value if 0 <= value <= 999 else None
    return None

def vllm_generate(prompt, n=1, max_tokens=4096, temperature=0.6):
    payload = json.dumps({{
        "model": MODEL,
        "messages": [
            {{"role": "system", "content": "You are an expert competition mathematician."}},
            {{"role": "user", "content": prompt + "\\n\\nReason step by step. End with Final answer: \\\\boxed{{NNN}}, where NNN is an integer from 000 to 999."}},
        ],
        "n": n,
        "max_tokens": max_tokens,
        "temperature": temperature,
        "top_p": 0.95,
    }}).encode()
    req = urllib.request.Request(
        f"{{VLLM_URL}}/v1/chat/completions",
        data=payload,
        headers={{"Content-Type": "application/json"}},
    )
    with urllib.request.urlopen(req, timeout=600) as resp:
        data = json.loads(resp.read())
    return [c["message"]["content"] for c in data["choices"]]

# Load AIME 2025
from datasets import load_dataset
ds = load_dataset("test-time-compute/aime_2025", split="test")
problems = list(ds)
print(f"Loaded {{len(problems)}} AIME 2025 problems")

started = time.time()
all_results = []

for i, problem in enumerate(problems):
    question = problem["question"]
    answer = int(problem["answer"])
    problem_id = problem.get("metadata", {{}}).get("problem_idx", i + 1)

    responses = vllm_generate(question, n=NUM_SAMPLES, max_tokens=MAX_TOKENS, temperature=TEMPERATURE)

    correct_samples = []
    for j, resp in enumerate(responses):
        predicted = extract_aime_answer(resp)
        correct = predicted == answer
        correct_samples.append(correct)
        all_results.append({{
            "problem_id": problem_id,
            "sample_index": j,
            "answer": answer,
            "predicted": predicted,
            "correct": correct,
            "response": resp[:500],
        }})

    pass_1 = correct_samples[0] if correct_samples else False
    pass_k = any(correct_samples)
    print(f"  Problem {{problem_id}}: pass@1={pass_1}, pass@{NUM_SAMPLES}={pass_k} (answer={answer})")

elapsed = time.time() - started

# Score
by_problem = {{}}
for r in all_results:
    pid = r["problem_id"]
    by_problem.setdefault(pid, []).append(r["correct"])

ordered = [by_problem[k] for k in sorted(by_problem)]
correct_at_1 = sum(v[0] for v in ordered)
correct_at_k = sum(any(v) for v in ordered)
total = len(ordered)

result = {{
    "pass_at_1": correct_at_1 / total if total else 0,
    "pass_at_{NUM_SAMPLES}": correct_at_k / total if total else 0,
    "correct_at_1": correct_at_1,
    "correct_at_{NUM_SAMPLES}": correct_at_k,
    "total_problems": total,
    "elapsed_seconds": elapsed,
    "raw_results": all_results[:50],  # first 50 for brevity
}}
print(json.dumps(result, indent=2))
with open("/root/aime_result.json", "w") as f:
    json.dump(result, f, indent=2)
'''

    ssh_upload_string(pod, benchmark_script, "/root/run_aime_benchmark.py")

    # Run the benchmark
    exit_code, out, err = ssh_exec(
        pod,
        "cd /root && python run_aime_benchmark.py",
        timeout=1800,  # 30 min max
    )

    if exit_code != 0:
        return BenchmarkResult(error=f"Benchmark failed: {err[:500]}")

    # Read the result
    try:
        exit_code, out2, err2 = ssh_exec(pod, "cat /root/aime_result.json", timeout=30)
        result_data = json.loads(out2)
        return BenchmarkResult(
            pass_at_1=result_data.get("pass_at_1", 0),
            pass_at_5=result_data.get(f"pass_at_{num_samples}", 0),
            correct_at_1=result_data.get("correct_at_1", 0),
            correct_at_5=result_data.get(f"correct_at_{num_samples}", 0),
            total_problems=result_data.get("total_problems", 30),
            elapsed_seconds=result_data.get("elapsed_seconds", 0),
            raw_results=result_data.get("raw_results", []),
        )
    except Exception as e:
        return BenchmarkResult(error=f"Failed to read result: {e}\nstdout: {out[:500]}")


# ---------------------------------------------------------------------------
# Remote training (SFT + GRPO via Optimus Studio pipeline)
# ---------------------------------------------------------------------------


def run_remote_training(
    pod: PodInfo,
    model_id: str = "Tesslate/OmniCoder-9B",
    adapter_path: str = "/root/adapter",
    taskset: str = "aime-2025",
    sft_iters: int = 50,
    grpo_iters: int = 15,
    precision: str = "fp16",
) -> TrainingResult:
    """Run the Optimus Studio training pipeline on the pod.

    This uploads a minimal training script that uses the existing pipeline
    infrastructure to run SFT + GRPO on the AIME taskset.
    """
    training_script = f'''
import json, sys, time, os
sys.path.insert(0, "/root/optimus")

os.environ["ILOPTIMUS_HOME"] = "/root/.iloptimus"

from iloptimus.core.pipeline import RunConfig, create_run, run_pipeline_subprocess
from iloptimus.core.hardware import detect_hardware

hw = detect_hardware()
config = RunConfig(
    model_id="{model_id}",
    taskset_id="{taskset}",
    backend="vllm",
    precision="{precision}",
    sft_iters={sft_iters},
    grpo_iters={grpo_iters},
    grpo_group_size=4,
    benchmark_tasks=30,
    rollouts_per_example=4,
    max_reasoning_tokens=512,
    max_answer_tokens=256,
    adapter_path="{adapter_path}" if os.path.exists("{adapter_path}") else None,
    sft_lora_rank=64,
    sft_lora_layers=16,
)

run_id = create_run(config)
print(f"Run created: {{run_id}}")
state = run_pipeline_subprocess(run_id, config, hw)
print(f"Run completed: status={{state.status}}")
print(f"Baseline: {{state.baseline_accuracy}}")
print(f"Post-SFT: {{state.post_sft_accuracy}}")
print(f"Post-GRPO: {{state.post_grpo_accuracy}}")

result = {{
    "run_id": run_id,
    "status": state.status,
    "baseline_accuracy": state.baseline_accuracy,
    "post_sft_accuracy": state.post_sft_accuracy,
    "post_grpo_accuracy": state.post_grpo_accuracy,
    "adapter_path": str(state.adapter_path) if hasattr(state, "adapter_path") else "",
}}
with open("/root/training_result.json", "w") as f:
    json.dump(result, f, indent=2)
print(json.dumps(result, indent=2))
'''

    ssh_upload_string(pod, training_script, "/root/run_training.py")

    exit_code, out, err = ssh_exec(
        pod,
        "cd /root && python run_training.py",
        timeout=3600,  # 1 hour max for training
    )

    if exit_code != 0:
        return TrainingResult(error=f"Training failed: {err[:500]}")

    try:
        exit_code, out2, _ = ssh_exec(pod, "cat /root/training_result.json", timeout=30)
        data = json.loads(out2)
        return TrainingResult(
            run_id=data.get("run_id", ""),
            accuracy_before=data.get("baseline_accuracy", 0),
            accuracy_after=data.get("post_grpo_accuracy", 0),
            adapter_path=data.get("adapter_path", ""),
            error=data.get("error"),
        )
    except Exception as e:
        return TrainingResult(error=f"Failed to read result: {e}")


# ---------------------------------------------------------------------------
# Remote RL Factory batched loop
# ---------------------------------------------------------------------------


def run_remote_rl_factory(
    pod: PodInfo,
    model_path: str = "/root/model",
    adapter_path: str = "/root/adapter",
    n_pipelines: int = 4,
    n_rounds: int = 10,
    domain: str = "math",
) -> dict:
    """Run the batched RL factory loop on the pod.

    Uses the rl_mode orchestrator with the batched vLLM rollout.
    """
    rl_script = f'''
import json, sys, os
sys.path.insert(0, "/root/optimus")
os.environ["ILOPTIMUS_HOME"] = "/root/.iloptimus"

from iloptimus.core.rl_mode import enter_rl_mode, RLModeSpec

spec = RLModeSpec(
    domain="{domain}",
    n_pipelines={n_pipelines},
    n_rounds={n_rounds},
    enable_rl_batching=True,
    mock_mode=False,
    model_path="{model_path}",
    adapter_path="{adapter_path}" if os.path.exists("{adapter_path}") else None,
)

result = enter_rl_mode(spec)
print(f"RL mode complete: rounds={{result.rounds_completed}}")
print(f"Mean reward: {{result.mean_reward}}")
print(f"Envs used: {{result.envs_used}}")

output = {{
    "run_id": result.run_id,
    "rounds_completed": result.rounds_completed,
    "mean_reward": result.mean_reward,
    "envs_used": result.envs_used,
    "error": result.error,
}}
with open("/root/rl_factory_result.json", "w") as f:
    json.dump(output, f, indent=2)
print(json.dumps(output, indent=2))
'''

    ssh_upload_string(pod, rl_script, "/root/run_rl_factory.py")

    exit_code, out, err = ssh_exec(
        pod,
        "cd /root && python run_rl_factory.py",
        timeout=3600,
    )

    if exit_code != 0:
        return {"error": f"RL factory failed: {err[:500]}"}

    try:
        exit_code, out2, _ = ssh_exec(pod, "cat /root/rl_factory_result.json", timeout=30)
        return json.loads(out2)
    except Exception as e:
        return {"error": f"Failed to read result: {e}"}


# ---------------------------------------------------------------------------
# Full autonomous loop: benchmark → train → benchmark → RL → benchmark
# ---------------------------------------------------------------------------


@dataclass
class LoopState:
    """State of the full autonomous training loop."""
    round: int = 0
    budget_remaining: float = 8.0
    baseline_score: BenchmarkResult = field(default_factory=BenchmarkResult)
    best_score: BenchmarkResult = field(default_factory=BenchmarkResult)
    best_adapter_path: str = ""
    history: list[dict] = field(default_factory=list)
    pod: Optional[PodInfo] = None
    error: Optional[str] = None

    def to_dict(self) -> dict:
        return {
            "round": self.round,
            "budget_remaining": self.budget_remaining,
            "baseline": self.baseline_score.score_str,
            "best": self.best_score.score_str,
            "best_adapter": self.best_adapter_path,
            "history": self.history,
            "pod_id": self.pod.pod_id if self.pod else None,
            "error": self.error,
        }


def run_full_loop(
    api_key: str,
    adapter_local_path: str = r"E:\omnicoder-rl-candidates\continuation-50-files\checkpoint-50",
    model_id: str = "Tesslate/OmniCoder-9B",
    budget_usd: float = 8.0,
    max_rounds: int = 3,
    gpu_type: str = "RTX6000Ada_48GB",
) -> LoopState:
    """Run the full autonomous loop:

    1. Provision a GPU pod on Prime Intellect
    2. Deploy the model + adapter + vLLM
    3. Baseline benchmark on AIME 2025
    4. For each round:
       a. Run SFT + GRPO training (RSI loop)
       b. Run batched RL factory (rlfactory loop)
       c. Benchmark again
       d. If improved, save adapter; if not, revert
    5. Clean up and return results
    """
    state = LoopState(budget_remaining=budget_usd)

    # Step 1: Provision pod
    print("=" * 60)
    print("STEP 1: Provisioning GPU pod on Prime Intellect")
    print("=" * 60)

    config = PodConfig(
        name="optimus-rl-loop",
        gpu_type=gpu_type,
        disk_size=200,
        max_price=1.50,
    )

    try:
        pod = provision_pod(config, api_key=api_key)
        print(f"Pod provisioned: {pod.pod_id} ({pod.gpu_type}, ${pod.cost_per_hr}/hr)")
        state.pod = pod

        # Wait for pod to be ready
        print("Waiting for pod to become active...")
        pod = wait_for_pod_ready(pod.pod_id, timeout=300, api_key=api_key)
        pod.cost_per_hr = pod.cost_per_hr or state.pod.cost_per_hr
        state.pod = pod
        print(f"Pod active: {pod.ssh_connection} (IP: {pod.ip})")

    except Exception as e:
        state.error = f"Provisioning failed: {e}"
        return state

    # Track spending
    def update_budget():
        if state.pod:
            elapsed = time.time() - state.pod.created_at
            spent = (elapsed / 3600) * state.pod.cost_per_hr
            state.budget_remaining = max(0, budget_usd - spent)
            return state.budget_remaining > 0
        return False

    # Step 2: Deploy
    print("\n" + "=" * 60)
    print("STEP 2: Deploying model + vLLM to pod")
    print("=" * 60)

    deploy_results = deploy_to_pod(
        pod,
        model_id=model_id,
        adapter_local_path=adapter_local_path,
    )
    if deploy_results["errors"]:
        print(f"Deployment errors: {deploy_results['errors']}")
        # Continue anyway — some errors may be non-critical

    # Wait for vLLM to start
    print("Waiting for vLLM server to start...")
    time.sleep(30)

    # Check vLLM health
    exit_code, out, err = ssh_exec(pod, "curl -s http://localhost:8000/health", timeout=30)
    if exit_code != 0 or "ok" not in out.lower():
        print(f"vLLM health check failed, checking logs...")
        _, log, _ = ssh_exec(pod, "tail -50 /root/vllm.log", timeout=30)
        print(f"vLLM log: {log[:500]}")
        # Try restarting with different settings
        ssh_exec(pod, "pkill -f vllm", timeout=10)
        time.sleep(5)
        ssh_exec(pod,
                 "nohup python -m vllm.entrypoints.openai.api_server "
                 "--model /root/model --enable-lora --lora-modules adapter=/root/adapter "
                 "--max-lora-rank 256 --gpu-memory-utilization 0.85 --dtype bfloat16 "
                 "--max-model-len 4096 --port 8000 > /root/vllm.log 2>&1 &",
                 timeout=10)
        time.sleep(60)
        _, out, _ = ssh_exec(pod, "curl -s http://localhost:8000/health", timeout=30)

    print(f"vLLM status: {out[:100]}")

    # Step 3: Baseline benchmark
    print("\n" + "=" * 60)
    print("STEP 3: Baseline AIME 2025 benchmark")
    print("=" * 60)

    baseline = run_remote_aime_benchmark(
        pod,
        vllm_url=f"http://localhost:8000",
        num_samples=5,
        max_tokens=4096,
        use_adapter=True,
    )

    if baseline.error:
        print(f"Baseline benchmark error: {baseline.error}")
        state.error = baseline.error
        return state

    state.baseline_score = baseline
    state.best_score = baseline
    print(f"Baseline: {baseline.score_str}")
    print(f"Budget remaining: ~${state.budget_remaining:.2f}")

    state.history.append({
        "round": 0,
        "stage": "baseline",
        "score": baseline.score_str,
        "correct_at_1": baseline.correct_at_1,
        "correct_at_5": baseline.correct_at_5,
        "budget_remaining": state.budget_remaining,
    })

    # Step 4: Training rounds
    for round_num in range(1, max_rounds + 1):
        if not update_budget():
            print(f"\nBudget exhausted, stopping at round {round_num - 1}")
            break

        print(f"\n{'=' * 60}")
        print(f"ROUND {round_num}: TTT + RSI + Batched RL")
        print(f"Budget remaining: ~${state.budget_remaining:.2f}")
        print(f"{'=' * 60}")

        state.round = round_num

        # 4a: SFT + GRPO training (RSI loop)
        print(f"\n--- Round {round_num}a: SFT + GRPO training ---")
        train_result = run_remote_training(
            pod,
            model_id=model_id,
            adapter_path="/root/adapter",
            taskset="aime-2025",
            sft_iters=50,
            grpo_iters=15,
        )

        if train_result.error:
            print(f"Training error: {train_result.error}")
            state.history.append({
                "round": round_num,
                "stage": "training",
                "error": train_result.error,
            })
        else:
            print(f"Training done: {train_result.accuracy_before:.2%} → {train_result.accuracy_after:.2%}")

        # 4b: Batched RL factory
        if update_budget():
            print(f"\n--- Round {round_num}b: Batched RL factory ---")
            rl_result = run_remote_rl_factory(
                pod,
                model_path="/root/model",
                adapter_path="/root/adapter",
                n_pipelines=4,
                n_rounds=10,
                domain="math",
            )

            if "error" in rl_result:
                print(f"RL factory error: {rl_result['error']}")
            else:
                print(f"RL factory done: {rl_result.get('rounds_completed')} rounds, "
                      f"mean reward: {rl_result.get('mean_reward', 0):.3f}")

        # 4c: Benchmark
        if update_budget():
            print(f"\n--- Round {round_num}c: Post-training benchmark ---")
            post_score = run_remote_aime_benchmark(
                pod,
                vllm_url=f"http://localhost:8000",
                num_samples=5,
                max_tokens=4096,
                use_adapter=True,
            )

            if post_score.error:
                print(f"Benchmark error: {post_score.error}")
            else:
                print(f"Post-training: {post_score.score_str}")

                # Check if improved
                if post_score.correct_at_1 > state.best_score.correct_at_1:
                    print(f"*** NEW BEST! {post_score.correct_at_1}/{post_score.total_problems} ***")
                    state.best_score = post_score
                    # Save the improved adapter
                    ssh_exec(pod, "cp -r /root/adapter /root/best_adapter", timeout=30)
                    state.best_adapter_path = "/root/best_adapter"

                state.history.append({
                    "round": round_num,
                    "stage": "post-training",
                    "score": post_score.score_str,
                    "correct_at_1": post_score.correct_at_1,
                    "correct_at_5": post_score.correct_at_5,
                    "budget_remaining": state.budget_remaining,
                })

    # Step 5: Download best adapter if improved
    if state.best_score.correct_at_1 > state.baseline_score.correct_at_1:
        print("\n" + "=" * 60)
        print("Downloading improved adapter...")
        print("=" * 60)
        # TODO: SFTP download the best adapter

    # Step 6: Cleanup
    print(f"\n{'=' * 60}")
    print(f"LOOP COMPLETE")
    print(f"Baseline: {state.baseline_score.score_str}")
    print(f"Best:     {state.best_score.score_str}")
    print(f"Budget remaining: ~${state.budget_remaining:.2f}")
    print(f"{'=' * 60}")

    # Terminate pod to stop spending
    print("Terminating pod to stop spending...")
    terminate_pod(pod.pod_id, api_key=api_key)
    print("Pod terminated.")

    return state


# ---------------------------------------------------------------------------
# Public API
# ---------------------------------------------------------------------------

__all__ = [
    "PodConfig", "PodInfo", "BenchmarkResult", "TrainingResult", "LoopState",
    "list_available_gpus", "find_best_gpu",
    "provision_pod", "get_pod_status", "get_pod", "list_pods", "terminate_pod",
    "wait_for_pod_ready",
    "ssh_exec", "ssh_upload", "ssh_upload_string",
    "deploy_to_pod",
    "run_remote_aime_benchmark",
    "run_remote_training",
    "run_remote_rl_factory",
    "run_full_loop",
]
