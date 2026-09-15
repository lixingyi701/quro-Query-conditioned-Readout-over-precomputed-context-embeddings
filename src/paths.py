"""Single source of truth for on-disk locations.

``/home`` on this machine is full, so every large artefact (model weights, latent
caches, run outputs) lives under ``QURO_ROOT`` while the code stays in the repo.
Everything is overridable by environment variable so that a different machine
only needs to export a handful of paths.
"""

from __future__ import annotations

import os

# Large artefacts: weights, latent caches, prepared data, run outputs.
QURO_ROOT = os.environ.get("QURO_ROOT", "/data02/quro")

MODELS_DIR = os.environ.get("QURO_MODELS_DIR", os.path.join(QURO_ROOT, "models"))
DATA_DIR = os.environ.get("QURO_DATA_DIR", os.path.join(QURO_ROOT, "data"))
CACHE_ROOT = os.environ.get("QURO_CACHE_DIR", os.path.join(QURO_ROOT, "cache"))
RUNS_DIR = os.environ.get("QURO_RUNS_DIR", os.path.join(QURO_ROOT, "runs"))

# Frozen backbones that are already present on this machine.
MISTRAL_PATH = os.environ.get(
    "MISTRAL_PATH", "/home/lxy/selecom/baselineModel/Mistral-7B-Instruct-v0.2")
ENCODER_PATH = os.environ.get(
    "ENCODER_PATH", "/home/lxy/selecom/baselineModel/Qwen3-Embedding-0.6B")

# Frozen offline compressors (downloaded by scripts/download_compressors.py).
PISCO_MISTRAL = os.environ.get(
    "PISCO_MISTRAL", os.path.join(MODELS_DIR, "pisco-mistral"))
COCOM_MISTRAL = {
    rate: os.path.join(MODELS_DIR, f"cocom-v1-{rate}-mistral-7b") for rate in (4, 16, 128)
}

# SeleCom release used for training/eval data.
SELECOM_ROOT = os.environ.get("SELECOM_ROOT", "/home/lxy/selecom")
SELECOM_STAGE1 = os.path.join(SELECOM_ROOT, "data/stage1/stage1_train_data.jsonl")
SELECOM_STAGE2 = os.path.join(SELECOM_ROOT, "data/stage2/stage2_train_data.jsonl")
SELECOM_TRIVIA_EVAL = os.path.join(SELECOM_ROOT, "data/trivia_qa/trivia_qa_eval.jsonl")


def require(path: str, what: str) -> str:
    """Fail loudly and early rather than deep inside a model loader."""
    if not os.path.exists(path):
        raise FileNotFoundError(f"{what} not found at {path}; set the matching environment variable")
    return path
