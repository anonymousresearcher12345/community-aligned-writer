"""Community-aligned note scoring with a rater-conditioned rating model.

Loads model weights and rater tables lazily, samples a shared jury per post,
calibrates predicted rating probabilities, and fits note helpfulness and factor.
Call predict_helpfulness() for the note helpfulness score and factor.
"""
from __future__ import annotations

import hashlib
import os
from collections import OrderedDict
from pathlib import Path
from typing import Any, Dict, List, Tuple

import numpy as np
import pandas as pd
import torch
import torch.nn as nn
import torch.nn.functional as F
from safetensors.torch import load_file
from transformers import AutoModel, AutoTokenizer

# =============================================================================
# Paths / constants
# =============================================================================
# External artifacts are supplied by the user; no local paths are assumed.
BERT_CKPT = os.environ.get("BERT_CKPT", "")
D6_RATER_TSV = os.environ.get("D6_RATER_TSV", "")
D2_HELPFULNESS_TSV = os.environ.get("D2_HELPFULNESS_TSV", "")


def _require_path(name: str, value: str) -> str:
    if not value.strip():
        raise ValueError(f"Set {name} to the external artifact path before using this component.")
    return value


BERT_MODEL_NAME = "answerdotai/ModernBERT-base"
MAX_LEN = 1024
BERT_BATCH = 32
N_RATERS = 1000
SEED_BASE = 42

RM_TEMPERATURE = 1.44
GLOBAL_INTERCEPT_D2 = 0.177

RATER_FEATURE_COLS: List[str] = [
    "internalRaterIntercept",
    "internalRaterFactor1",
    "internalRaterFactor2",
    "internalRaterFactor3",
    "internalRaterFactor4",
    "internalRaterFactor5",
]
RATER_DIM = len(RATER_FEATURE_COLS)
RATING_CLASSES = np.array([0.0, 0.5, 1.0], dtype=np.float32)


# =============================================================================
# Rating-model architecture used for inference
# =============================================================================
class RaterProj(nn.Module):
    def __init__(self, in_dim: int, hidden_size: int) -> None:
        super().__init__()
        self.fc1 = nn.Linear(in_dim, hidden_size)
        self.act = nn.GELU()
        self.fc2 = nn.Linear(hidden_size, hidden_size)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.fc2(self.act(self.fc1(x)))


class RaterPrefixModernBERT(nn.Module):
    """ModernBERT backbone, continuous rater prefix, and rating head.

    Only the rating head is needed for inference.
    """

    def __init__(
        self,
        model_name: str,
        rater_dim: int = RATER_DIM,
    ) -> None:
        super().__init__()
        self.backbone = AutoModel.from_pretrained(model_name, reference_compile=False)
        hidden_size = self.backbone.config.hidden_size
        self.rater_proj = RaterProj(rater_dim, hidden_size)
        self.rating_head = nn.Linear(hidden_size, 3)

    def _embed_with_prefix(
        self,
        input_ids: torch.Tensor,
        rater_features: torch.Tensor,
        attention_mask: torch.Tensor,
    ) -> Tuple[torch.Tensor, torch.Tensor]:
        try:
            tok_emb_layer = self.backbone.embeddings.tok_embeddings
        except AttributeError:
            tok_emb_layer = self.backbone.get_input_embeddings()
        tok_emb = tok_emb_layer(input_ids)
        rater_emb = self.rater_proj(rater_features).unsqueeze(1)
        inputs_embeds = torch.cat(
            [tok_emb[:, :1], rater_emb, tok_emb[:, 1:]], dim=1
        )
        attn = F.pad(attention_mask, (1, 0), value=1)
        return inputs_embeds, attn

    def forward(
        self,
        input_ids: torch.Tensor,
        attention_mask: torch.Tensor,
        rater_features: torch.Tensor,
        **kwargs: Any,
    ) -> Dict[str, torch.Tensor]:
        inputs_embeds, attn = self._embed_with_prefix(
            input_ids, rater_features, attention_mask
        )
        out = self.backbone(inputs_embeds=inputs_embeds, attention_mask=attn)
        cls = out.last_hidden_state[:, 0]
        return {
            "logits_rating": self.rating_head(cls),
        }


# =============================================================================
# Rating aggregation
# =============================================================================
def aggregate_d2_full(
    ratings: np.ndarray,
    user_intercept: np.ndarray,
    user_factor: np.ndarray,
    global_intercept: float = GLOBAL_INTERCEPT_D2,
) -> Tuple[float, float]:
    """Fit note factor and intercept by least squares.

    Solves the 1-factor matrix factorization for a single note across the
    sampled raters. The design matrix columns are [user_factor, 1], so the
    fitted coefficients are [note_factor, note_intercept]:

      - note_intercept (m3[1, 0]) is the predicted helpfulness.
      - note_factor    (m3[0, 0]) is the predicted note factor (noteFactor1).
    """
    m1 = ratings - user_intercept - global_intercept
    m2 = np.hstack((user_factor, np.ones((len(user_factor), 1))))
    m3 = np.linalg.lstsq(m2, m1, rcond=None)[0]
    return float(m3[0, 0]), float(m3[1, 0])


def calibrated_probs(entry: Dict[str, Any], temperature: float = RM_TEMPERATURE) -> np.ndarray:
    """Temperature-scale the rating probabilities of a ``score_note_with_bert`` entry.

    Binary-margin scaling: the SOMEWHAT_HELPFUL mass is held fixed and only the
    NOT_HELPFUL / HELPFUL split is re-tempered. At ``temperature == 1`` this
    reproduces the original softmax exactly (because
    ``p2 / (p0 + p2) == sigmoid(l2 - l0)``).

    Parameters
    ----------
    entry : dict
        Output of :func:`score_note_with_bert` (needs ``logits`` and ``probs``).
    temperature : float, optional
        Calibration temperature ``T``. Defaults to :data:`RM_TEMPERATURE`.

    Returns
    -------
    numpy.ndarray
        ``(N, 3)`` tempered probabilities over (NOT, SOMEWHAT, HELPFUL).
    """
    logits = entry["logits"]
    probs = entry["probs"]
    z = (logits[:, 2] - logits[:, 0]).astype(np.float64)
    p_help_binary = (1.0 / (1.0 + np.exp(-z / float(temperature)))).astype(np.float32)
    binary_mass = probs[:, 0] + probs[:, 2]

    out = np.empty_like(probs)
    out[:, 1] = probs[:, 1]
    out[:, 2] = binary_mass * p_help_binary
    out[:, 0] = binary_mass * (1.0 - p_help_binary)
    return out


# =============================================================================
# Lazy singletons
# =============================================================================
_BERT: Dict[str, Any] = {}  # tokenizer, model, device
_RATER_POOL: Dict[str, np.ndarray] = {}  # d6_params, intercept_d2, factor1_d2


def _device() -> torch.device:
    return torch.device("cuda" if torch.cuda.is_available() else "cpu")


def get_bert() -> Tuple[Any, RaterPrefixModernBERT, torch.device]:
    if "model" in _BERT:
        return _BERT["tokenizer"], _BERT["model"], _BERT["device"]
    _require_path("BERT_CKPT", BERT_CKPT)
    device = _device()
    print(f"[helpfulness_score] loading BERT on {device} from {BERT_CKPT}", flush=True)
    tokenizer = AutoTokenizer.from_pretrained(BERT_MODEL_NAME)
    model = RaterPrefixModernBERT(model_name=BERT_MODEL_NAME)
    st = Path(BERT_CKPT) / "model.safetensors"
    bn = Path(BERT_CKPT) / "pytorch_model.bin"
    if st.exists():
        state = load_file(str(st))
    elif bn.exists():
        state = torch.load(str(bn), map_location="cpu")
    else:
        raise FileNotFoundError(f"no weights in {BERT_CKPT}")
    # The Bradley-Terry head is used only in training. Require every scoring weight.
    state = {k: v for k, v in state.items() if not k.startswith("bt_head.")}
    model.load_state_dict(state, strict=True)
    model.to(device).eval()
    _BERT.update({"tokenizer": tokenizer, "model": model, "device": device})
    return tokenizer, model, device


def get_rater_pool() -> Tuple[np.ndarray, np.ndarray, np.ndarray]:
    """Load D6 ∩ D2 rater pool. Returns (d6_params, intercept_d2, factor1_d2).

    `factor1_d2` supplies the core rater factor used in note-score aggregation.
    """
    if "d6_params" in _RATER_POOL:
        return (
            _RATER_POOL["d6_params"],
            _RATER_POOL["intercept_d2"],
            _RATER_POOL["factor1_d2"],
        )
    _require_path("D6_RATER_TSV", D6_RATER_TSV)
    _require_path("D2_HELPFULNESS_TSV", D2_HELPFULNESS_TSV)
    print(f"[helpfulness_score] loading D6 rater params: {D6_RATER_TSV}", flush=True)
    d6_df = pd.read_csv(
        D6_RATER_TSV, sep="\t", dtype={"raterParticipantId": str}, low_memory=False
    )
    d6_df = (
        d6_df[["raterParticipantId"] + RATER_FEATURE_COLS]
        .dropna()
        .drop_duplicates(subset=["raterParticipantId"])
    )
    print(f"[helpfulness_score] loading D2 helpfulness: {D2_HELPFULNESS_TSV}", flush=True)
    d2_df = pd.read_csv(
        D2_HELPFULNESS_TSV, sep="\t", dtype={"raterParticipantId": str}, low_memory=False
    )
    d2_df = (
        d2_df[["raterParticipantId", "coreRaterIntercept", "coreRaterFactor1"]]
        .dropna()
        .drop_duplicates(subset=["raterParticipantId"])
    )
    merged = d6_df.merge(d2_df, on="raterParticipantId", how="inner")
    if len(merged) == 0:
        raise RuntimeError("rater pool is empty after D6 ∩ D2 merge")
    print(f"[helpfulness_score] rater pool size: {len(merged):,}", flush=True)
    d6_params = merged[RATER_FEATURE_COLS].to_numpy(dtype=np.float32)
    intercept_d2 = merged["coreRaterIntercept"].to_numpy(dtype=np.float32).reshape(-1, 1)
    factor1_d2 = merged["coreRaterFactor1"].to_numpy(dtype=np.float32).reshape(-1, 1)
    _RATER_POOL.update(
        {"d6_params": d6_params, "intercept_d2": intercept_d2, "factor1_d2": factor1_d2}
    )
    return d6_params, intercept_d2, factor1_d2


# =============================================================================
# Per-(post, note) cache and BERT scoring
# =============================================================================
_PAIR_CACHE: "OrderedDict[str, Dict[str, np.ndarray]]" = OrderedDict()
_PAIR_CACHE_MAX = 4


def _pair_key(post_text: str, note_text: str) -> str:
    h = hashlib.sha1()
    h.update(post_text.encode("utf-8"))
    h.update(b"\x00")
    h.update(note_text.encode("utf-8"))
    return h.hexdigest()


def _post_seed(post_text: str) -> int:
    h = hashlib.sha1(post_text.encode("utf-8")).digest()
    return (SEED_BASE + int.from_bytes(h[:4], "little")) & 0x7FFFFFFF


@torch.inference_mode()
def score_note_with_bert(post_text: str, note_text: str) -> Dict[str, np.ndarray]:
    """Run BERT across N_RATERS sampled raters for one (post, note) pair.

    Returns a cached dict with:
      - intercept_D2    (N, 1)   coreRaterIntercept of the sampled raters
      - factor1_D2      (N, 1)   coreRaterFactor1 (used for aggregation)
      - logits          (N, 3)   raw rating-head logits (for temperature scaling)
      - probs      (N, 3)   softmax over (NOT, SOMEWHAT, HELPFUL)

    ``probs`` are uncalibrated (T=1); apply
    :func:`calibrated_probs` for the calibrated scale.
    """
    key = _pair_key(post_text, note_text)
    if key in _PAIR_CACHE:
        _PAIR_CACHE.move_to_end(key)
        return _PAIR_CACHE[key]

    tokenizer, model, device = get_bert()
    d6_pool, ic_pool, f1_pool = get_rater_pool()

    # Per-post seed so candidates share the same sample of up to N_RATERS raters.
    seed = _post_seed(post_text)
    rng = np.random.default_rng(seed)
    pool_size = d6_pool.shape[0]
    n = min(N_RATERS, pool_size)
    idx = rng.choice(pool_size, size=n, replace=False)
    d6_s = d6_pool[idx]
    ic_s = ic_pool[idx]
    f1_s = f1_pool[idx]

    enc = tokenizer(
        post_text,
        note_text,
        truncation="longest_first",
        max_length=MAX_LEN - 1,
        return_tensors="pt",
    )
    input_ids = enc["input_ids"].to(device)
    attn = enc["attention_mask"].to(device)
    model_dtype = next(model.parameters()).dtype
    rater_feats_t = torch.from_numpy(d6_s).to(device=device, dtype=model_dtype)

    logits_chunks: List[torch.Tensor] = []
    for start in range(0, n, BERT_BATCH):
        end = min(start + BERT_BATCH, n)
        bsz = end - start
        ids = input_ids.expand(bsz, -1)
        am = attn.expand(bsz, -1)
        rf = rater_feats_t[start:end]
        out = model(input_ids=ids, attention_mask=am, rater_features=rf)
        logits_chunks.append(out["logits_rating"].float())

    logits_all = torch.cat(logits_chunks, dim=0)
    probs_t = F.softmax(logits_all, dim=-1)
    scores = torch.cat([logits_all, probs_t], dim=1)
    scores_np = scores.cpu().numpy().astype(np.float32)
    logits_np = scores_np[:, :3]
    probs_all = scores_np[:, 3:6]

    entry: Dict[str, Any] = {
        "intercept_D2": ic_s,
        "factor1_D2": f1_s,
        "logits": logits_np,
        "probs": probs_all,
    }
    _PAIR_CACHE[key] = entry
    if len(_PAIR_CACHE) > _PAIR_CACHE_MAX:
        _PAIR_CACHE.popitem(last=False)
    return entry


# Public scoring interface.


def predict_helpfulness(
    note_text: str, post_text: str, temperature: float = RM_TEMPERATURE
) -> dict:
    """Return the aggregated note helpfulness score and factor.

    Parameters
    ----------
    note_text : str
        Candidate Community Note text.
    post_text : str
        X post text the note is responding to.
    temperature : float, optional
        Calibration temperature applied to the rating head via
        ``calibrated_probs``. Defaults to
        ``RM_TEMPERATURE``; pass ``1.0`` for the raw model tier.

    Returns
    -------
    dict
        ``{"score": float, "factor": float}``. Both values come from the
        same least-squares fit to calibrated expected ratings.
    """
    entry = score_note_with_bert(post_text=post_text, note_text=note_text)
    probs = calibrated_probs(entry, temperature)
    expected = (probs * RATING_CLASSES[None, :]).sum(axis=1)
    factor, score = aggregate_d2_full(
        ratings=expected.astype(np.float64).reshape(-1, 1),
        user_intercept=entry["intercept_D2"].astype(np.float64),
        user_factor=entry["factor1_D2"].astype(np.float64),
    )

    return {"score": float(score), "factor": float(factor)}
