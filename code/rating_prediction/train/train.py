"""
Rater-conditioned ModernBERT reward model for Community Notes.

The model predicts how a given rater rates a note written for a post
(NOT_HELPFUL / SOMEWHAT_HELPFUL / HELPFUL), conditioned on that rater's
matrix-factorization parameters.

Architecture
------------
Input sequence (after tokenization + prefix injection):
    [CLS] [RATER]  post_tokens  [SEP] note_tokens [SEP]

[RATER] is a continuous token whose embedding is produced by a small MLP that
projects the 6-dim rater parameter vector (intercept + 5 factors) to
``hidden_size``. Two heads sit on the [CLS] hidden state:

    rating_head : Linear(hidden, 3)  -> cross-entropy over the 3 rating levels
    bt_head     : Linear(hidden, 1)  -> scalar Bradley-Terry preference score

Objective
---------
    loss = CE(rating) + w_bt * BT(pairs)

BT is ``-log sigmoid(s_chosen - s_rejected)`` over within-rater within-post
preference pairs (same rater, same post, two notes rated differently). One BT
batch is added every ``bt_every`` optimizer steps, with ``bt_every`` chosen so
the BT head makes ~``bt_target_passes`` passes over the pair set per epoch.

Model selection
---------------
Validation runs every ``eval_steps`` on AI-written notes only, and the
checkpoint with the lowest ``ai_binary_logloss`` (HELPFUL-vs-NOT_HELPFUL log
loss on AI notes) is restored at the end and evaluated on test.
"""

from __future__ import annotations

import argparse
import json
import random
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Dict, List, Optional, Tuple

import numpy as np
import pandas as pd
import torch
import torch.nn as nn
import torch.nn.functional as F
from sklearn.metrics import accuracy_score, f1_score, roc_auc_score
from torch.utils.data import Dataset, SequentialSampler
from transformers import (
    AutoModel,
    AutoTokenizer,
    PreTrainedTokenizerBase,
    Trainer,
    TrainingArguments,
    set_seed,
)
from transformers.trainer_pt_utils import LengthGroupedSampler, get_parameter_names


# -----------------------------------------------------------------------------
# Constants
# -----------------------------------------------------------------------------

RATING_TO_INT: Dict[str, int] = {
    "NOT_HELPFUL": 0,
    "SOMEWHAT_HELPFUL": 1,
    "HELPFUL": 2,
}

RATER_FEATURE_COLS: List[str] = [
    "internalRaterIntercept",
    "internalRaterFactor1",
    "internalRaterFactor2",
    "internalRaterFactor3",
    "internalRaterFactor4",
    "internalRaterFactor5",
]
RATER_DIM = len(RATER_FEATURE_COLS)


# -----------------------------------------------------------------------------
# Rating dataset
# -----------------------------------------------------------------------------


class RatingDataset(Dataset):
    """One item per rating row, joined to note/post text and rater parameters.

    The dataset returns un-tokenized text; the collator tokenizes so batches
    can be dynamically padded.

    Parameters
    ----------
    rows_df : pd.DataFrame
        Rating rows with ``noteId``, ``raterParticipantId``, ``rating``, ``writer``.
    notes_df : pd.DataFrame
        ``noteId`` -> ``note_text``, ``post_text``.
    rater_df : pd.DataFrame
        ``raterParticipantId`` -> ``RATER_FEATURE_COLS``.
    """

    def __init__(
        self,
        rows_df: pd.DataFrame,
        notes_df: pd.DataFrame,
        rater_df: pd.DataFrame,
    ) -> None:
        df = rows_df.copy()
        n_before = len(df)
        df = df.merge(
            notes_df[["noteId", "note_text", "post_text"]],
            on="noteId",
            how="inner",
            validate="many_to_one",
        )
        df = df.merge(
            rater_df[["raterParticipantId"] + RATER_FEATURE_COLS],
            on="raterParticipantId",
            how="inner",
            validate="many_to_one",
        )
        if len(df) != n_before:
            print(
                f"[data] lookup joins dropped {n_before - len(df):,} of "
                f"{n_before:,} rows (missing note/rater)"
            )

        df = df[df["rating"].isin(RATING_TO_INT.keys())]
        df = df.dropna(subset=["post_text", "note_text"] + RATER_FEATURE_COLS)
        df = df.reset_index(drop=True)

        self.post_text: List[str] = df["post_text"].astype(str).tolist()
        self.note_text: List[str] = df["note_text"].astype(str).tolist()
        # Char-count proxy for token length, used by LengthGroupedSampler.
        self.lengths: List[int] = [
            len(p) + len(n) for p, n in zip(self.post_text, self.note_text)
        ]
        # writer_flag: 1 = AI-written note, 0 = human-written.
        self.writer_flag: np.ndarray = (
            (df["writer"].astype(str) == "ai").to_numpy(dtype=np.int64)
        )
        self.rater_features: np.ndarray = df[RATER_FEATURE_COLS].to_numpy(
            dtype=np.float32
        )
        self.rating_label: np.ndarray = (
            df["rating"].map(RATING_TO_INT).to_numpy(dtype=np.int64)
        )

    def __len__(self) -> int:
        return len(self.post_text)

    def __getitem__(self, idx: int) -> Dict[str, Any]:
        return {
            "post_text": self.post_text[idx],
            "note_text": self.note_text[idx],
            "rater_features": self.rater_features[idx],
            "rating_label": int(self.rating_label[idx]),
            "writer_flag": int(self.writer_flag[idx]),
        }


@dataclass
class TokenizingCollator:
    tokenizer: PreTrainedTokenizerBase
    max_len: int  # total budget, including the inserted rater token

    def __call__(self, batch: List[Dict[str, Any]]) -> Dict[str, torch.Tensor]:
        posts = [b["post_text"] for b in batch]
        notes = [b["note_text"] for b in batch]
        enc = self.tokenizer(
            posts,
            notes,
            truncation="longest_first",
            max_length=self.max_len - 1,  # reserve 1 slot for the rater token
            padding=True,
            return_tensors="pt",
        )
        rater_features = torch.from_numpy(np.stack([b["rater_features"] for b in batch])).float()
        rating_label = torch.tensor([b["rating_label"] for b in batch], dtype=torch.long)
        writer_flag = torch.tensor([b["writer_flag"] for b in batch], dtype=torch.long)
        return {
            "input_ids": enc["input_ids"],
            "attention_mask": enc["attention_mask"],
            "rater_features": rater_features,
            "rating_label": rating_label,
            "writer_flag": writer_flag,
        }


# -----------------------------------------------------------------------------
# Model
# -----------------------------------------------------------------------------


class RaterProj(nn.Module):
    def __init__(self, in_dim: int, hidden_size: int) -> None:
        super().__init__()
        self.fc1 = nn.Linear(in_dim, hidden_size)
        self.act = nn.GELU()
        self.fc2 = nn.Linear(hidden_size, hidden_size)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.fc2(self.act(self.fc1(x)))


class RaterPrefixModernBERT(nn.Module):
    """ModernBERT backbone + continuous rater prefix token + rating and BT heads.

    Parameters
    ----------
    model_name : str
        Hugging Face model id of the backbone.
    rater_dim : int
        Dimension of the rater parameter vector.
    """

    def __init__(self, model_name: str, rater_dim: int = RATER_DIM) -> None:
        super().__init__()
        self.backbone = AutoModel.from_pretrained(
            model_name,
            reference_compile=False,
            attn_implementation="sdpa",
        )
        hidden_size = self.backbone.config.hidden_size
        self.hidden_size = hidden_size
        self.rater_proj = RaterProj(rater_dim, hidden_size)
        self.rating_head = nn.Linear(hidden_size, 3)
        self.bt_head = nn.Linear(hidden_size, 1)

        nn.init.normal_(self.rating_head.weight, std=0.02)
        nn.init.zeros_(self.rating_head.bias)
        nn.init.normal_(self.bt_head.weight, std=0.02)
        nn.init.zeros_(self.bt_head.bias)

    def _embed_with_prefix(
        self,
        input_ids: torch.Tensor,
        rater_features: torch.Tensor,
        attention_mask: torch.Tensor,
    ) -> Tuple[torch.Tensor, torch.Tensor]:
        tok_emb = self.backbone.embeddings.tok_embeddings(input_ids)  # [B, T, H]
        rater_emb = self.rater_proj(rater_features).unsqueeze(1)  # [B, 1, H]
        # Insert the rater token right after [CLS].
        inputs_embeds = torch.cat(
            [tok_emb[:, :1], rater_emb, tok_emb[:, 1:]], dim=1
        )  # [B, T+1, H]
        attn = F.pad(attention_mask, (1, 0), value=1)
        return inputs_embeds, attn

    def _cls(
        self,
        input_ids: torch.Tensor,
        attention_mask: torch.Tensor,
        rater_features: torch.Tensor,
    ) -> torch.Tensor:
        inputs_embeds, attn = self._embed_with_prefix(
            input_ids, rater_features, attention_mask
        )
        out = self.backbone(inputs_embeds=inputs_embeds, attention_mask=attn)
        return out.last_hidden_state[:, 0]  # [B, H]

    def score_bt(
        self,
        input_ids: torch.Tensor,
        attention_mask: torch.Tensor,
        rater_features: torch.Tensor,
    ) -> torch.Tensor:
        """Scalar Bradley-Terry preference score per (rater, post, note) -> [B]."""
        cls = self._cls(input_ids, attention_mask, rater_features)
        return self.bt_head(cls).squeeze(-1)

    def forward(
        self,
        input_ids: torch.Tensor,
        attention_mask: torch.Tensor,
        rater_features: torch.Tensor,
        rating_label: Optional[torch.Tensor] = None,
        **kwargs: Any,
    ) -> Dict[str, torch.Tensor]:
        cls = self._cls(input_ids, attention_mask, rater_features)
        logits_rating = self.rating_head(cls)  # [B, 3]
        result: Dict[str, torch.Tensor] = {"logits_rating": logits_rating}
        if rating_label is not None:
            result["loss"] = F.cross_entropy(logits_rating, rating_label)
        return result


# -----------------------------------------------------------------------------
# Pairwise data (BT training + pairwise evaluation)
# -----------------------------------------------------------------------------


def _filter_resolvable_same_post_pairs(
    pairs_df: pd.DataFrame,
    notes_df: pd.DataFrame,
    rater_df: pd.DataFrame,
    tag: str,
) -> pd.DataFrame:
    """Keep pairs whose rater and both notes resolve in the lookup tables.

    Also asserts that the chosen and rejected notes share the same post text.
    """
    rater_ids = set(rater_df["raterParticipantId"])
    note_ids = set(notes_df["noteId"])
    ok = (
        pairs_df["raterParticipantId"].isin(rater_ids)
        & pairs_df["chosen_noteId"].isin(note_ids)
        & pairs_df["rejected_noteId"].isin(note_ids)
    )
    n_dropped = int((~ok).sum())
    if n_dropped:
        print(
            f"[{tag}] dropped {n_dropped} of {len(pairs_df)} pairs "
            f"with missing rater/note lookup"
        )
    pairs_df = pairs_df[ok].reset_index(drop=True)

    assert notes_df["noteId"].is_unique, "notes_df has duplicate noteId rows"
    note_post = notes_df.set_index("noteId")["post_text"]
    same_post = (
        note_post.loc[pairs_df["chosen_noteId"]].to_numpy()
        == note_post.loc[pairs_df["rejected_noteId"]].to_numpy()
    )
    assert same_post.all(), (
        f"[{tag}] {int((~same_post).sum())} pairs have different post_text for "
        f"chosen vs rejected notes"
    )
    return pairs_df


class PairwiseTrainDataset(Dataset):
    """Within-rater within-post (chosen, rejected) pairs for BT training.

    Each item carries both sides of a pair; they share the rater and post and
    differ only in the note.
    """

    def __init__(
        self,
        pairs_df: pd.DataFrame,
        rater_df: pd.DataFrame,
        notes_df: pd.DataFrame,
    ) -> None:
        pairs_df = _filter_resolvable_same_post_pairs(
            pairs_df, notes_df, rater_df, tag="pairwise train"
        )
        if len(pairs_df) == 0:
            raise ValueError("PairwiseTrainDataset: no pairs survived the filter")

        rater_idx = rater_df.set_index("raterParticipantId")[RATER_FEATURE_COLS]
        notes_idx = notes_df.set_index("noteId")[["note_text", "post_text"]]
        self.items: List[Dict[str, Any]] = []
        for r in pairs_df.itertuples(index=False):
            rf = rater_idx.loc[r.raterParticipantId]
            ctx = notes_idx.loc[r.chosen_noteId]
            rej = notes_idx.loc[r.rejected_noteId]
            self.items.append(
                {
                    "post_text": str(ctx["post_text"]),
                    "chosen_note_text": str(ctx["note_text"]),
                    "rejected_note_text": str(rej["note_text"]),
                    "rater_features": rf.to_numpy(dtype=np.float32),
                }
            )

    def __len__(self) -> int:
        return len(self.items)

    def __getitem__(self, idx: int) -> Dict[str, Any]:
        return self.items[idx]


@dataclass
class PairwiseTrainCollator:
    tokenizer: PreTrainedTokenizerBase
    max_len: int

    def __call__(self, batch: List[Dict[str, Any]]) -> Dict[str, torch.Tensor]:
        # Stack chosen over rejected into one [2B, T] batch so both sides go
        # through the backbone in a single forward pass.
        posts = [b["post_text"] for b in batch] + [b["post_text"] for b in batch]
        notes = (
            [b["chosen_note_text"] for b in batch]
            + [b["rejected_note_text"] for b in batch]
        )
        enc = self.tokenizer(
            posts,
            notes,
            truncation="longest_first",
            max_length=self.max_len - 1,
            padding=True,
            return_tensors="pt",
        )
        rf = np.stack([b["rater_features"] for b in batch])
        rater_features = torch.from_numpy(np.concatenate([rf, rf], axis=0)).float()
        return {
            "input_ids": enc["input_ids"],
            "attention_mask": enc["attention_mask"],
            "rater_features": rater_features,
        }


class PairwiseInferenceDataset(Dataset):
    """One item per (rater, post, note) triple; pairs are reassembled after scoring."""

    def __init__(
        self,
        records: List[Dict[str, Any]],
        rater_df: pd.DataFrame,
        notes_df: pd.DataFrame,
    ) -> None:
        rater_idx = rater_df.set_index("raterParticipantId")[RATER_FEATURE_COLS]
        notes_idx = notes_df.set_index("noteId")[["note_text", "post_text"]]
        self.items: List[Dict[str, Any]] = []
        for r in records:
            rf = rater_idx.loc[r["raterParticipantId"]]
            tx = notes_idx.loc[r["noteId"]]
            self.items.append(
                {
                    "post_text": str(tx["post_text"]),
                    "note_text": str(tx["note_text"]),
                    "rater_features": rf.to_numpy(dtype=np.float32),
                }
            )

    def __len__(self) -> int:
        return len(self.items)

    def __getitem__(self, idx: int) -> Dict[str, Any]:
        return self.items[idx]


@dataclass
class PairwiseInferenceCollator:
    tokenizer: PreTrainedTokenizerBase
    max_len: int

    def __call__(self, batch: List[Dict[str, Any]]) -> Dict[str, torch.Tensor]:
        posts = [b["post_text"] for b in batch]
        notes = [b["note_text"] for b in batch]
        enc = self.tokenizer(
            posts,
            notes,
            truncation="longest_first",
            max_length=self.max_len - 1,
            padding=True,
            return_tensors="pt",
        )
        rater_features = torch.from_numpy(
            np.stack([b["rater_features"] for b in batch])
        ).float()
        return {
            "input_ids": enc["input_ids"],
            "attention_mask": enc["attention_mask"],
            "rater_features": rater_features,
        }


def score_pairwise(
    model: nn.Module,
    tokenizer: PreTrainedTokenizerBase,
    pairs_df: pd.DataFrame,
    notes_df: pd.DataFrame,
    rater_df: pd.DataFrame,
    batch_size: int,
    max_len: int,
    device: torch.device,
) -> Dict[str, float]:
    """Pairwise ranking accuracy on a (chosen, rejected) pairs file.

    Two scores are computed per note:

    * rating-derived ``1 * P(SOMEWHAT_HELPFUL) + 2 * P(HELPFUL)`` -> ``pairwise/accuracy``
    * BT-head scalar                                        -> ``pairwise/bt_accuracy``

    A pair is correct iff score(chosen) > score(rejected); ties get 0.5 credit.
    Accuracies are also reported for pairs where both notes are AI-written
    (``both_ai``) and where exactly one is (``mixed``).

    Returns
    -------
    dict of str to float
    """
    pairs_df = _filter_resolvable_same_post_pairs(
        pairs_df, notes_df, rater_df, tag="pairwise"
    )

    # Each pair contributes 2 consecutive triples: chosen, then rejected.
    records: List[Dict[str, Any]] = []
    for r in pairs_df.itertuples(index=False):
        records.append(
            {"raterParticipantId": r.raterParticipantId, "noteId": r.chosen_noteId}
        )
        records.append(
            {"raterParticipantId": r.raterParticipantId, "noteId": r.rejected_noteId}
        )

    ds = PairwiseInferenceDataset(records, rater_df=rater_df, notes_df=notes_df)
    loader = torch.utils.data.DataLoader(
        ds,
        batch_size=batch_size,
        shuffle=False,
        num_workers=2,
        collate_fn=PairwiseInferenceCollator(tokenizer=tokenizer, max_len=max_len),
    )

    model.eval()
    autocast_ctx = (
        torch.autocast(device_type="cuda", dtype=torch.bfloat16)
        if device.type == "cuda"
        else torch.autocast(device_type=device.type, enabled=False)
    )
    scores_all: List[float] = []
    bt_scores_all: List[float] = []
    with torch.no_grad():
        for batch in loader:
            batch = {k: v.to(device) for k, v in batch.items()}
            with autocast_ctx:
                out = model(**batch)
                bt = model.score_bt(
                    batch["input_ids"],
                    batch["attention_mask"],
                    batch["rater_features"],
                )
            probs = torch.softmax(out["logits_rating"].float(), dim=-1)  # [B, 3]
            score = probs[:, 1] * 1.0 + probs[:, 2] * 2.0
            scores_all.extend(score.detach().float().cpu().tolist())
            bt_scores_all.extend(bt.detach().float().cpu().tolist())

    assert len(scores_all) == 2 * len(pairs_df), (
        f"pairwise score count {len(scores_all)} != 2 * {len(pairs_df)} pairs"
    )
    pairs_keep = pairs_df.copy()
    pairs_keep["score_chosen"] = scores_all[0::2]
    pairs_keep["score_rejected"] = scores_all[1::2]
    pairs_keep["bt_chosen"] = bt_scores_all[0::2]
    pairs_keep["bt_rejected"] = bt_scores_all[1::2]

    def _acc(df: pd.DataFrame, chosen_col: str, rejected_col: str) -> float:
        if len(df) == 0:
            return float("nan")
        diff = df[chosen_col] - df[rejected_col]
        win = (diff > 0).sum() + 0.5 * (diff == 0).sum()
        return float(win / len(df))

    cw = pairs_keep["chosen_writer"]
    rw = pairs_keep["rejected_writer"]
    both_ai = pairs_keep[(cw == "ai") & (rw == "ai")]
    mixed = pairs_keep[(cw == "ai") ^ (rw == "ai")]
    return {
        "pairwise/accuracy": _acc(pairs_keep, "score_chosen", "score_rejected"),
        "pairwise/bt_accuracy": _acc(pairs_keep, "bt_chosen", "bt_rejected"),
        "pairwise/n": float(len(pairs_keep)),
        "pairwise/both_ai_accuracy": _acc(both_ai, "score_chosen", "score_rejected"),
        "pairwise/both_ai_bt_accuracy": _acc(both_ai, "bt_chosen", "bt_rejected"),
        "pairwise/both_ai_n": float(len(both_ai)),
        "pairwise/mixed_accuracy": _acc(mixed, "score_chosen", "score_rejected"),
        "pairwise/mixed_bt_accuracy": _acc(mixed, "bt_chosen", "bt_rejected"),
        "pairwise/mixed_n": float(len(mixed)),
    }


# -----------------------------------------------------------------------------
# Trainer (grouped learning rates + BT auxiliary loss)
# -----------------------------------------------------------------------------


class RewardModelTrainer(Trainer):
    """Hugging Face Trainer with:

    * per-group learning rates (backbone / heads / rater projection)
    * the Bradley-Terry auxiliary loss, interleaved every ``bt_every`` steps
    * pairwise validation metrics appended to each evaluation
    """

    def __init__(
        self,
        *args: Any,
        backbone_lr: float,
        head_lr: float,
        rater_lr_mult: float,
        weight_decay: float,
        bt_train_dataset: PairwiseTrainDataset,
        bt_collator: PairwiseTrainCollator,
        w_bt: float,
        bt_batch_size: int,
        bt_target_passes: float,
        pairwise_val_pairs: Optional[pd.DataFrame] = None,
        pairwise_notes_df: Optional[pd.DataFrame] = None,
        pairwise_rater_df: Optional[pd.DataFrame] = None,
        pairwise_tokenizer: Optional[PreTrainedTokenizerBase] = None,
        pairwise_max_len: int = 1024,
        **kwargs: Any,
    ) -> None:
        super().__init__(*args, **kwargs)
        # The custom loss is a batch mean; Trainer handles accumulation scaling.
        self.model_accepts_loss_kwargs = False
        self.backbone_lr = backbone_lr
        self.head_lr = head_lr
        self.rater_lr_mult = rater_lr_mult
        self.cfg_weight_decay = weight_decay
        self.pairwise_val_pairs = pairwise_val_pairs
        self.pairwise_notes_df = pairwise_notes_df
        self.pairwise_rater_df = pairwise_rater_df
        self.pairwise_tokenizer = pairwise_tokenizer
        self.pairwise_max_len = pairwise_max_len

        # ---- Bradley-Terry auxiliary loss ----
        self.w_bt = w_bt
        self._bt_iter = None
        self._bt_microstep = 0
        # A separate loader cycles over the (much smaller) pair set on demand;
        # the rating loader alone defines the epoch.
        self._bt_loader = torch.utils.data.DataLoader(
            bt_train_dataset,
            batch_size=bt_batch_size,
            shuffle=True,
            drop_last=True,
            num_workers=2,
            collate_fn=bt_collator,
        )
        grad_accum = max(1, self.args.gradient_accumulation_steps)
        # Choose bt_every so the BT head makes ~bt_target_passes passes over
        # the pair set per rating epoch, rather than replaying the pairs on
        # every step.
        n_pairs = len(bt_train_dataset)
        eff_batch = self.args.train_batch_size * grad_accum
        steps_per_epoch = max(1.0, len(self.train_dataset) / max(1, eff_batch))
        bt_every = max(
            1,
            round(steps_per_epoch * bt_batch_size / max(1e-9, bt_target_passes * n_pairs)),
        )
        passes = steps_per_epoch * bt_batch_size / (bt_every * n_pairs)
        print(
            f"[bt] auto bt_every={bt_every} optim-steps "
            f"(~{passes:.2f} passes/epoch over {n_pairs:,} pairs; "
            f"steps/epoch~{steps_per_epoch:.0f}, bs={bt_batch_size})"
        )
        self.bt_every = bt_every
        # BT is added on one micro-step per bt_every optimizer steps.
        self._bt_period = bt_every * grad_accum
        self._bt_grad_accum = grad_accum

    def _next_bt_batch(self) -> Dict[str, torch.Tensor]:
        # Rebuild the iterator on exhaustion so each pass is reshuffled.
        if self._bt_iter is None:
            self._bt_iter = iter(self._bt_loader)
        try:
            return next(self._bt_iter)
        except StopIteration:
            self._bt_iter = iter(self._bt_loader)
            return next(self._bt_iter)

    def evaluate(
        self,
        eval_dataset=None,
        ignore_keys=None,
        metric_key_prefix: str = "eval",
    ):
        # With persistent dataloader workers, HF Trainer caches the eval loader
        # under a fixed key, so evaluate(test_ds) would silently reuse the val
        # loader. Drop the cache whenever an explicit dataset is passed.
        if eval_dataset is not None and getattr(self, "_eval_dataloaders", None):
            self._eval_dataloaders = {}
        metrics = super().evaluate(eval_dataset, ignore_keys, metric_key_prefix)
        if self.pairwise_val_pairs is not None and metric_key_prefix == "eval":
            device = next(self.model.parameters()).device
            pw = score_pairwise(
                model=self.model,
                tokenizer=self.pairwise_tokenizer,
                pairs_df=self.pairwise_val_pairs,
                notes_df=self.pairwise_notes_df,
                rater_df=self.pairwise_rater_df,
                batch_size=self.args.per_device_eval_batch_size,
                max_len=self.pairwise_max_len,
                device=device,
            )
            pw_prefixed = {f"{metric_key_prefix}_{k}": v for k, v in pw.items()}
            metrics.update(pw_prefixed)
            self.log(pw_prefixed)
        return metrics

    def _get_eval_sampler(self, eval_dataset):
        return SequentialSampler(eval_dataset)

    def _get_train_sampler(self, *args, **kwargs):
        # HF's LengthGroupedSampler expects tokenized `input_ids`; supply the
        # precomputed character-length proxy instead.
        return LengthGroupedSampler(
            batch_size=self.args.train_batch_size * self.args.gradient_accumulation_steps,
            dataset=self.train_dataset,
            lengths=self.train_dataset.lengths,
        )

    def compute_loss(self, model, inputs, return_outputs: bool = False, **kwargs):
        outputs = model(**inputs)
        loss = outputs["loss"]

        if model.training:
            self._bt_microstep += 1
        if model.training and self._bt_microstep % self._bt_period == 0:
            bt_batch = self._prepare_inputs(self._next_bt_batch())
            s = model.score_bt(
                bt_batch["input_ids"],
                bt_batch["attention_mask"],
                bt_batch["rater_features"],
            )  # [2B]: chosen stacked over rejected
            b = s.shape[0] // 2
            s_chosen, s_rejected = s[:b], s[b:]
            # -log sigmoid(s_chosen - s_rejected)
            bt_loss = F.softplus(-(s_chosen - s_rejected).float()).mean()
            # HF divides the returned loss by grad_accum before backward, and BT
            # is added on only one of grad_accum micro-steps; scaling by
            # grad_accum makes the per-optimizer-step BT gradient w_bt * bt_loss.
            loss = loss + self.w_bt * self._bt_grad_accum * bt_loss
            self.log(
                {
                    "loss_bt": float(bt_loss.detach()),
                    "bt_margin": float((s_chosen - s_rejected).detach().float().mean()),
                }
            )

        return (loss, outputs) if return_outputs else loss

    def create_optimizer(self):
        if self.optimizer is not None:
            return self.optimizer

        decay_param_names = set(get_parameter_names(self.model, [nn.LayerNorm]))
        decay_param_names = {n for n in decay_param_names if "bias" not in n}

        backbone_params: List[Tuple[str, nn.Parameter]] = []
        head_params: List[Tuple[str, nn.Parameter]] = []
        rater_params: List[Tuple[str, nn.Parameter]] = []
        for name, p in self.model.named_parameters():
            if not p.requires_grad:
                continue
            if name.startswith("rater_proj."):
                rater_params.append((name, p))
            elif name.startswith("rating_head.") or name.startswith("bt_head."):
                head_params.append((name, p))
            else:
                backbone_params.append((name, p))

        def split(group: List[Tuple[str, nn.Parameter]], wd: float):
            decay = [p for n, p in group if n in decay_param_names]
            no_decay = [p for n, p in group if n not in decay_param_names]
            out = []
            if decay:
                out.append({"params": decay, "weight_decay": wd})
            if no_decay:
                out.append({"params": no_decay, "weight_decay": 0.0})
            return out

        optim_groups: List[Dict[str, Any]] = []
        for g in split(backbone_params, self.cfg_weight_decay):
            g["lr"] = self.backbone_lr
            optim_groups.append(g)
        for g in split(head_params, self.cfg_weight_decay):
            g["lr"] = self.head_lr
            optim_groups.append(g)
        # rater_proj: no weight decay, higher learning rate.
        for g in split(rater_params, 0.0):
            g["lr"] = self.head_lr * self.rater_lr_mult
            g["weight_decay"] = 0.0
            optim_groups.append(g)

        self.optimizer = torch.optim.AdamW(optim_groups, lr=self.backbone_lr)
        return self.optimizer


# -----------------------------------------------------------------------------
# Metrics
# -----------------------------------------------------------------------------


def _softmax(x: np.ndarray, axis: int = -1) -> np.ndarray:
    x = x - x.max(axis=axis, keepdims=True)
    e = np.exp(x)
    return e / e.sum(axis=axis, keepdims=True)


def _multiclass_auc(y_true: np.ndarray, probs: np.ndarray) -> float:
    """Macro one-vs-rest AUC over the 3 rating levels (NaN if undefined)."""
    try:
        return float(
            roc_auc_score(y_true, probs, multi_class="ovr", average="macro", labels=[0, 1, 2])
        )
    except ValueError:
        return float("nan")


def _per_class_auc(y_true: np.ndarray, probs: np.ndarray) -> Dict[str, float]:
    """One-vs-rest AUC for each rating level, keyed by level name."""
    out: Dict[str, float] = {}
    for name, c in RATING_TO_INT.items():
        y = (y_true == c).astype(np.int32)
        try:
            out[name] = float(roc_auc_score(y, probs[:, c]))
        except ValueError:
            out[name] = float("nan")
    return out


def _binary_helpful_metrics(
    y_true: np.ndarray, probs: np.ndarray, eps: float = 1e-7
) -> Dict[str, float]:
    """HELPFUL-vs-NOT_HELPFUL log loss and AUC.

    SOMEWHAT_HELPFUL rows are dropped. The score is the conditional probability
    ``p = P(HELPFUL) / (P(HELPFUL) + P(NOT_HELPFUL))``, clipped to
    ``[eps, 1 - eps]`` for the log loss.
    """
    helpful_c = RATING_TO_INT["HELPFUL"]
    not_c = RATING_TO_INT["NOT_HELPFUL"]
    keep = (y_true == helpful_c) | (y_true == not_c)
    if keep.sum() == 0:
        return {"binary_logloss": float("nan"), "binary_auc": float("nan")}

    y = (y_true[keep] == helpful_c).astype(np.int32)
    p_help = probs[keep, helpful_c].astype(np.float64)
    p_not = probs[keep, not_c].astype(np.float64)
    p = p_help / np.clip(p_help + p_not, eps, None)

    if y.min() == y.max():
        auc = float("nan")
    else:
        auc = float(roc_auc_score(y, p))

    p_clip = np.clip(p, eps, 1.0 - eps)
    logloss = float(-(y * np.log(p_clip) + (1 - y) * np.log(1.0 - p_clip)).mean())
    return {"binary_logloss": logloss, "binary_auc": auc}


def compute_metrics(eval_pred) -> Dict[str, float]:
    """Rating metrics over all eval rows and over AI-written notes (``ai_*``).

    Reports accuracy, macro F1, macro / per-class one-vs-rest AUC, and the
    HELPFUL-vs-NOT_HELPFUL ``binary_logloss`` / ``binary_auc``.
    """
    logits_rating = eval_pred.predictions
    if isinstance(logits_rating, (tuple, list)):
        logits_rating = logits_rating[0]
    rating_label, writer_flag = eval_pred.label_ids

    rating_probs = _softmax(logits_rating, axis=-1)
    rating_pred = logits_rating.argmax(axis=-1)

    out: Dict[str, float] = {
        "rating_accuracy": float(accuracy_score(rating_label, rating_pred)),
        "rating_f1_macro": float(f1_score(rating_label, rating_pred, average="macro")),
        "rating_auc": _multiclass_auc(rating_label, rating_probs),
    }
    for name, auc in _per_class_auc(rating_label, rating_probs).items():
        out[f"rating_auc/{name}"] = auc
    out.update(_binary_helpful_metrics(rating_label, rating_probs))

    ai = writer_flag.astype(bool)
    out["ai_n"] = float(ai.sum())
    out["ai_rating_accuracy"] = float(accuracy_score(rating_label[ai], rating_pred[ai]))
    out["ai_rating_auc"] = _multiclass_auc(rating_label[ai], rating_probs[ai])
    for name, auc in _per_class_auc(rating_label[ai], rating_probs[ai]).items():
        out[f"ai_rating_auc/{name}"] = auc
    for k, v in _binary_helpful_metrics(rating_label[ai], rating_probs[ai]).items():
        out[f"ai_{k}"] = v
    return out


# -----------------------------------------------------------------------------
# Data loading
# -----------------------------------------------------------------------------


def load_lookup_tables(
    notes_path: str, rater_params_tsv: str
) -> Tuple[pd.DataFrame, pd.DataFrame]:
    """Load the note-text and rater-parameter lookup tables.

    Parameters
    ----------
    notes_path : str
        notes.parquet from the data-prep step.
    rater_params_tsv : str
        TSV with ``raterParticipantId`` and ``RATER_FEATURE_COLS``.

    Returns
    -------
    notes : pd.DataFrame
        Unique on ``noteId``; columns ``noteId``, ``note_text``, ``post_text``.
    rater : pd.DataFrame
        Unique on ``raterParticipantId``; raters with a null intercept dropped.
    """
    notes = pd.read_parquet(notes_path)
    notes = notes[["noteId", "note_text", "post_text"]].drop_duplicates(subset="noteId")

    rater = pd.read_csv(rater_params_tsv, sep="\t", dtype={"raterParticipantId": str})
    keep = ["raterParticipantId"] + RATER_FEATURE_COLS
    missing = [c for c in keep if c not in rater.columns]
    if missing:
        raise ValueError(f"rater params missing cols: {missing}")
    rater = rater[keep].dropna(subset=["internalRaterIntercept"]).drop_duplicates(
        subset="raterParticipantId"
    )
    return notes, rater


def build_split_datasets(
    ratings_split_path: str,
    notes_df: pd.DataFrame,
    rater_df: pd.DataFrame,
) -> Dict[str, RatingDataset]:
    """Build train/val/test datasets from ratings_split.parquet.

    Train uses ratings on both AI- and human-written notes; val and test are
    restricted to ratings on AI-written notes.
    """
    df = pd.read_parquet(ratings_split_path)
    out: Dict[str, RatingDataset] = {}
    for split in ["train", "val", "test"]:
        sub = df[df["split"] == split]
        if split in ("val", "test"):
            n_before = len(sub)
            sub = sub[sub["writer"].astype(str) == "ai"]
            print(f"[eval_ai_only] {split}: {n_before:,} -> {len(sub):,} AI-only rows")
        out[split] = RatingDataset(sub.reset_index(drop=True), notes_df, rater_df)
        print(f"[data] {split}: {len(out[split]):,}")
    return out


# -----------------------------------------------------------------------------
# Main
# -----------------------------------------------------------------------------


def parse_args() -> argparse.Namespace:
    p = argparse.ArgumentParser(description=__doc__,
                                formatter_class=argparse.RawDescriptionHelpFormatter)
    # Data
    p.add_argument("--ratings_split", type=str, required=True,
                   help="ratings_split.parquet (rating rows + 'split' column)")
    p.add_argument("--notes_parquet", type=str, required=True,
                   help="notes.parquet (noteId -> note_text, post_text)")
    p.add_argument("--rater_params_tsv", type=str, required=True,
                   help="rater parameter TSV (intercept + 5 factors per rater)")
    p.add_argument("--pairwise_train_file", type=str, required=True,
                   help="pairwise_train.parquet (BT training pairs)")
    p.add_argument("--pairwise_val_file", type=str, default=None,
                   help="pairwise_eval_val.parquet (pairwise metrics at each eval)")
    p.add_argument("--pairwise_test_file", type=str, default=None,
                   help="pairwise_eval_test.parquet (pairwise metrics at final test)")

    # Model
    p.add_argument("--model_name", type=str, default="answerdotai/ModernBERT-base")
    p.add_argument("--max_len", type=int, default=1024)

    # Optimization
    p.add_argument("--backbone_lr", type=float, default=2e-5)
    p.add_argument("--head_lr", type=float, default=1e-4)
    p.add_argument("--rater_lr_mult", type=float, default=5.0,
                   help="rater_proj learning rate = head_lr * rater_lr_mult")
    p.add_argument("--weight_decay", type=float, default=0.01)
    p.add_argument("--num_train_epochs", type=float, default=2.0)
    p.add_argument("--per_device_train_batch_size", type=int, default=96)
    p.add_argument("--per_device_eval_batch_size", type=int, default=96)
    p.add_argument("--gradient_accumulation_steps", type=int, default=1)
    p.add_argument("--warmup_steps", type=int, default=2000)

    # Bradley-Terry auxiliary loss
    p.add_argument("--w_bt", type=float, default=0.2,
                   help="weight of the Bradley-Terry auxiliary loss")
    p.add_argument("--bt_batch_size", type=int, default=4,
                   help="pairs per BT batch (each forwards 2x sequences)")
    p.add_argument("--bt_target_passes", type=float, default=0.75,
                   help="target passes over the pair set per rating epoch")

    # Evaluation / checkpointing
    p.add_argument("--eval_steps", type=int, default=6000)
    p.add_argument("--save_total_limit", type=int, default=None)
    p.add_argument("--save_only_model", action="store_true",
                   help="save weights only (no optimizer state) in checkpoints")
    p.add_argument("--metric_for_best_model", type=str, default="ai_binary_logloss",
                   help="checkpoint selection metric; names containing 'loss' are "
                        "lower-is-better, all others higher-is-better")

    # Misc
    p.add_argument("--logging_steps", type=int, default=50)
    p.add_argument("--dataloader_num_workers", type=int, default=6)
    p.add_argument("--dataloader_prefetch_factor", type=int, default=2)
    p.add_argument("--seed", type=int, default=42)
    p.add_argument("--output_dir", type=str, required=True)
    p.add_argument("--run_name", type=str, default=None)
    p.add_argument("--report_to", type=str, default="none",
                   help="Hugging Face report_to integration, e.g. 'wandb' or 'none'")
    return p.parse_args()


def main() -> None:
    args = parse_args()

    set_seed(args.seed)
    random.seed(args.seed)
    np.random.seed(args.seed)
    torch.manual_seed(args.seed)

    notes_df, rater_df = load_lookup_tables(args.notes_parquet, args.rater_params_tsv)
    print(f"[lookup] loaded notes={len(notes_df):,}  raters={len(rater_df):,}")

    splits = build_split_datasets(args.ratings_split, notes_df, rater_df)
    train_ds, val_ds, test_ds = splits["train"], splits["val"], splits["test"]

    tokenizer = AutoTokenizer.from_pretrained(args.model_name)
    collator = TokenizingCollator(tokenizer=tokenizer, max_len=args.max_len)
    model = RaterPrefixModernBERT(model_name=args.model_name, rater_dim=RATER_DIM)

    bf16_ok = torch.cuda.is_available() and torch.cuda.is_bf16_supported()
    greater_is_better = "loss" not in args.metric_for_best_model.lower()
    print(
        f"[select] metric_for_best_model={args.metric_for_best_model} "
        f"greater_is_better={greater_is_better}"
    )

    training_args = TrainingArguments(
        output_dir=args.output_dir,
        run_name=args.run_name,
        seed=args.seed,
        num_train_epochs=args.num_train_epochs,
        per_device_train_batch_size=args.per_device_train_batch_size,
        per_device_eval_batch_size=args.per_device_eval_batch_size,
        gradient_accumulation_steps=args.gradient_accumulation_steps,
        learning_rate=args.backbone_lr,  # base LR; per-group LRs set in create_optimizer
        weight_decay=args.weight_decay,
        warmup_steps=args.warmup_steps,
        lr_scheduler_type="cosine",
        max_grad_norm=1.0,
        bf16=bf16_ok,
        fp16=(not bf16_ok) and torch.cuda.is_available(),
        eval_strategy="steps",
        eval_steps=args.eval_steps,
        save_strategy="steps",
        save_steps=args.eval_steps,
        save_total_limit=args.save_total_limit,
        save_only_model=args.save_only_model,
        logging_strategy="steps",
        logging_steps=args.logging_steps,
        load_best_model_at_end=True,
        metric_for_best_model=args.metric_for_best_model,
        greater_is_better=greater_is_better,
        report_to=[args.report_to] if args.report_to != "none" else [],
        label_names=["rating_label", "writer_flag"],
        remove_unused_columns=False,
        dataloader_num_workers=args.dataloader_num_workers,
        dataloader_prefetch_factor=(
            args.dataloader_prefetch_factor if args.dataloader_num_workers > 1 else None
        ),
        dataloader_pin_memory=True,
        dataloader_persistent_workers=args.dataloader_num_workers > 0,
        group_by_length=True,
    )

    pairs_val = None
    if args.pairwise_val_file:
        pairs_val = pd.read_parquet(args.pairwise_val_file)
        print(f"[pairwise val] {len(pairs_val):,} pairs")

    bt_train_ds = PairwiseTrainDataset(
        pd.read_parquet(args.pairwise_train_file), rater_df=rater_df, notes_df=notes_df
    )
    bt_collator = PairwiseTrainCollator(tokenizer=tokenizer, max_len=args.max_len)
    print(f"[bt] w_bt={args.w_bt} pairs={len(bt_train_ds):,}")

    trainer = RewardModelTrainer(
        model=model,
        args=training_args,
        train_dataset=train_ds,
        eval_dataset=val_ds,
        data_collator=collator,
        compute_metrics=compute_metrics,
        backbone_lr=args.backbone_lr,
        head_lr=args.head_lr,
        rater_lr_mult=args.rater_lr_mult,
        weight_decay=args.weight_decay,
        bt_train_dataset=bt_train_ds,
        bt_collator=bt_collator,
        w_bt=args.w_bt,
        bt_batch_size=args.bt_batch_size,
        bt_target_passes=args.bt_target_passes,
        pairwise_val_pairs=pairs_val,
        pairwise_notes_df=notes_df,
        pairwise_rater_df=rater_df,
        pairwise_tokenizer=tokenizer,
        pairwise_max_len=args.max_len,
    )

    trainer.train()
    print(f"[select] best checkpoint: {trainer.state.best_model_checkpoint} "
          f"({args.metric_for_best_model}={trainer.state.best_metric})")

    # Final metrics are computed with the best checkpoint loaded.
    out_dir = Path(args.output_dir)
    out_dir.mkdir(parents=True, exist_ok=True)
    val_metrics = trainer.evaluate(val_ds)
    print(f"[final val] {val_metrics}")
    with open(out_dir / "val_metrics.json", "w") as f:
        json.dump(val_metrics, f, indent=2)

    test_metrics = trainer.evaluate(test_ds, metric_key_prefix="test")
    if args.pairwise_test_file:
        pw = score_pairwise(
            model=trainer.model,
            tokenizer=tokenizer,
            pairs_df=pd.read_parquet(args.pairwise_test_file),
            notes_df=notes_df,
            rater_df=rater_df,
            batch_size=args.per_device_eval_batch_size,
            max_len=args.max_len,
            device=next(trainer.model.parameters()).device,
        )
        for k, v in pw.items():
            test_metrics[f"test_{k}"] = v
    print(f"[final test] {test_metrics}")
    with open(out_dir / "test_metrics.json", "w") as f:
        json.dump(test_metrics, f, indent=2)


if __name__ == "__main__":
    main()
