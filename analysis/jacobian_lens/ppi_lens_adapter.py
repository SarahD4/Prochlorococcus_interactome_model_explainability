"""jlens.protocol.LensModel adapter for ppiGPLM (nanoGPT-style char-level GPT-2).

ppiGPLM predicts protein-protein interaction by reading off P('1') vs P('0')
at the final position of the prompt "<ps1>,{protein1},<ps2>,{protein2},<".
Its forward pass (model/model.py in the ppiGPLM interpretability repo) is a
standard pre-LN decoder transformer with causal self-attention and a tied
lm_head, which is exactly the shape jlens.protocol.LensModel expects: no
HuggingFace wrapping needed, just expose n_layers/d_model/layers/encode/
forward/unembed directly against the nanoGPT module.
"""
from __future__ import annotations

import pathlib
import pickle
import sys

import torch

REPO_ROOT = pathlib.Path(__file__).resolve().parents[2]  # analysis/jacobian_lens -> analysis -> repo root
sys.path.insert(0, str(REPO_ROOT / "model"))
from model import GPT, GPTConfig  # noqa: E402


class PPIGPLMTokenizer:
    """Minimal decode()-only tokenizer; jlens' fitting/apply paths never call it."""

    def __init__(self, stoi: dict, itos: dict):
        self.stoi = stoi
        self.itos = itos

    def decode(self, token_ids) -> str:
        if torch.is_tensor(token_ids):
            token_ids = token_ids.tolist()
        return "".join(self.itos.get(int(i), "?") for i in token_ids)


class PPIGPLMLensModel:
    """LensModel over the ppiGPLM checkpoint."""

    def __init__(
        self,
        ckpt_path: str,
        meta_path: str,
        device: str = "cuda",
    ) -> None:
        self.device = torch.device(device)

        checkpoint = torch.load(ckpt_path, map_location=self.device, weights_only=False)
        config = GPTConfig(**checkpoint["model_args"])
        model = GPT(config)
        state_dict = checkpoint["model"]
        unwanted_prefix = "_orig_mod."
        for k, v in list(state_dict.items()):
            if k.startswith(unwanted_prefix):
                state_dict[k[len(unwanted_prefix):]] = state_dict.pop(k)
        model.load_state_dict(state_dict)
        model.to(self.device)
        model.eval()
        for p in model.parameters():
            p.requires_grad_(False)
        self.model = model
        self.config = config

        with open(meta_path, "rb") as f:
            meta = pickle.load(f)
        self.stoi = meta["stoi"]
        self.itos = meta["itos"]
        self.tokenizer = PPIGPLMTokenizer(self.stoi, self.itos)
        self.token_1_id = self.stoi["1"]
        self.token_0_id = self.stoi["0"]

        # --- LensModel protocol surface ---
        self.n_layers = config.n_layer
        self.d_model = config.n_embd
        self.layers = model.transformer.h  # nn.ModuleList; hook-compatible (Block.forward returns a plain Tensor)

    @property
    def input_device(self) -> torch.device:
        return self.device

    def encode(self, text: str, *, max_length: int = 4096) -> torch.Tensor:
        ids = [self.stoi[c] for c in text]
        if len(ids) > max_length:
            # keep the tail (decision token + protein2) if a prompt ever exceeds
            # max_length; the fitting/analysis corpora are pre-filtered to avoid
            # this so protein1's N-terminus is never the part that gets cut.
            ids = ids[-max_length:]
        return torch.tensor([ids], dtype=torch.long, device=self.device)

    def forward(self, input_ids: torch.Tensor):
        """Residual stack only, no LM head -- matches LensModel.forward contract."""
        b, t = input_ids.size()
        pos = torch.arange(0, t, dtype=torch.long, device=input_ids.device)
        tok_emb = self.model.transformer.wte(input_ids)
        pos_emb = self.model.transformer.wpe(pos)
        x = self.model.transformer.drop(tok_emb + pos_emb)
        for block in self.model.transformer.h:
            x = block(x)
        return x

    def unembed(self, residual: torch.Tensor) -> torch.Tensor:
        x = self.model.transformer.ln_f(residual.to(self.device))
        return self.model.lm_head(x)
