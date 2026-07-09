"""Fit a Jacobian lens on ppiGPLM using the 1,084 PRS (real) MED4 pairs.

skip_first is overridden from jlens' default of 16 down to 6 -- the length of
the "<ps1>," marker -- so residue 0 of protein1 (the N-terminus the paradox
is about) is included in the fitted average Jacobian rather than being
discarded as an "attention sink" position.
"""
import logging
import pathlib
import sys
import time

import pandas as pd
import torch

HERE = pathlib.Path(__file__).resolve().parent  # analysis/jacobian_lens
REPO_ROOT = HERE.parents[1]  # repo root
sys.path.insert(0, str(HERE / "vendor_jacobian_lens"))
sys.path.insert(0, str(HERE))
from ppi_lens_adapter import PPIGPLMLensModel  # noqa: E402
from jlens.fitting import fit  # noqa: E402
from jlens import configure_logging  # noqa: E402

REPO = str(REPO_ROOT)
OUT_DIR = str(REPO_ROOT / "results")

MAX_SEQ_LEN = 512
MARKER_LEN = len("<ps1>,")  # =6, position where protein1 residue 0 lands


def build_prompts() -> list[str]:
    df = pd.read_csv(f"{REPO}/data/formatted_real_PPIs.csv")
    prompts = []
    for _, row in df.iterrows():
        p1, p2 = row["protein1"], row["protein2"]
        prompt = f"<ps1>,{p1},<ps2>,{p2},<"
        if len(prompt) <= MAX_SEQ_LEN:  # no truncation -> N-terminus of protein1 always intact
            prompts.append(prompt)
    return prompts


def main() -> None:
    configure_logging(level=logging.INFO)
    logging.basicConfig(level=logging.INFO, format="%(asctime)s %(message)s")

    model = PPIGPLMLensModel(
        f"{REPO}/model/out_3e/ckpt.pt",
        f"{REPO}/model/data/meta.pkl",
        device="cuda",
    )
    prompts = build_prompts()
    print(f"fitting on {len(prompts)} PRS prompts (no truncation, max_seq_len={MAX_SEQ_LEN})", flush=True)

    t0 = time.time()
    lens = fit(
        model,
        prompts,
        source_layers=list(range(11)),  # layers 0..10; target is layer 11 (final block)
        target_layer=11,
        dim_batch=16,
        max_seq_len=MAX_SEQ_LEN,
        skip_first=MARKER_LEN,
        checkpoint_path=f"{OUT_DIR}/ppiGPLM_lens_fit_checkpoint.pt",
        checkpoint_every=25,
        resume=True,
    )
    dt = time.time() - t0
    print(f"fit done in {dt:.0f}s: {lens}", flush=True)
    lens.save(f"{OUT_DIR}/ppiGPLM_jacobian_lens.pt")
    print("saved to ppiGPLM_jacobian_lens.pt", flush=True)
    print(f"peak GPU mem: {torch.cuda.max_memory_allocated()/1e9:.2f} GB", flush=True)


if __name__ == "__main__":
    main()
