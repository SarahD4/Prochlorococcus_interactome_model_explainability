"""Apply the fitted ppiGPLM Jacobian lens at N-terminal vs control positions,
across layers, for real (PRS) vs random (RRS) pairs.

Question: is the interaction decision ('1' vs '0') already linearly readable
-- in the model's own output-vocabulary basis -- from the residual stream at
early N-terminal positions, well before the model has attended to protein2 or
reached the final decision token? If so, at which layer does it appear, and
is it specific to real pairs (vs random) and to the N-terminus (vs middle/
C-terminal control positions)? This is a representation-level complement to
the DeepLift/IG attribution and alanine-ablation results already in the
manuscript.
"""
import pathlib
import random
import sys

import pandas as pd
import torch

HERE = pathlib.Path(__file__).resolve().parent  # analysis/jacobian_lens
REPO_ROOT = HERE.parents[1]  # repo root
sys.path.insert(0, str(HERE / "vendor_jacobian_lens"))
sys.path.insert(0, str(HERE))
from ppi_lens_adapter import PPIGPLMLensModel  # noqa: E402
from jlens.lens import JacobianLens  # noqa: E402

REPO = str(REPO_ROOT)
OUT_DIR = str(REPO_ROOT / "results")
MARKER_LEN = len("<ps1>,")
N_WINDOW = 15  # matches manuscript's "position 0 versus position 14" framing
MAX_SEQ_LEN = 2048
N_SAMPLE = 200
SEED = 0


def build_pairs(csv_path: str) -> list[tuple[str, str]]:
    df = pd.read_csv(csv_path)
    out = []
    for _, row in df.iterrows():
        p1, p2 = row["protein1"], row["protein2"]
        prompt = f"<ps1>,{p1},<ps2>,{p2},<"
        if len(prompt) <= MAX_SEQ_LEN and len(p1) >= 20:  # need room for N/C/middle windows
            out.append((p1, p2, prompt))
    return out


def position_sets(p1: str) -> dict[str, list[int]]:
    """Absolute prompt positions (0-indexed) for protein1's N-term/C-term/middle windows."""
    n = len(p1)
    w = min(N_WINDOW, n)
    nterm = list(range(MARKER_LEN, MARKER_LEN + w))
    cterm = list(range(MARKER_LEN + n - w, MARKER_LEN + n))
    mid_start = MARKER_LEN + n // 2 - w // 2
    middle = list(range(mid_start, mid_start + w)) if n >= 3 * w else []
    return {"n_term": nterm, "c_term": cterm, "middle": middle}


def analyze_dataset(model, lens, pairs, dataset_name, n_sample, layers) -> pd.DataFrame:
    random.seed(SEED)
    sample = random.sample(pairs, min(n_sample, len(pairs)))
    rows = []
    for i, (p1, p2, prompt) in enumerate(sample):
        pos_sets = position_sets(p1)
        all_positions = sorted(set(sum(pos_sets.values(), [])))
        if not all_positions or max(all_positions) >= min(len(prompt), MAX_SEQ_LEN):
            continue  # would be out of range after any truncation; skip defensively
        lens_logits, model_logits, input_ids = lens.apply(
            model, prompt, layers=layers, positions=all_positions, max_seq_len=MAX_SEQ_LEN
        )
        pos_to_idx = {p: idx for idx, p in enumerate(all_positions)}

        # ground-truth final decision (last position, real lm_head, not lens-transported)
        with torch.no_grad():
            h = model.forward(input_ids)
            final_logits = model.unembed(h[:, [-1], :])[0, 0]
            final_prob1 = torch.softmax(final_logits, dim=0)[model.token_1_id].item()

        for region, positions in pos_sets.items():
            if not positions:
                continue
            for layer in layers:
                margins = []
                for p in positions:
                    logit_vec = lens_logits[layer][pos_to_idx[p]]
                    margin = (logit_vec[model.token_1_id] - logit_vec[model.token_0_id]).item()
                    margins.append(margin)
                rows.append(
                    {
                        "dataset": dataset_name,
                        "pair_idx": i,
                        "region": region,
                        "layer": layer,
                        "mean_margin": sum(margins) / len(margins),
                        "final_model_prob1": final_prob1,
                    }
                )
        if (i + 1) % 25 == 0:
            print(f"  {dataset_name}: {i+1}/{len(sample)}", flush=True)
    return pd.DataFrame(rows)


def main() -> None:
    model = PPIGPLMLensModel(
        f"{REPO}/model/out_3e/ckpt.pt",
        f"{REPO}/model/data/meta.pkl",
        device="cuda",
    )
    lens = JacobianLens.load(f"{OUT_DIR}/ppiGPLM_jacobian_lens.pt")
    layers = lens.source_layers
    print("lens:", lens, "layers:", layers, flush=True)

    real_pairs = build_pairs(f"{REPO}/data/formatted_real_PPIs.csv")
    random_pairs = build_pairs(f"{REPO}/data/formatted_random_PPIs.csv")
    print(f"real pairs usable: {len(real_pairs)}, random pairs usable: {len(random_pairs)}", flush=True)

    df_real = analyze_dataset(model, lens, real_pairs, "real_PRS", N_SAMPLE, layers)
    df_random = analyze_dataset(model, lens, random_pairs, "random_RRS", N_SAMPLE, layers)
    df = pd.concat([df_real, df_random], ignore_index=True)
    df.to_csv(f"{OUT_DIR}/ppiGPLM_lens_readout.csv", index=False)
    print(f"saved {len(df)} rows to ppiGPLM_lens_readout.csv", flush=True)


if __name__ == "__main__":
    main()
