# Prochlorococcus Interactome Model Explainability

Interpretability and explainability analysis of ppiGPT protein-protein interaction predictions in *Prochlorococcus* MED4.

This repository contains all code, data, and results for the model interpretability analyses described in the accompanying manuscript.

## Overview

ppiGPT is a 12-layer, 12-head, 768-dimensional GPT-2 architecture (84.98M parameters) trained from scratch with a 29-token character-level vocabulary (20 amino acids + 9 special tokens) to predict protein-protein interactions. This repository provides the interpretability pipeline used to understand what sequence features drive the model's predictions.

## Repository Structure

```
.
├── model/                          # ppiGPT model architecture
│   └── model.py                    # GPT-2 based PPI prediction model (GPTConfig, GPT)
│
├── data/                           # Input datasets
│   ├── formatted_real_PPIs.csv     # 1,084 Y2H-validated interactions (PRS)
│   └── formatted_random_PPIs.csv   # 1,084 randomly paired MED4 proteins (RRS)
│
├── analysis/                       # Interpretability methods
│   ├── deeplift/                   # Captum DeepLift attribution analysis
│   ├── integrated_gradients/       # Captum Integrated Gradients
│   ├── gradient_attribution/       # Gradient-based attribution
│   ├── lrp/                        # Layer-wise Relevance Propagation
│   ├── perturbation/               # Alanine substitution scanning
│   ├── attention/                  # Attention pattern extraction and analysis
│   ├── counterfactual/             # Counterfactual sequence generation
│   ├── probes/                     # Linear probing of internal representations
│   ├── motif_discovery/            # Attribution-guided motif discovery
│   └── uncertainty_quantification.py
│
├── af3_structural_analysis/        # AlphaFold3 N-terminal ablation experiments
│   ├── submit_to_af3_batch.py      # AF3 batch submission
│   ├── analyze_af3_results.py      # AF3 output parsing
│   └── analyze_med4_results_20251011.py  # ipTM analysis of N20A ablations
│
├── visualization/                  # Figure generation
│   ├── create_explainability_dashboard.py  # Six-panel summary dashboard
│   ├── create_pair_attribution_heatmap.py  # Per-residue heatmaps
│   └── figure_config.py            # Matplotlib style configuration
│
├── results/                        # Analysis outputs
│   └── integrated_gradients_*.csv           # IG attribution tables
│
├── figures/                        # Publication figures
│   ├── DeepLift-gptPPI-FIGURE.pdf           # Per-residue attribution heatmaps
│   ├── DeepLift-gptPPI-sixPanel.pdf         # Summary dashboard
│   └── *.svg                                # Vector figure components
│
└── documentation/
    ├── data_provenance.md           # Full chain of custody for all outputs
    └── results_and_figure_legends.txt
```

## Key Results

| Metric | PRS (Real PPIs) | RRS (Random) |
|--------|-----------------|--------------|
| Mean prediction | 0.718 +/- 0.347 | 0.207 +/- 0.152 |
| Mean \|attribution\| | 0.0081 +/- 0.0047 | 0.0083 +/- 0.0047 |
| Unique motifs (3-5 mers) | 80,119 | 270,576 |

| Test | Statistic | p-value |
|------|-----------|---------|
| Two-sample t-test | t = 44.36 | 2.76 x 10^-306 |
| Mann-Whitney U | U = 1,018,862 | 1.64 x 10^-192 |
| Cohen's d | 1.91 (large) | -- |

## Model Checkpoint & Large Files

The trained ppiGPT checkpoint and large result files are hosted on Hugging Face:

**https://huggingface.co/GreenGenomicsLab/Prochlorococcus_interactome_model_explainability**

Download and place them locally before running analysis scripts:

| HF Path | Local Path | Size | Description |
|---------|------------|------|-------------|
| `model/out_3e/ckpt.pt` | `model/out_3e/ckpt.pt` | 1.0 GB | ppiGPT model checkpoint (84.98M params, 3 epochs) |
| `model/data/meta.pkl` | `model/data/meta.pkl` | 343 B | Tokenizer metadata (29-token vocabulary) |
| `results/deeplift_motif_analysis_results.pkl` | `results/deeplift_motif_analysis_results.pkl` | 78 MB | Complete DeepLift attributions for all 2,168 pairs |
| `results/integrated_gradients_random_ppi_per_token_attributions.csv` | `results/integrated_gradients_random_ppi_per_token_attributions.csv` | 174 MB | Per-token IG attributions |

## Reproduction

```bash
pip install -r requirements.txt

# 1. DeepLift attribution analysis (requires GPU + model checkpoint)
python analysis/deeplift/captum_deeplift_proper_analysis.py

# 2. Generate summary dashboard from results
python visualization/create_explainability_dashboard.py

# 3. Generate per-pair heatmaps
python visualization/create_pair_attribution_heatmap.py
```

## Software

- Python 3.10+
- PyTorch >= 2.0.0
- Captum (DeepLift, Integrated Gradients)
- scipy, numpy, matplotlib, seaborn, pandas

## Contributing (Co-authors)

This repository accompanies a manuscript currently in preparation. If you are a co-author, please add your contributions to the appropriate sections:

- **Data contributors**: Add input datasets (e.g., Y2H results, proteomics, structural data) to `data/` with a brief description in `documentation/data_provenance.md`
- **Analysis contributors**: Add analysis scripts to the relevant `analysis/` subdirectory. Use relative paths (see existing scripts for the pattern)
- **Figure contributors**: Add publication-ready figures to `figures/` and update `documentation/results_and_figure_legends.txt`
- **AF3 structural analysis**: Add AlphaFold3 results or N-terminal ablation data to `af3_structural_analysis/`

When adding files, please:
1. Include a brief docstring at the top of any new script describing what it does
2. Use relative paths — no hardcoded local or HPC paths
3. Note any additional dependencies not in `requirements.txt`
4. Update this README if you add a new analysis method or directory

For large files (>50 MB), upload to the companion Hugging Face repository:
https://huggingface.co/GreenGenomicsLab/Prochlorococcus_interactome_model_explainability

## License

MIT License. See [LICENSE](LICENSE).
