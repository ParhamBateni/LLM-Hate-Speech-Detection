# Definition-based hate speech detection with LLMs

Code for **Tailoring In-Context Learning Techniques for Definition-Based Hate Speech Detection in Large Language Models**, submitted to [WOAH 2026](https://www.workshopononlineabuse.com/) (EMNLP 2026).

A link to the paper will be added here after the conference presentation.

This repo contains hateful example text from HateCheck. It is only used for evaluation.

## What this is about

Hate speech is not defined the same way everywhere: a dataset, a platform policy, and a criminal code can draw different lines. Instruction-tuned LLMs are still often used as hate speech classifiers with a generic prompt, so it is unclear whether they follow a *given* definition or just their own default.

We keep the HateCheck test cases fixed and relabel them under three scopes (HateCheck itself, Reddit’s policy, and Bulgarian criminal law), encoded with Hate Speech Criteria (HSC). We then prompt three ~2–3B instruction-tuned models (Gemma-2, Llama-3.2, Qwen2.5) with or without that definition, using zero-shot and three few-shot example-selection methods.

The main question is: **how well can prompting alone make an LLM follow a specified hate speech definition?** More specifically:

1. Does putting the definition in the prompt help compared to a generic instruction?
2. Which prompting strategy works better (zero-shot vs few-shot, and how examples are chosen)?
3. How much does the model matter?

In short: adding the definition is not reliably helpful, and scores drop on the Reddit and Bulgaria relabelings. Few-shot prompting, especially nearest-neighbor examples, moves the numbers more than the definition text does. Residual errors on those relabeled sets concentrate on cases the definition *excludes* (dominant-group targets on Reddit; gender / sexual orientation / disability on Bulgaria). Details are in the paper.

## Layout

```
config.yaml          experiment grid (datasets, models, prompts)
example.env          Hugging Face token template
requirements.txt
src/                 inference loop
  main.py            runs the full grid from config.yaml
scripts/run_job.sh   optional tmux wrapper around main.py
data/
  datasets/HateCheck/   processed and relabeled CSVs
  definitions/          HSC JSON for HateCheck, Reddit, Bulgaria
notebooks/           data prep, relabeling, analysis
  tables/            paper result CSVs (scores, significance tests, HSC errors)
  figures/           paper plots (relabeling, few-shot viz, HSC errors)
runs/final_run/      paper predictions, prompts, and scores (~26 MB)
```

Notebooks, roughly in order:

- `data_processing.ipynb` — drop spelling-error HateCheck cases, add dominant-group templates
- `relabel_hatecheck.ipynb` — assign Reddit / Bulgaria gold labels from HSC tags
- `fewshot_visualization.ipynb` — silhouette / cluster plots for few-shot design
- `results_analysis.ipynb`, `statistical_testing.ipynb`, `error_analysis.ipynb` — scores, significance tests, functionality errors
- `hsc_aspect_error_analysis.ipynb` — where models still miss out-of-scope cases after the definition is given

## Setup

Python 3.11. A GPU is strongly recommended; the paper run used a single RTX A4000 (16 GB).

```bash
python3 -m venv .venv
source .venv/bin/activate   # Windows: .venv\Scripts\activate
pip install -r requirements.txt
```

Copy `example.env` to `.env` and set `HF_TOKEN`. Llama-3.2 and Gemma-2 are gated on Hugging Face, so the token needs access to those models.

Edit `config.yaml` to change which datasets, definitions, models, or prompting methods are run. Generation uses seed 42 and greedy decoding (`do_sample: false`, `temperature: 0.0`).

## Running experiments

From the repo root:

```bash
python src/main.py
```

Or, on a remote box, `bash scripts/run_job.sh` starts the same command in tmux and tees logs under `logs/`.

Each setting is written under `runs/<timestamp>/…/predictions.csv` (plus a copied `config.yaml` and the system prompt).

The paper grid lives in `runs/final_run` (predictions, system prompts, and `scores_comparison.csv`). Derived tables and figures used in the paper are under `notebooks/tables/` and `notebooks/figures/`. You can inspect those without a GPU. To recompute them, point the analysis notebooks at `runs/final_run`.
