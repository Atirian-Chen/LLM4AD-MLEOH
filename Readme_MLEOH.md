# Readme_MLEOH

## MLEOH: Multi-LLM Collaborative Heuristic Design for EoH

This repository contains my project-specific extension built on top of **LLM4AD** for my undergraduate thesis:

> **Collaborative Multi-LLM Heuristic Design in the Evolution of Heuristics (EoH) Framework**

The original `README.md` in this repository belongs to **LLM4AD** and describes the base platform.
This file, **`Readme_MLEOH.md`**, focuses only on the code, scripts, and experiment organization added for my thesis project.

---

## 1. Project Overview

The goal of this project is to extend the original **single-LLM EoH** setting into a **multi-LLM collaborative EoH framework**.
Instead of using one backend model for all heuristic generation steps, this project introduces a **router + multiple backend LLMs** design, so different models can be selected for different stages of heuristic search.

The current implementation focuses on the **Online Bin Packing Problem (OBP)** and compares several routing strategies under a unified search budget.

### Implemented routing methods

- **Single-model replication**
  - DeepSeek
  - Qwen
  - Doubao
- **Random Router**
- **Static Rule Router**
- **Statistic LLM Router**
- **Bandit Router**
- **TextGrad Router**
- **TextGrad Router Beta / Delta variants**

### Main research questions

This project mainly studies:

1. Whether multi-LLM collaboration is feasible in EoH.
2. Whether different LLM backends show complementary behavior during heuristic search.
3. How different routing strategies affect performance, stability, and specialization.
4. Whether a learnable and interpretable routing policy can be built using textual strategy optimization.

---

## 2. Relationship to LLM4AD

This project is **not a replacement** for LLM4AD.
It is an **extension layer** built on top of the original framework.

In particular, this project reuses:

- the LLM4AD task interface,
- the EoH search pipeline,
- the OBP evaluation environment,
- the profiling and logging infrastructure.

The main modification is at the **LLM invocation layer**:

- original EoH: one fixed backend LLM,
- this project: one router chooses among multiple backend LLMs.

So the key idea is:

> keep the original EoH search logic as unchanged as possible, and only replace the fixed generator with a router-based multi-LLM generator.

---

## 3. Directory Structure

The thesis-related code is mainly placed under:

```text
MLEOH_3llm/
├── replication/               # Single-model replication / baseline runs
├── random/                    # Random router
├── static_rule/               # Static rule-based router
├── statistic_llmrouter/       # Statistic-based LLM router
├── bandit_learned/            # Bandit router (LinUCB-style)
├── textgrad_router/           # Early TextGrad router version
├── textgrad_router_beta/      # Beta version of TextGrad router
├── textgrad_router_delta/     # Final TextGrad delta version used in later experiments
├── run_all_final_progress.py  # Unified script for full experiment pipeline
├── run_ablation.py            # Ablation runner
├── all_results/               # Aggregated experiment results
└── output_form.txt            # Output formatting notes / helper text
```

### Important subfolders

#### `replication/`
Used for reproducing the single-model EoH baselines on OBP.

Typical scripts:
- `test_eoh_obp4ds.py`
- `test_eoh_obp4qwen.py`
- `test_eoh_obp4doubao.py`
- `run_eoh_obp_repro.py`

#### `random/`
Implements the random multi-LLM routing baseline.

Typical script:
- `run_eoh_obp_random_router.py`

#### `static_rule/`
Implements rule-based routing using operator type, error signals, prompt length, code density, and similar hand-crafted signals.

Typical scripts:
- `rule_router.py`
- `rule_router2.py`
- `run_eoh_obp_rule_router.py`

#### `statistic_llmrouter/`
Uses a router LLM together with historical operator × backend statistics.

Typical scripts:
- `llm_router.py`
- `train_eoh_obp_llm_router.py`
- `test_eoh_obp_llm_router.py`

#### `bandit_learned/`
Implements a contextual bandit router, mainly using a LinUCB-style update.

Typical scripts:
- `bandit_router.py`
- `router_trace.py`
- `train_eoh_obp_bandit_router.py`
- `test_eoh_obp_bandit_router.py`

#### `textgrad_router_delta/`
Contains the more complete TextGrad-style router used in later experiments.
This version supports profile text, critic-guided updates, reward shaping, controlled exploration, router trace logging, and frozen-state testing.

Typical scripts:
- `textgrad_router3.py`
- `train_eoh_obp_textgrad_router3.py`
- `test_eoh_obp_textgrad_router3.py`

---

## 4. Environment Setup

This project is built on top of the original LLM4AD environment.
Please first refer to the original `README.md` and `environment.yml` in the repository root.

A typical setup process is:

```bash
conda env create -f environment.yml
conda activate llm4ad
```

If you prefer `pip`, make sure the core dependencies used by LLM4AD and this project are installed, especially:

- Python 3.9+
- `numpy`
- `pandas`
- `tqdm`
- `matplotlib`
- `jupyter`
- any dependencies required by LLM4AD itself

---

## 5. API Keys and Model Configuration

This project uses multiple backend LLM APIs.
Before running the scripts, set the corresponding environment variables.

Typical defaults used in the code are:

```bash
export DEEPSEEK_API_KEY=your_deepseek_key
export QWEN_API_KEY=your_qwen_key
export ARK_API_KEY=your_doubao_or_ark_key
```

On Windows PowerShell:

```powershell
$env:DEEPSEEK_API_KEY="your_deepseek_key"
$env:QWEN_API_KEY="your_qwen_key"
$env:ARK_API_KEY="your_doubao_or_ark_key"
```

### Default backend settings in the scripts

Examples of default host / model pairs used by the project:

- DeepSeek: `api.deepseek.com` + `deepseek-chat`
- Qwen: `dashscope.aliyuncs.com` + `qwen-flash`
- Doubao/ARK: `ark.cn-beijing.volces.com` + `doubao-seed-2-0-mini-260215`

These can usually be overridden through command-line arguments in the training and testing scripts.

---

## 6. Quick Start

### 6.1 Single-model replication

Run the single-model EoH baseline:

```bash
cd MLEOH_3llm/replication
python run_eoh_obp_repro.py
```

Or test a specific backend script directly, depending on your local version and setup.

---

### 6.2 Random router

```bash
cd MLEOH_3llm/random
python run_eoh_obp_random_router.py
```

---

### 6.3 Static rule router

```bash
cd MLEOH_3llm/static_rule
python run_eoh_obp_rule_router.py
```

---

### 6.4 Statistic router

Training:

```bash
cd MLEOH_3llm/statistic_llmrouter
python train_eoh_obp_llm_router.py
```

Testing:

```bash
python test_eoh_obp_llm_router.py --router_state path/to/router_state.json
```

> Note: depending on the exact local version, the saved state path and generated output folder names may vary.

---

### 6.5 Bandit router

Training:

```bash
cd MLEOH_3llm/bandit_learned
python train_eoh_obp_bandit_router.py
```

Testing:

```bash
python test_eoh_obp_bandit_router.py --bandit_state path/to/bandit_state.json
```

---

### 6.6 TextGrad router (delta version)

Training:

```bash
cd MLEOH_3llm/textgrad_router_delta
python train_eoh_obp_textgrad_router3.py
```

Testing:

```bash
python test_eoh_obp_textgrad_router3.py --router_state path/to/textgrad_router_state.json
```

---

## 7. Running the Full Experiment Pipeline

The project includes a unified runner:

```bash
cd MLEOH_3llm
python run_all_final_progress.py
```

This script is intended to automate a full comparison pipeline including:

- replication baselines,
- random router,
- static rule router,
- statistic router,
- bandit router,
- TextGrad router.

### Default settings used in this runner

Some important defaults in `run_all_final_progress.py` are:

- `seeds = 0..9`
- `train_instance_seed = 2026`
- `test_instance_seed = 2025`
- `pop_size = 10`
- `max_generations = 6`
- `main_max_sample_nums = 1000`
- `learn_train_max_sample_nums = 300`

These settings correspond to the main comparison setting used in the thesis experiments.

---

## 8. Running Ablation Experiments

For TextGrad-router ablation studies, use:

```bash
cd MLEOH_3llm
python run_ablation.py
```

This script is used to compare different TextGrad-related variants, such as removing or changing parts of the routing mechanism.

Typical factors explored include:

- profile mode,
- update schedule,
- reward shaping,
- exploration settings,
- policy length constraints,
- and other TextGrad-related design choices.

---

## 9. Output and Logs

The project writes experiment outputs to folders such as:

- `logs/...`
- `MLEOH_3llm/all_results/...`
- router-specific run directories

Typical saved artifacts include:

- `result.json`
- `summary.csv`
- `router_trace.csv`
- `best_program.txt`
- saved router state files such as `bandit_state.json` or `textgrad_router_state.json`
- per-run logs and population snapshots

These files are used for:

- result aggregation,
- train/test comparison,
- routing behavior analysis,
- best-program collection,
- and thesis figure generation.

---

## 10. Notes on Reproducibility

This project tries to keep comparisons fair by using:

- unified search budgets,
- fixed seed lists,
- explicit train/test instance seeds,
- and separate train/test phases for learnable routers.

However, exact numeric results may still vary because:

1. API-based LLM outputs are not perfectly deterministic,
2. backend model versions may change over time,
3. timeout behavior and remote service latency may differ,
4. different accounts or providers may have slightly different deployment settings.

So the main goal of reproduction is usually to recover the **overall trend** rather than bitwise-identical scores.

---

## 11. Important GitHub Upload Note

Before pushing this project to GitHub, please check the following carefully:

### 1) Remove all hardcoded API keys
Some local scripts may contain directly written API keys from the experimentation stage.
These **must be removed before uploading**.

Recommended practice:
- replace hardcoded keys with environment variables,
- rotate any key that has ever been written into the repository,
- double-check git history before making the repository public.

### 2) Clean large result folders if needed
Folders such as `all_results/`, `logs/`, and large intermediate artifacts may be too large or too noisy for a public repository.
If your goal is code release rather than raw-result archiving, consider uploading:

- the source code,
- one small example output,
- a clean `results/README` describing how full results were generated.

### 3) Add `.gitignore`
A typical `.gitignore` may include:

```gitignore
__pycache__/
*.pyc
.env
logs/
all_results/
*.ipynb_checkpoints
.DS_Store
```

---

## 12. Suggested Citation

If you use or reference this project, please cite:

```bibtex
@misc{chen2026mleoh,
  title={Collaborative Multi-LLM Heuristic Design in the Evolution of Heuristics (EoH) Framework},
  author={Yuda Chen},
  year={2026},
  note={Undergraduate thesis project based on LLM4AD}
}
```

If you use the original LLM4AD platform, please also cite the official LLM4AD paper and repository.

---

## 13. Acknowledgment

This project is built on top of **LLM4AD**.
I sincerely thank the LLM4AD authors and maintainers for providing the original platform, documentation, and examples.

This repository extension was implemented for my undergraduate thesis research on multi-LLM routing for automatic heuristic design.

---

## 14. Contact

If this repository is released publicly, you can add your preferred contact information here, for example:

- GitHub Issues
- Email
- Project page

For academic use, you may also state that this code corresponds to the thesis experiments in the OBP setting.
