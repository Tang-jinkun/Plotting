# Grouped Volcano Plot (Python)

This project reproduces a grouped volcano-style plot with:

- grouped jitter points
- bubble size mapped to `-log10(p_adj)`
- colored group band in the center
- up/down counts per group
- auto labels for top significant genes

## 1) Install dependencies

Use your `plotting` environment:

```powershell
conda activate plotting
pip install -r requirements.txt
```

## 2) Data file (decoupled)

Input data is independent from code and stored in CSV.

Required columns:

- `group`: group name (e.g. `group1` ... `group6`)
- `gene`: gene symbol for optional labels
- `log2_fc`: log2 fold-change value
- `p_adj`: adjusted p-value (0 to 1)

## 3) Generate demo data

```powershell
python src/generate_demo_data.py --output data/volcano_data.csv --points-per-group 350 --seed 42
```

## 4) Draw grouped volcano plot

```powershell
python src/plot_grouped_volcano.py --input data/volcano_data.csv --output output/grouped_volcano_plot.png
```

Optional parameters:

- `--fc-threshold` (default `1.0`)
- `--padj-threshold` (default `0.05`)
- `--labels-per-side` (default `2`)
- `--dpi` (default `300`)

## 5) Trust score volcano-style plot (new)

This mode maps information as:

- X axis: grouped time windows (each group has its own `sample_time` range 0-300)
- Y axis: `trust_score`
- Point size: fixed (`s=20` by default)
- No x-axis jitter is applied (points stay at original sampled times)
- Optional horizontal threshold lines: `theta0`, `thetaT`

### Data format (decoupled)

Required columns:

- `group`
- `sample_time`
- `trust_score`

Optional column:

- `sample_id`

Real source mapping used for your data:

- `Train Legit` -> `theta_train_sampled.mat`
- `Test Legit` -> `theta_pool_legal`
- `Attacker 1` -> `theta_pool_attack.mat`
- `Attacker 2` -> keep the existing placeholder behavior unchanged for now

The first three groups are treated as 300 sampled points over `0` to `300`.

### Generate demo trust-score data

```powershell
python src/generate_trust_score_data.py --output data/trust_score_data.csv --seed 42
```

### Generate trust-score data from real MAT files

Put `theta_train_sampled.mat`, `theta_pool_legal.mat`, and `theta_pool_attack.mat` in one folder, then run:

```powershell
python src/generate_trust_score_data.py --mode real --mat-dir data --output data/trust_score_data.csv
```

The loader expects one numeric vector per file with exactly 300 samples. `Attacker 2` still uses the existing placeholder series because you said that data is not available yet.

### Draw trust-score volcano-style plot

```powershell
python src/plot_trust_volcano_style.py --input data/trust_score_data.csv --output output/trust_score_volcano_style.png --theta0 0 --thetaT 1
```

Optional switches:

- `--hide-thresholds` to hide threshold lines
- `--hide-counts` to hide per-group High/Low counts
