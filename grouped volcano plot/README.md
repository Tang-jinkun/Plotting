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
