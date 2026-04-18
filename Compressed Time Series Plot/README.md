# Compressed Time Series Plot

复刻“时间轴截断/压缩”的双子图示例。核心点是：

- 横轴不是连续时间，而是多个时间片段拼接后形成的“压缩轴”
- 片段之间显示断裂符号（`//`）
- 每个片段仍保留原始时间刻度标签，读图时不丢失真实时间语义

## 目录结构

- `src/compressed_axis.py`: 时间压缩映射模块（低耦合，可复用）
- `src/replicate_compressed_plot.py`: 主绘图脚本（示例数据 + 图形元素）
- `output/`: 输出图目录

## 安装

```bash
pip install -r requirements.txt
```

## 运行

```bash
python src/replicate_compressed_plot.py --output output/compressed_time_series_demo.png
```

## 替换成你的真实数据

你只需要：

1. 保留 `TimeAxisCompressor` 的 `segments`
2. 用你的时间序列替换脚本里的 `build_demo_series`
3. 调用 `plot_series_with_compressed_axis(...)`

这样可以最大限度保持绘图逻辑和数据来源解耦。
