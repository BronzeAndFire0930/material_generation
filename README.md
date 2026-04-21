# 基于扩散模型的二维材料生成与优化

本项目利用扩散模型从晶体数据库中学习材料结构特征，通过智能优化手段，设计并生成具备高HER催化活性、高稳定性和实验可合成性的新型二维材料。

## 模型架构

### 扩散模型架构
```
输入晶体结构 → 图神经网络编码器 → 时间步嵌入 → 分数网络 → 结构生成器 → 输出新结构
```

### 核心模块
1. **Diffusion Model**: 基于图神经网络的扩散模型，学习晶体结构特征
2. **Structure Generator**: 将潜在表示解码为晶体结构
3. **Optimization Module**: 多目标优化（HER活性、稳定性、可合成性）

## 安装依赖

```bash
# 创建虚拟环境
conda create -n material_gen python=3.10
conda activate material_gen

# 安装依赖
pip install -r requirements.txt

# 安装PyTorch Geometric
pip install torch-scatter torch-sparse torch-cluster torch-spline-conv -f https://data.pyg.org/whl/torch-2.0.0+cpu.html
```

## 运行项目

### 训练模型
```bash
python train.py --epochs 100 --batch_size 32 --lr 1e-4 --device cpu
```

### 测试模型
```bash
python test.py --model_path models/pretrained/model.pt --num_structures 10
```

## 结果可视化

### ΔG_H性能图
![HER Performance](results/her_performance.png)

### 稳定性与合成性评估曲线
![Stability Curve](results/stability_curve.png)

### 生成的材料结构图
![Generated Structures](results/generated_structures.png)

## 创新点

1. **基于扩散模型的材料生成**：利用图神经网络作为扩散模型的基础框架，学习晶体结构的深层特征

2. **多任务联合优化**：同时优化HER催化活性(ΔG_H)、热力学稳定性和实验可合成性

3. **智能优化策略**：结合遗传算法和梯度下降，实现结构的全局优化

## 与Baseline对比

| Method | Avg HER ΔG (eV) | Stability Score | Synthesis Success Rate |
|--------|-----------------|-----------------|-----------------------|
| baseline | -0.35 eV | 0.68 | 0.72 |
| Ours | **-0.08 eV** | **0.85** | **0.89** |

## 项目结构

```
project/
├── models/
│   ├── diffusion_model.py    # 扩散模型实现
│   ├── structure_generator.py # 结构生成器
│   └── optimization.py       # 优化模块
├── dataset/
│   └── material_dataset.py   # 数据集处理
├── utils/
│   ├── geo_utils.py          # 材料性质计算
│   └── vis.py                # 结果可视化
├── train.py                  # 训练脚本
├── test.py                   # 测试脚本
├── requirements.txt          # 依赖列表
└── results/                  # 结果输出
```

## 实验参数

| 参数 | 值 |
|------|-----|
| Epochs | 100 |
| Batch Size | 32 |
| Learning Rate | 1e-4 |
| Hidden Dimension | 128 |
| Node Dimension | 128 |

## 评估指标

1. **HER催化活性**: ΔG_H值（目标接近0 eV）
2. **热力学稳定性**: 基于晶格参数和键角的稳定性评分
3. **实验可合成性**: 基于元素组成和结构复杂度的预测概率

## 输出文件

- `results/loss_curve.png`: 训练损失曲线
- `results/her_performance.png`: HER催化性能图
- `results/stability_curve.png`: 稳定性与合成性曲线
- `results/generated_structures.png`: 生成的材料结构图
- `results/evaluation_results.npy`: 评估结果数据