# TfeNet 气道分割推理指南 / Airway Segmentation Inference Guide

[中文](#中文指南) | [English](#english-guide)

---

## 中文指南

本指南将帮助您使用训练好的 TfeNet 模型在自己的 DICOM 格式 CT 扫描数据上进行气道分割。

### 快速开始

#### 1. 环境准备

首先，确保已安装所需依赖：

```bash
# 创建并激活 conda 环境
conda create --name TfeNet python==3.8
conda activate TfeNet

# 安装 PyTorch (CUDA 12.1)
pip install torch==2.4.1 torchvision==0.19.1 torchaudio==2.4.1 --index-url https://download.pytorch.org/whl/cu121

# 安装其他依赖
pip install -r requirements.txt

# 安装 pydicom (用于 DICOM 转换)
pip install pydicom
```

#### 2. 安装方向感知卷积模块

```bash
# 进入 DAConv 文件夹并安装
cd DAConv
python setup.py install
cd ..

# 进入 DSConv 文件夹并安装
cd DSConv
python setup.py install
cd ..
```

#### 3. 下载预训练模型

从 [Google Drive](https://drive.google.com/file/d/1DEKyAMhV90AL80qoy2QgDxmO0F1cXx1r/view?usp=drive_link) 下载预训练模型，并解压到 `./checkpoint` 目录：

```
checkpoint/
├── AIIB23
│   ├── TfeNet_checkpoint.ckpt
│   └── TfeNetSmall_checkpoint.ckpt
├── ATM22
│   ├── TfeNet_checkpoint.ckpt
│   └── TfeNetSmall_checkpoint.ckpt
└── BAS
    ├── TfeNet_checkpoint.ckpt
    └── TfeNetSmall_checkpoint.ckpt
```

### 完整推理流程

#### 步骤 1: 将 DICOM 转换为 NIfTI 格式

TfeNet 需要 NIfTI 格式 (.nii.gz) 的输入数据。使用提供的脚本转换您的 DICOM 数据：

```bash
# 单个 DICOM 序列（一个文件夹包含一个患者的所有 DICOM 文件）
python dicom_to_nifti.py --input /path/to/dicom/folder --output /path/to/nifti/output

# 多个 DICOM 序列（父文件夹包含多个患者文件夹）
python dicom_to_nifti.py --input /path/to/parent/folder --output /path/to/nifti/output

# 递归处理所有子文件夹
python dicom_to_nifti.py --input /path/to/parent/folder --output /path/to/nifti/output --recursive
```

**示例：**
```bash
# 假设您的数据结构如下：
# my_data/
# ├── patient001/
# │   ├── IM0001.dcm
# │   ├── IM0002.dcm
# │   └── ...
# ├── patient002/
# │   ├── IM0001.dcm
# │   └── ...

python dicom_to_nifti.py --input ./my_data --output ./nifti_data
```

转换后，您将在 `./nifti_data` 中得到：
```
nifti_data/
├── patient001.nii.gz
├── patient002.nii.gz
└── ...
```

#### 步骤 2: 运行气道分割

使用 `inference.py` 脚本在您的 NIfTI 数据上进行推理：

```bash
# 基本用法（仅使用主模型）
python inference.py --input ./nifti_data --output ./segmentation_results

# 使用小气道模型获得更好的细节分割
python inference.py --input ./nifti_data --output ./segmentation_results --use-small

# 指定不同的检查点数据集
python inference.py --input ./nifti_data --output ./segmentation_results --dataset BAS

# 在 CPU 上运行（如果没有 GPU）
python inference.py --input ./nifti_data --output ./segmentation_results --device cpu
```

**主要参数说明：**

- `--input` (`-i`): NIfTI 文件所在文件夹路径
- `--output` (`-o`): 分割结果保存路径
- `--checkpoint` (`-c`): 检查点文件夹路径（默认：`./checkpoint`）
- `--dataset` (`-d`): 使用的数据集检查点，可选 `ATM22`、`BAS`、`AIIB23`（默认：`ATM22`）
- `--use-small`: 同时使用小气道模型并合并结果（推荐用于更精细的分割）
- `--no-postprocess`: 跳过后处理（提取最大连通分量）
- `--device`: 运行设备，`cuda` 或 `cpu`（默认：`cuda`）
- `--cube-size`: 滑动窗口立方体大小（默认：128）
- `--step`: 滑动窗口步长（默认：64）

#### 步骤 3: 查看结果

分割结果将保存为 NIfTI 格式 (.nii.gz)，可以使用以下软件查看：

- **ITK-SNAP**: https://www.itksnap.org/
- **3D Slicer**: https://www.slicer.org/
- **MITK Workbench**: https://www.mitk.org/

结果文件结构：
```
segmentation_results/
├── patient001.nii.gz  # 最终分割结果
├── patient002.nii.gz
├── temp_pred/         # 临时预测（主模型）
└── temp_small/        # 临时预测（小气道模型，如果使用了 --use-small）
```

### 一键运行示例脚本

我们提供了一个完整的示例脚本，演示从 DICOM 到最终分割的完整流程：

```bash
bash run_inference_example.sh
```

或者手动运行每一步：

```bash
# 1. 转换 DICOM 到 NIfTI
python dicom_to_nifti.py \
    --input /path/to/your/dicom/data \
    --output ./converted_nifti

# 2. 运行气道分割
python inference.py \
    --input ./converted_nifti \
    --output ./results \
    --dataset ATM22 \
    --use-small
```

### 常见问题

**Q1: 我的数据是什么格式？**

A: 本工具支持 DICOM 格式的 CT 扫描。DICOM 文件通常以 `.dcm` 结尾。如果您的数据已经是 NIfTI 格式 (`.nii` 或 `.nii.gz`)，可以直接跳到步骤 2。

**Q2: 转换 DICOM 时出错怎么办？**

A: 尝试使用 `--manual` 参数使用手动加载方法：
```bash
python dicom_to_nifti.py --input /path/to/dicom --output /path/to/output --manual
```

**Q3: 推理时显示 CUDA out of memory 怎么办？**

A: 可以尝试：
1. 减小 cube-size: `--cube-size 96`
2. 使用 CPU: `--device cpu`
3. 一次只处理一个病例

**Q4: 应该选择哪个数据集的检查点？**

A: 
- **ATM22**: 平衡准确性和连续性，适合大多数情况（推荐）
- **BAS**: 在 BAS 数据集上训练，准确性和连续性都很好
- **AIIB23**: 更好的泄漏控制和精度

**Q5: 是否需要使用小气道模型？**

A: 使用 `--use-small` 可以更好地分割细小气道结构，但会增加约 2 倍的计算时间。推荐用于需要高精度细节的场景。

### 技术细节

#### 数据预处理
- CT 图像会自动应用 Hounsfield Unit (HU) 窗口 [-1000, 600]
- 图像会被归一化到 [0, 1] 范围
- 使用滑动窗口方法处理整个 3D 体积

#### 后处理
- 默认会提取最大连通分量，去除小的孤立分割
- 可以使用 `--no-postprocess` 跳过此步骤

#### 性能
- 使用 GPU (CUDA): 约 2-5 分钟每个病例
- 使用 CPU: 约 20-60 分钟每个病例（取决于图像大小）

---

## English Guide

This guide will help you use the trained TfeNet model to perform airway segmentation on your own DICOM format CT scan data.

### Quick Start

#### 1. Environment Setup

First, make sure you have installed the required dependencies:

```bash
# Create and activate conda environment
conda create --name TfeNet python==3.8
conda activate TfeNet

# Install PyTorch (CUDA 12.1)
pip install torch==2.4.1 torchvision==0.19.1 torchaudio==2.4.1 --index-url https://download.pytorch.org/whl/cu121

# Install other dependencies
pip install -r requirements.txt

# Install pydicom (for DICOM conversion)
pip install pydicom
```

#### 2. Install Direction-Aware Convolution Modules

```bash
# Enter DAConv folder and install
cd DAConv
python setup.py install
cd ..

# Enter DSConv folder and install
cd DSConv
python setup.py install
cd ..
```

#### 3. Download Pre-trained Models

Download the pre-trained models from [Google Drive](https://drive.google.com/file/d/1DEKyAMhV90AL80qoy2QgDxmO0F1cXx1r/view?usp=drive_link) and extract to the `./checkpoint` directory:

```
checkpoint/
├── AIIB23
│   ├── TfeNet_checkpoint.ckpt
│   └── TfeNetSmall_checkpoint.ckpt
├── ATM22
│   ├── TfeNet_checkpoint.ckpt
│   └── TfeNetSmall_checkpoint.ckpt
└── BAS
    ├── TfeNet_checkpoint.ckpt
    └── TfeNetSmall_checkpoint.ckpt
```

### Complete Inference Pipeline

#### Step 1: Convert DICOM to NIfTI Format

TfeNet requires NIfTI format (.nii.gz) input data. Use the provided script to convert your DICOM data:

```bash
# Single DICOM series (one folder containing all DICOM files for one patient)
python dicom_to_nifti.py --input /path/to/dicom/folder --output /path/to/nifti/output

# Multiple DICOM series (parent folder containing multiple patient folders)
python dicom_to_nifti.py --input /path/to/parent/folder --output /path/to/nifti/output

# Recursively process all subfolders
python dicom_to_nifti.py --input /path/to/parent/folder --output /path/to/nifti/output --recursive
```

**Example:**
```bash
# Assuming your data structure is:
# my_data/
# ├── patient001/
# │   ├── IM0001.dcm
# │   ├── IM0002.dcm
# │   └── ...
# ├── patient002/
# │   ├── IM0001.dcm
# │   └── ...

python dicom_to_nifti.py --input ./my_data --output ./nifti_data
```

After conversion, you will have in `./nifti_data`:
```
nifti_data/
├── patient001.nii.gz
├── patient002.nii.gz
└── ...
```

#### Step 2: Run Airway Segmentation

Use the `inference.py` script to perform inference on your NIfTI data:

```bash
# Basic usage (main model only)
python inference.py --input ./nifti_data --output ./segmentation_results

# Use small airway model for better fine structure segmentation
python inference.py --input ./nifti_data --output ./segmentation_results --use-small

# Specify different checkpoint dataset
python inference.py --input ./nifti_data --output ./segmentation_results --dataset BAS

# Run on CPU (if no GPU available)
python inference.py --input ./nifti_data --output ./segmentation_results --device cpu
```

**Main Parameters:**

- `--input` (`-i`): Path to folder containing NIfTI files
- `--output` (`-o`): Path to save segmentation results
- `--checkpoint` (`-c`): Path to checkpoint folder (default: `./checkpoint`)
- `--dataset` (`-d`): Dataset checkpoint to use, choices: `ATM22`, `BAS`, `AIIB23` (default: `ATM22`)
- `--use-small`: Also use small airway model and combine results (recommended for finer segmentation)
- `--no-postprocess`: Skip postprocessing (largest component extraction)
- `--device`: Device to run on, `cuda` or `cpu` (default: `cuda`)
- `--cube-size`: Sliding window cube size (default: 128)
- `--step`: Sliding window step size (default: 64)

#### Step 3: View Results

The segmentation results will be saved in NIfTI format (.nii.gz) and can be viewed using:

- **ITK-SNAP**: https://www.itksnap.org/
- **3D Slicer**: https://www.slicer.org/
- **MITK Workbench**: https://www.mitk.org/

Result file structure:
```
segmentation_results/
├── patient001.nii.gz  # Final segmentation result
├── patient002.nii.gz
├── temp_pred/         # Temporary predictions (main model)
└── temp_small/        # Temporary predictions (small airway model, if --use-small used)
```

### One-Click Example Script

We provide a complete example script demonstrating the full pipeline from DICOM to final segmentation:

```bash
bash run_inference_example.sh
```

Or run each step manually:

```bash
# 1. Convert DICOM to NIfTI
python dicom_to_nifti.py \
    --input /path/to/your/dicom/data \
    --output ./converted_nifti

# 2. Run airway segmentation
python inference.py \
    --input ./converted_nifti \
    --output ./results \
    --dataset ATM22 \
    --use-small
```

### FAQ

**Q1: What format should my data be in?**

A: This tool supports DICOM format CT scans. DICOM files typically end with `.dcm`. If your data is already in NIfTI format (`.nii` or `.nii.gz`), you can skip directly to Step 2.

**Q2: What if DICOM conversion fails?**

A: Try using the `--manual` parameter to use the manual loading method:
```bash
python dicom_to_nifti.py --input /path/to/dicom --output /path/to/output --manual
```

**Q3: What if I get "CUDA out of memory" during inference?**

A: You can try:
1. Reduce cube-size: `--cube-size 96`
2. Use CPU: `--device cpu`
3. Process one case at a time

**Q4: Which dataset checkpoint should I choose?**

A: 
- **ATM22**: Balances accuracy and continuity, suitable for most cases (recommended)
- **BAS**: Trained on BAS dataset, good accuracy and continuity
- **AIIB23**: Better leakage control and precision

**Q5: Do I need to use the small airway model?**

A: Using `--use-small` provides better segmentation of small airway structures but increases computation time by about 2x. Recommended for scenarios requiring high-precision details.

### Technical Details

#### Data Preprocessing
- CT images are automatically windowed with Hounsfield Units (HU) range [-1000, 600]
- Images are normalized to [0, 1] range
- Sliding window approach is used to process the entire 3D volume

#### Postprocessing
- By default, extracts the largest connected component to remove small isolated segments
- Can be skipped with `--no-postprocess`

#### Performance
- Using GPU (CUDA): ~2-5 minutes per case
- Using CPU: ~20-60 minutes per case (depending on image size)

---

## Citation

If you use this code or pre-trained models in your research, please cite:

```latex
@article{WU2026103882,
title = {Direction-Aware convolution for airway tubular feature enhancement network},
journal = {Medical Image Analysis},
volume = {108},
pages = {103882},
year = {2026},
issn = {1361-8415},
doi = {https://doi.org/10.1016/j.media.2025.103882},
url = {https://www.sciencedirect.com/science/article/pii/S1361841525004281},
author = {Qibiao Wu and Yagang Wang and Qian Zhang},
}
```
