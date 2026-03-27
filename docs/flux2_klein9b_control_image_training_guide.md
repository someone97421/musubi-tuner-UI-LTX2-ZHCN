# FLUX.2-klein-9B 参考图对照训练详细教程

> 本教程介绍如何使用 `flux2_ui_zh.py` 界面进行 FLUX.2-klein-9B 的参考图（Control Image）对照训练

## 📋 目录

1. [概述](#概述)
2. [环境准备](#环境准备)
3. [模型下载](#模型下载)
4. [数据集准备](#数据集准备)
5. [启动 UI 并配置](#启动-ui-并配置)
6. [预缓存](#预缓存)
7. [训练参数设置](#训练参数设置)
8. [开始训练](#开始训练)
9. [推理测试](#推理测试)
10. [常见问题](#常见问题)

---

## 概述

FLUX.2-klein-9B 是 Black Forest Labs 发布的图像生成模型，支持参考图输入（Image-to-Image/Control Image）。本教程讲解如何训练 LoRA 来实现：

- 🎨 **风格迁移**：学习特定艺术风格并应用到新图像
- 👤 **角色保持**：训练特定角色在不同姿势/场景下的生成
- 🖼️ **图像编辑**：学习特定的图像变换/编辑模式

### 为什么选择 klein-base-9b 进行训练？

| 模型版本 | 类型 | 训练建议 |
|---------|------|---------|
| `dev` | 蒸馏模型 | 仅推理，不推荐训练 |
| `klein-4b` | 蒸馏模型 | 仅推理，不推荐训练 |
| `klein-9b` | 蒸馏模型 | 仅推理，不推荐训练 |
| `klein-base-4b` | 基础模型 | ✅ 推荐训练 |
| `klein-base-9b` | 基础模型 | ✅ **最佳训练选择** |

基础模型（base）比蒸馏模型更适合训练，9B 比 4B 容量更大，表达能力更强。

---

## 环境准备

### 1. 确保已安装 Musubi-Tuner

```bash
# 进入项目目录
cd d:\LTXtrainUI

# 检查 Python 环境
python --version  # 建议 3.10+

# 检查依赖
pip list | findstr gradio
pip list | findstr torch
```

### 2. 启动 UI

```bash
# 方式1：直接运行
python flux2_ui_zh.py

# 方式2：使用启动脚本（Windows）
00---run_flux2_ui.bat
```

浏览器会自动打开 `http://127.0.0.1:7861`

---

## 模型下载

### 需要下载的文件

| 组件 | 文件名 | 来源 |
|------|--------|------|
| DiT | `flux2-klein-base-9b.safetensors` | [black-forest-labs/FLUX.2-klein-base-9B](https://huggingface.co/black-forest-labs/FLUX.2-klein-base-9B) |
| VAE | `ae.safetensors` | [black-forest-labs/FLUX.2-dev](https://huggingface.co/black-forest-labs/FLUX.2-dev) |
| Text Encoder | `00001-of-00004.safetensors` (及同组其他文件) | [black-forest-labs/FLUX.2-klein-9B](https://huggingface.co/black-forest-labs/FLUX.2-klein-9B) |

> **注意**：Text Encoder 需要下载所有分割文件（00001-of-00004 到 00004-of-00004），但配置时只需指定第一个文件即可。

### 下载后目录结构示例

```
D:\Models\FLUX2\
├── dit\
│   └── flux2-klein-base-9b.safetensors
├── vae\
│   └── ae.safetensors
└── text_encoder\
    ├── 00001-of-00004.safetensors
    ├── 00002-of-00004.safetensors
    ├── 00003-of-00004.safetensors
    └── 00004-of-00004.safetensors
```

---

## 数据集准备

对照训练需要准备**三要素**：
1. **目标图像**（Target）：希望生成的结果图像
2. **参考图像**（Control）：输入给模型的参考/条件图像
3. **文本描述**（Caption）：描述目标内容的文本

### 1. 文件夹结构

```
D:\TrainingData\MyStyle\
├── images\              # 目标图像（生成结果）
│   ├── portrait_01.jpg
│   ├── portrait_02.jpg
│   └── portrait_03.jpg
├── control_images\      # 参考图像（输入条件）
│   ├── portrait_01.png  # 与目标图像同名
│   ├── portrait_02.png
│   └── portrait_03.png
└── captions\            # 文本描述（可选，也可放在images旁）
    ├── portrait_01.txt
    ├── portrait_02.txt
    └── portrait_03.txt
```

### 2. 文件命名规则

**基本规则**：目标图像和参考图像需要**同名**（扩展名可不同）

```
目标: images\photo_01.jpg    →   参考: control_images\photo_01.png
目标: images\photo_02.jpg    →   参考: control_images\photo_02.png
```

**多张参考图**（可选）：如需多参考图，使用编号后缀

```
目标: images\scene_01.jpg
参考1: control_images\scene_01_0.png
参考2: control_images\scene_01_1.png
参考3: control_images\scene_01_2.png
```

### 3. 图像要求

| 项目 | 建议 | 说明 |
|------|------|------|
| 目标图像分辨率 | 1024×1024 | 正方形最佳 |
| 参考图像分辨率 | 2024×2024 | FLUX.2 官方推荐（单张参考图时） |
| 格式 | PNG/JPG | PNG 保留更多细节 |
| 色彩 | sRGB | 避免颜色空间问题 |

### 4. 文本描述（Caption）编写

每个目标图像需要一个对应的 `.txt` 文件：

```
# images\portrait_01.txt
masterpiece, best quality, portrait of a young woman, 
soft lighting, studio background, looking at viewer

# images\portrait_02.txt
masterpiece, best quality, portrait of a young woman, 
profile view, natural lighting, outdoor garden
```

**Caption 编写建议**：
- 描述**目标图像**的内容（不是参考图）
- 包含通用质量词（masterpiece, best quality）
- 描述具体的风格、姿势、场景等变化
- 保持角色/主体描述一致（如训练特定角色）

### 5. 训练场景示例

#### 场景 A：风格迁移训练
```
参考图：普通照片/线稿
目标图：特定风格化图像（如油画、动漫）
Caption：描述目标风格和内容
```

#### 场景 B：角色保持训练（I2I）
```
参考图：角色草图/姿势骨架
目标图：完成的高质量角色图
Caption：描述角色外貌、服装、场景
```

#### 场景 C：图像编辑训练
```
参考图：原始照片
目标图：编辑后的结果（如改变季节、添加特效）
Caption：描述编辑后的内容和效果
```

---

## 启动 UI 并配置

### 1. 全局路径设置

在 UI 顶部「📂 全局路径设置」区域填写：

```
★ DiT 模型路径 (--dit，必填): 
    D:\Models\FLUX2\dit\flux2-klein-base-9b.safetensors

模型版本 (--model_version): 
    klein-base-9b  ⬅️ 重要！选择正确版本

★ VAE 路径 (--vae，必填): 
    D:\Models\FLUX2\vae\ae.safetensors

★ Text Encoder 路径 (--text_encoder，必填): 
    D:\Models\FLUX2\text_encoder\00001-of-00004.safetensors

数据集配置文件 (--dataset_config): 
    dataset_flux2_klein9b.toml

输出目录 (--output_dir): 
    ./outputs/flux2_klein9b

输出模型名称 (--output_name): 
    mystyle_flux2_9b

训练配置文件路径（保存/读取）: 
    flux2_klein9b_train_config.toml
```

---

## 预缓存

### 1. 生成数据集配置 TOML

切换到「🗂️ 1. 数据集准备」标签页：

**1.1 填写数据集参数**

```
Resolution Width: 1024
Resolution Height: 1024
Caption Extension: .txt
Batch Size: 1
Enable Bucket: ☑️ 勾选
Bucket No Upscale: ☐ 不勾选

Image Directory (图像文件夹): D:\TrainingData\MyStyle\images
Cache Directory (缓存文件夹): D:\TrainingData\MyStyle\cache
Control Images Dir (参考图像文件夹，可选): D:\TrainingData\MyStyle\control_images

保存路径: dataset_flux2_klein9b.toml
```

**1.2 点击「📄 生成数据集配置」按钮**

成功后状态栏显示：
```
✅ 数据集配置已成功保存至 dataset_flux2_klein9b.toml
```

### 2. 缓存 Latents

在「🗂️ 1.2 预缓存 Latent 与文本特征」区域：

**Cache Latents 设置**：
```
VAE Dtype: bfloat16    ⬅️ 可减少显存，默认float32也OK
跳过已缓存文件: ☑️ 勾选
```

点击 **「▶ 运行 Cache Latents」** 按钮

> ⏱️ 根据数据集大小，可能需要几分钟到几十分钟

### 3. 缓存 Text Encoder

**Cache Text Encoder 设置**：
```
Text Encoder Dtype: bf16
跳过已缓存文件: ☑️ 勾选
FP8 Text Encoder: ☐ 不勾选  ⬅️ klein-base-9b 使用 Qwen3-8B，不支持FP8
```

点击 **「▶ 运行 Cache Text Encoder」** 按钮

> ⚠️ 注意：`--fp8_text_encoder` 选项对 dev (Mistral 3) 不支持，但 klein-9b 使用 Qwen3-8B，需要测试是否支持

---

## 训练参数设置

切换到「⚙️ 2. 训练参数配置」标签页：

### 1. 模型与 LoRA 设置

```
网络模块 (--network_module): networks.lora_flux_2  ⬅️ 固定值，不要改
Network Dim / Rank: 32  ⬅️ 32-64 适合风格/角色
Network Alpha: 16       ⬅️ 通常设为 Dim 的一半
Network Dropout: 0      ⬅️ 0-0.1，0表示不使用
Scale Weight Norms: 0

续训 LoRA 权重路径: (空，从头训练)
从权重读取 Dim: ☐
续训状态路径: (空)

启用 LoRA+: ☑️ 勾选
LoRA+ LR Ratio: 4       ⬅️ 4-16，4是保守值
启用块控制: ☐ 不勾选   ⬅️ 高级功能，一般不需要
```

### 2. 训练时长与批次

```
最大 Epoch 数: 16       ⬅️ 根据数据量调整，单图训练可设高些
最大训练步数: (空)     ⬅️ 优先使用 Epoch
批次大小: 1            ⬅️ 显存受限保持1
随机种子: 42

梯度检查点: ☑️ 勾选     ⬅️ 必须勾选，节省显存
梯度累积步数: 1
Caption Dropout Rate: 0.0  ⬅️ 0-0.1，对照训练建议0
```

### 3. 精度与内存优化（关键！）

klein-base-9b 模型较大，需要启用内存优化：

```
混合精度: bf16           ⬅️ FLUX.2 推荐 bf16
VAE Dtype: (空)         ⬅️ 使用默认

FP8 Base (--fp8_base): ☑️ 勾选    ⬅️ DiT 使用 FP8
FP8 Scaled (--fp8_scaled): ☑️ 勾选 ⬅️ 必须同时启用
FP8 Text Encoder: ☐ 不勾选        ⬅️ Qwen3-8B 不支持

注意力机制: sdpa        ⬅️ 或 flash_attn/xformers（如已安装）
Blocks to Swap: 16      ⬅️ klein-9b 最大支持16，建议拉满
Pinned Memory: ☑️ 勾选
梯度检查点 CPU Offload: ☐ 不勾选  ⬅️ 如仍爆显存可启用
```

> 💡 **内存优化组合**：`fp8_base + fp8_scaled + blocks_to_swap=16` 是 klein-9b 训练的黄金组合

### 4. 学习率设置

```
学习率: 1e-4            ⬅️ 1e-4 到 5e-4 之间
调度器: constant_with_warmup
预热步数: 50
衰减步数: 0.2
余弦重启次数: 1
```

### 5. 时间步采样（关键！）

```
时间步采样方式: flux2_shift   ⬅️ FLUX.2 专用，必须选这个
加权方案: none               ⬅️ 可选 sigma_sqrt/logit_normal
Logit Mean: 0.0
Logit Std: 1.0
Mode Scale: 1.29
Min Timestep: 0
Max Timestep: 1000
```

### 6. 优化器

```
优化器类型: AdamW8bit    ⬅️ 或 PagedAdamW8bit
Max Grad Norm: 1.0
```

### 7. 保存设置

```
每N Epoch保存: 1         ⬅️ 每轮都保存，方便测试
每N步保存: (空)         ⬅️ 使用 Epoch 模式
只保留最后N轮: (空)     ⬅️ 空表示保留全部
只保留最后N步: (空)

保存训练状态: ☐ 不勾选   ⬅️ 除非需要中断续训
仅结束时保存状态: ☐

日志监控: tensorboard
日志路径: ./logs
```

### 8. 数据加载

```
数据加载线程数: 8
持久化工人: ☑️ 勾选
cuda_allow_tf32: ☑️ 勾选
cuda_cudnn_benchmark: ☑️ 勾选
```

### 9. 保存配置

点击底部 **「💾 保存配置」** 按钮

状态栏显示：
```
✅ 保存成功！→ flux2_klein9b_train_config.toml
```

---

## 开始训练

切换到「🚀 3. 训练及进度显示」标签页

### 1. 确认配置

在点击启动前，确认以下配置已正确保存到 TOML 文件：

```toml
# 关键配置检查清单
dit = "D:\\Models\\FLUX2\\dit\\flux2-klein-base-9b.safetensors"
model_version = "klein-base-9b"      # ⭐ 确认版本正确
vae = "D:\\Models\\FLUX2\\vae\\ae.safetensors"
text_encoder = "D:\\Models\\FLUX2\\text_encoder\\00001-of-00004.safetensors"
network_module = "networks.lora_flux_2"
fp8_base = true
fp8_scaled = true
blocks_to_swap = 16
timestep_sampling = "flux2_shift"
```

### 2. 启动训练

点击 **「🚀 一键启动 FLUX.2 训练」** 按钮

### 3. 监控训练

- **UI 日志窗口**：实时显示训练输出
- **TensorBoard**：如启用，访问 `http://127.0.0.1:6006` 查看学习曲线
- **模型输出**：每轮保存在 `./outputs/flux2_klein9b/` 目录

### 4. 训练完成

训练完成后，你会得到类似这样的文件：

```
./outputs/flux2_klein9b/
├── mystyle_flux2_9b.safetensors      # 最终模型
├── mystyle_flux2_9b-000001.safetensors  # 第1轮
├── mystyle_flux2_9b-000002.safetensors  # 第2轮
...
└── logs/                              # TensorBoard 日志
```

---

## 推理测试

### 1. 使用 generate_image 脚本

```bash
python src/musubi_tuner/flux_2_generate_image.py \
    --model_version klein-base-9b \
    --dit path/to/flux2-klein-base-9b.safetensors \
    --vae path/to/ae.safetensors \
    --text_encoder path/to/00001-of-00004.safetensors \
    --control_image_path path/to/your/test_control_image.jpg \
    --prompt "your trained style description" \
    --image_size 1024 1024 \
    --infer_steps 50 \
    --fp8_scaled \
    --save_path ./test_outputs \
    --output_type images \
    --seed 1234 \
    --lora_multiplier 1.0 \
    --lora_weight ./outputs/flux2_klein9b/mystyle_flux2_9b.safetensors
```

### 2. 关键推理参数说明

| 参数 | 说明 |
|------|------|
| `--model_version klein-base-9b` | 必须匹配训练时的版本 |
| `--control_image_path` | 测试用的参考图像路径 |
| `--lora_weight` | 训练好的 LoRA 路径 |
| `--lora_multiplier` | LoRA 强度，通常 0.8-1.2 |
| `--fp8_scaled` | 如训练时使用了 fp8，推理也建议启用 |

### 3. 对照训练的特殊效果

训练完成后，模型将学会：
- 根据参考图的**构图/姿势**生成新图像
- 应用训练时学习的**风格/特征**
- 结合文本描述的**内容指导**

---

## 常见问题

### Q1: 显存不足（OOM）怎么办？

**解决步骤**：
1. 确保 `fp8_base` 和 `fp8_scaled` 已启用
2. 将 `blocks_to_swap` 提高到 16（klein-9b 最大值）
3. 启用 `梯度检查点 CPU Offload`
4. 减小批次大小到 1
5. 如仍不足，考虑使用 klein-base-4b 或减少图像分辨率

### Q2: 训练效果不明显？

**检查事项**：
1. 确认 `--model_version` 设置为 `klein-base-9b` 而非 `klein-9b`
2. 检查数据集：目标图和参考图是否正确配对
3. 增加训练轮数（Epochs）
4. 尝试调整学习率（提高到 2e-4 或 5e-4）
5. 增加 LoRA Dim（提高到 64 或 128）

### Q3: 如何训练特定角色？

**数据准备建议**：
- 准备 20-50 张同一角色的不同姿势图像
- 参考图使用骨架/线稿，目标图使用完成的角色图
- Caption 中统一使用角色名（如 `sks character`）
- 训练更多轮数（20-50 Epochs）

### Q4: 参考图分辨率必须 2024×2024 吗？

**不一定**：
- 官方推荐单张参考图用 2024×2024
- 但实际训练可以用其他分辨率
- 可在数据集 TOML 中设置 `control_resolution = [1024, 1024]`
- 或使用 `no_resize_control = true` 保持原图尺寸

### Q5: 可以训练没有参考图的普通 LoRA 吗？

**可以**：
- 不填写 `Control Images Dir` 即可
- FLUX.2 支持纯文本到图像的 LoRA 训练
- 但这不是 FLUX.2 的主要优势，建议利用其参考图能力

---

## 附录：推荐配置速查表

### 单图训练（1张图过拟合）
```
Max Epochs: 100
Learning Rate: 5e-4
Network Dim: 64
Network Alpha: 32
Caption Dropout: 0.0
```

### 小数据集（10-30张图）
```
Max Epochs: 30
Learning Rate: 2e-4
Network Dim: 32
Network Alpha: 16
Caption Dropout: 0.05
```

### 大数据集（100+张图）
```
Max Epochs: 10
Learning Rate: 1e-4
Network Dim: 32
Network Alpha: 16
Caption Dropout: 0.1
```

---

## 相关链接

- [FLUX.2 官方文档](./flux_2.md)
- [数据集配置文档](./dataset_config.md#flux2)
- [Musubi-Tuner GitHub](https://github.com/kohya-ss/musubi-tuner)

---

**祝你训练顺利！** 🎉
