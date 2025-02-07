# YOLOv8-Improved: Enhanced Object Detection with RFCBAMConv and FHPN

## 1. 项目介绍
本项目基于Ultralytics YOLOv8模型进行改进，针对目标检测任务提出三项核心创新：
- **RFCBAMConv模块**：将标准卷积替换为融合感受野与通道空间注意力的增强卷积模块
- **FHPN特征融合架构**：设计多尺度特征金字塔网络实现更高效的特征融合
- **动态任务对齐检测头**：提出自适应任务对齐机制提升检测头性能


## 2. 实现功能
代码主要实现以下改进：
- ✅ **RFCBAMConv模块**：
  - 结合空洞卷积扩大感受野
  - 嵌入CBAM注意力机制
  - 动态调整特征权重分布
- ✅ **FHPN特征金字塔**：
  - 跨尺度特征交互增强
  - 双向特征融合路径
  - 轻量化特征重组模块
- ✅ **动态检测头**：
  - 任务感知特征对齐
  - 自适应空间注意力
  - 动态卷积核生成

## 3. 使用指南

### 系统要求
- Python 3.8+
- PyTorch 1.12.1+
- CUDA 11.6+
- NVIDIA GPU with ≥8GB VRAM

### 快速开始
1. 克隆仓库：
```bash
git clone https://github.com/guanguansheng/RFT_YOLO
cd yolov8-improved
