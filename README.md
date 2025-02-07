## 1. Project Introduction
Based on the improvement of the Ultralytics YOLOv8 model, this project proposes three core innovations for target detection tasks:
- **RFCBAMConv module**: Replaces the standard convolution with an enhanced convolution module that fuses the receptive field and channels space attention
- **Feature hourglass pyramid network**: Design multi-scale feature pyramid networks for more efficient feature fusion
- **Task dynamic alignment detection head**: An adaptive task alignment mechanism is proposed to improve the performance of the header


## 2. Realization function
The code mainly implements the following improvements:
- ✅ **RFCBAMConv module**：
- Combined with void convolution to enlarge the receptive field
- Embedded CBAM attention mechanism
- Dynamically adjust feature weight distribution
- ✅ **Feature hourglass pyramid network**：
- Cross-scale feature interaction enhancement
- Bidirectional feature fusion path
- Lightweight feature recombination module
- ✅ **Task dynamic alignment detection head**：
- Task-aware feature alignment
- Adaptive spatial attention
- Dynamic convolution kernel generation

## 3. User guide

### System requirement
    python: 3.8.16
    torch: 1.13.1+cu117
    torchvision: 0.14.1+cu117
    timm: 0.9.8
    mmcv: 2.1.0
    mmengine: 0.9.0






