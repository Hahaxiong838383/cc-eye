# cc-eye

摄像头视觉感知系统 -- 识别用户身份、读懂表情、智能互动。

## 功能

- **身份识别**：基于 face_recognition (dlib) 的人脸注册与识别
- **表情识别**：7 类表情实时检测 (FER)，EMA 平滑
- **状态融合**：多信号（表情 + EAR + 眨眼 + 头部朝向 + MAR）融合为 5 种状态
- **智能互动**：cc 风格对话气泡，状态变化自动回应

## 安装

```bash
# macOS 需要先装 dlib 依赖
brew install cmake

# 安装 Python 依赖
pip install -r requirements.txt
```

> dlib 编译可能需要几分钟，耐心等。

## 运行

```bash
python main.py
```

- 首次运行会自动进入人脸注册流程
- ESC 退出，R 重新注册

## 文件结构

```
cc-eye/
  config.py         配置中心
  identity.py       人脸身份识别
  expression.py     表情识别
  state_fusion.py   多信号融合引擎
  interaction.py    互动引擎
  main.py           主循环入口
  face_data/        注册的人脸编码（.pkl）
```

## 要求

- Python 3.9+
- macOS / Linux（摄像头访问）
- 摄像头
