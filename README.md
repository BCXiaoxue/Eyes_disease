# RetinaScope | 眼底疾病智能筛查与辅助问诊系统

RetinaScope 是一套面向眼底筛查场景的比赛工程项目：输入左右眼底图像后，系统完成图像质量检查、八类疾病多标签预测、GradCAM++ 可解释热力图展示，并生成结构化辅助报告。项目同时集成本地眼科知识库检索和可选的大模型问诊辅助，用于呈现从影像识别到临床沟通参考的完整交互流程。

> 本系统用于科研展示与筛查辅助，不替代专业医生诊断或治疗决策。

## 功能亮点

| 模块 | 能力 |
| --- | --- |
| 智能筛查 | 支持正常、糖尿病视网膜病变、青光眼、白内障、AMD、高血压视网膜病变、病理性近视和其他异常的多标签预测 |
| 质量控制 | 对上传图像的清晰度、亮度和可用性进行提示，降低低质量影像带来的误判风险 |
| 可解释分析 | 展示 GradCAM++ 注意力热力图，便于观察模型关注区域 |
| 报告与反馈 | 自动生成 Markdown 辅助报告，支持医生反馈与复查标记的本地留存 |
| 知识辅助 | 基于 `knowledge/` 的轻量本地检索，为 AI 问诊提供可追溯的参考上下文 |
| 工程交付 | 提供训练、评估、测试及 Docker 部署配置，便于参赛演示与服务器部署 |

## 技术栈

`Python` · `Streamlit` · `PyTorch` · `Torchvision` · `TorchCAM` · `Pandas` · `Altair` · `Docker Compose`

## 项目结构

```text
eyes_diseases_code/
|-- app.py                       # Web 系统入口
|-- requirements.txt             # Python 依赖
|-- .env.example                 # API 配置示例，不含密钥
|-- .streamlit/                  # 本地 Streamlit 配置
|-- utils/                       # 推理、路径、知识检索与存储模块
|-- knowledge/                   # 本地眼科知识库 Markdown 文档
|-- data/                        # 标签与训练/验证 CSV
|-- train/                       # 训练图像目录，原始大文件按比赛要求处理
|-- test/                        # 测试图像目录，原始大文件按比赛要求处理
|-- artifacts/                   # 模型、图表与运行输出
|-- training/                    # 训练、评估、预测和数据预处理脚本
|-- tests/                       # 轻量功能测试
|-- deployment/                  # Dockerfile、Compose 与容器配置
|-- docs/                        # 部署文档
`-- LICENSE
```

## 比赛上传清单

建议上传以下内容：

| 类别 | 上传内容 | 说明 |
| --- | --- | --- |
| 核心程序 | `app.py`、`utils/` | Web 推理与功能实现 |
| 知识库 | `knowledge/` | RAG 使用的本地医学资料 |
| 训练代码 | `training/` | 可复现训练、评估和预测过程 |
| 测试 | `tests/` | 用于演示基础可运行性 |
| 配置与依赖 | `requirements.txt`、`.env.example`、`.streamlit/` | 不包含真实密钥 |
| 部署 | `deployment/`、`.dockerignore` | Docker 部署与服务器运行 |
| 文档 | `README.md`、`docs/`、`LICENSE` | 提交说明与部署指南 |
| 数据索引 | `data/*.csv` | 仅在比赛规则允许提交数据标签时上传 |

模型文件需按比赛平台的大小限制决定提交方式。Web 推理默认读取：

```text
artifacts/models/best_<标签>_fold5.pth
```

若平台允许上传权重，请提供 `artifacts/models/`；若平台不允许或大小受限，请在提交说明中给出权重下载方式，并保留上述目录结构。

## 不应上传的内容

以下内容属于本机环境、隐私数据或运行生成物，不建议放入参赛压缩包或公开仓库：

```text
.env                              # 真实 API Key
.venv_cuda/                       # 本机 Python/CUDA 环境
__pycache__/  *.pyc               # Python 缓存
artifacts/feedback/               # 医生反馈记录
artifacts/reports/                # 诊断报告，可能含病例信息
artifacts/logs/                    # 训练及运行日志
artifacts/server_trained_models/  # 临时服务器训练输出
```

如果 `train/images/`、`test/images/` 是赛事受限数据，也不要公开上传，应遵守赛事的数据使用规定。

## 本地运行

在项目根目录创建环境并安装依赖：

```bash
python -m venv .venv
pip install -r requirements.txt
```

需要 AI 问诊接口时，将示例配置复制为本地配置并填写自己的密钥：

```bash
cp .env.example .env
```

启动 Web 系统：

```bash
streamlit run app.py
```

系统推理前需要确保 `artifacts/models/` 下存在所需模型权重。

## 训练与验证

训练单标签模型：

```bash
python training/baseline.py --label N --epochs 20 --folds 5
```

训练 MoE 模型：

```bash
python training/train_moe.py --epochs 30 --batch 16
```

运行本地知识库检索冒烟测试：

```bash
python tests/test_rag.py
```

训练脚本默认读取 `data/` 与 `train/images/merged/`，并将新权重输出到 `artifacts/new_models/`。

## Docker 部署

部署配置统一位于 `deployment/`。在项目根目录执行：

```bash
docker compose -f deployment/docker-compose.yml up -d --build retinascope
```

服务默认通过 `8501` 端口访问。更完整的服务器部署与训练步骤见 `docs/DEPLOYMENT.md`。

## 提交前检查

提交压缩包或推送仓库前，建议确认：

- 不包含 `.env`、访问令牌、服务器密码或病例隐私记录。
- 模型文件是否符合平台体积限制，且与代码预期路径一致。
- 若不上传图像数据，文档中已说明数据获取或评测方挂载方式。
- 在干净环境中可安装 `requirements.txt` 并启动 `streamlit run app.py`。
- 已运行 `python tests/test_rag.py` 验证本地知识检索模块。
