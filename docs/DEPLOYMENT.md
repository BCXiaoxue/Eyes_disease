# Docker 部署与服务器训练

部署文件已统一放在 `deployment/` 目录。以下命令均在项目根目录执行。

## 1. SSH 用户名和密码放在哪里更安全

不要把服务器 SSH 用户名、密码写进 `.env`、`docker-compose.yml`、脚本或 Git 仓库。这个仓库的 `.gitignore` 已经忽略 `.env`，适合放应用运行时密钥，例如 `DEEPSEEK_API_KEY`，但不适合放 SSH 登录密码。

推荐做法：

- 最安全：在本机生成 SSH key，把公钥放到服务器 `~/.ssh/authorized_keys`，以后用私钥登录。
- 如果现在只有用户名和密码：部署时在 `ssh` / `scp` 提示里手动输入密码，或者放在 1Password、Bitwarden、系统凭据管理器这类密码管理器里。
- 不推荐：`sshpass`、PowerShell 明文变量、把密码写入 `.env`、把密码提交到 Git。

本机生成 SSH key 示例：

```powershell
ssh-keygen -t ed25519 -C "retinascope-deploy"
ssh-copy-id 用户名@服务器IP
```

如果 Windows 没有 `ssh-copy-id`，可以先用密码登录服务器，再把本机 `~/.ssh/id_ed25519.pub` 的内容追加到服务器的 `~/.ssh/authorized_keys`。

## 2. 上传项目并在服务器启动系统

在服务器安装 Docker 和 Docker Compose plugin 后，把项目传到服务器，例如：

```powershell
scp -r . 用户名@服务器IP:~/eyes_diseases_code
ssh 用户名@服务器IP
cd ~/eyes_diseases_code
cp .env.example .env
```

编辑服务器上的 `.env`，填应用密钥，例如：

```bash
nano .env
```

构建并启动 Web 系统：

```bash
docker compose -f deployment/docker-compose.yml up -d --build retinascope
docker compose -f deployment/docker-compose.yml ps
docker compose -f deployment/docker-compose.yml logs -f retinascope
```

服务器防火墙或云平台安全组需要放行 `8501` 端口。

## 3. 在服务器上通过 `training/baseline.py` 训练模型

训练数据需要在服务器项目目录中保持这个结构：

```text
data/train.csv
train/images/merged/<ID>_merge.jpg
artifacts/new_models/
artifacts/logs/
```

训练单个标签，例如训练 `N`：

```bash
TRAIN_LABEL=N TRAIN_EPOCHS=20 TRAIN_FOLDS=5 TRAIN_BATCH=32 docker compose -f deployment/docker-compose.yml --profile train run --rm trainer
```

训练其他标签，把 `TRAIN_LABEL` 改成 `D`、`G`、`C`、`A`、`H`、`M` 或 `O`：

```bash
TRAIN_LABEL=D docker compose -f deployment/docker-compose.yml --profile train run --rm trainer
```

训练输出会写到：

```text
artifacts/new_models/best_<标签>_fold<折数>.pth
```

如果服务器有 NVIDIA GPU，需要先安装 NVIDIA 驱动和 NVIDIA Container Toolkit，然后在 `deployment/docker-compose.yml` 的 `trainer` 服务中启用：

```yaml
gpus: all
```

## 4. 返回可点击的网址

部署成功后，系统网址通常是：

```text
http://服务器IP:8501
```

例如服务器公网 IP 是 `1.2.3.4`，访问地址就是：

```text
http://1.2.3.4:8501
```

如果配置了域名和反向代理，最终网址可以是：

```text
https://你的域名
```

快速检查当前服务是否健康：

```bash
curl http://127.0.0.1:8501/_stcore/health
```

返回 `ok` 后，就可以用浏览器打开 `http://服务器IP:8501`。
