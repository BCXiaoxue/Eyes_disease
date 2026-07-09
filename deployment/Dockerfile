FROM docker.m.daocloud.io/library/python:3.10-slim AS builder

WORKDIR /app

RUN sed -i \
    -e 's|http://deb.debian.org/debian|http://mirrors.aliyun.com/debian|g' \
    -e 's|http://security.debian.org/debian-security|http://mirrors.aliyun.com/debian-security|g' \
    /etc/apt/sources.list.d/debian.sources

RUN apt-get update && apt-get install -y --no-install-recommends \
    gcc g++ libgomp1 libglib2.0-0 libsm6 libxext6 libxrender-dev \
    && rm -rf /var/lib/apt/lists/*

COPY requirements.txt .
RUN pip install --no-cache-dir --upgrade pip \
    && pip install --no-cache-dir -i https://pypi.tuna.tsinghua.edu.cn/simple -r requirements.txt


FROM docker.m.daocloud.io/library/python:3.10-slim AS runtime

WORKDIR /app

RUN sed -i \
    -e 's|http://deb.debian.org/debian|http://mirrors.aliyun.com/debian|g' \
    -e 's|http://security.debian.org/debian-security|http://mirrors.aliyun.com/debian-security|g' \
    /etc/apt/sources.list.d/debian.sources

RUN apt-get update && apt-get install -y --no-install-recommends \
    libgomp1 libglib2.0-0 libsm6 libxext6 libxrender-dev \
    && rm -rf /var/lib/apt/lists/*

COPY --from=builder /usr/local/lib/python3.10/site-packages /usr/local/lib/python3.10/site-packages
COPY --from=builder /usr/local/bin /usr/local/bin

COPY app.py .
COPY assets/ assets/
COPY training/ training/
COPY utils/ utils/
COPY knowledge/ knowledge/

RUN mkdir -p artifacts/models artifacts/new_models artifacts/feedback \
    artifacts/figs artifacts/logs data train/images test/images

RUN mkdir -p /root/.streamlit
COPY --chown=root:root deployment/streamlit_config.toml /root/.streamlit/config.toml

EXPOSE 8501

HEALTHCHECK --interval=30s --timeout=10s --start-period=15s --retries=3 \
    CMD python -c "import urllib.request; urllib.request.urlopen('http://localhost:8501/_stcore/health')" || exit 1

CMD ["streamlit", "run", "app.py", \
     "--server.port=8501", \
     "--server.address=0.0.0.0", \
     "--server.headless=true", \
     "--browser.gatherUsageStats=false"]
