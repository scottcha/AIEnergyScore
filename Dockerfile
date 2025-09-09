FROM pytorch/pytorch:2.7.1-cuda12.6-cudnn9-runtime

ARG TARGETPLATFORM

ENV PATH=/opt/conda/bin:$PATH

RUN apt-get update && DEBIAN_FRONTEND=noninteractive apt-get install -y --no-install-recommends \
        build-essential \
        ca-certificates \
        ccache \
        curl \
        git && \
        rm -rf /var/lib/apt/lists/*

COPY requirements.txt requirements.txt
RUN pip install -r requirements.txt

RUN git clone https://github.com/huggingface/optimum-benchmark.git /optimum-benchmark && cd /optimum-benchmark && git checkout reasoning_test && pip install -e .

COPY ./check_h100.py /check_h100.py
COPY ./entrypoint.sh /entrypoint.sh
COPY ./summarize_gpu_wh.py /summarize_gpu_wh.py
RUN chmod +x /entrypoint.sh
RUN chmod +x /summarize_gpu_wh.py

ENTRYPOINT ["/entrypoint.sh"]
