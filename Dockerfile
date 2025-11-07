FROM pytorch/pytorch:2.9.0-cuda13.0-cudnn9-runtime 

# Update PyTorch to nightly for Blackwell support
#RUN pip install --pre torch torchvision --index-url https://download.pytorch.org/whl/nightly/cu126

ARG TARGETPLATFORM

ENV PATH=/opt/conda/bin:$PATH

RUN apt-get update && DEBIAN_FRONTEND=noninteractive apt-get install -y --no-install-recommends \
        build-essential \
        ca-certificates \
        ccache \
        curl \
        git && \
        rm -rf /var/lib/apt/lists/*

COPY requirements.txt /requirements.txt
# Can use this to install from TestPyPI if needed
# Install requirements including ai_energy_benchmarks from TestPyPI
# RUN pip install --index-url https://test.pypi.org/simple/ \
#     --extra-index-url https://pypi.org/simple/ \
#     -r requirements.txt
RUN pip install -r /requirements.txt

# Install optimum-benchmark (default backend)
RUN git clone https://github.com/huggingface/optimum-benchmark.git /optimum-benchmark && cd /optimum-benchmark && pip install -e .
# Alternative installation methods (for development):
# Option B: Install from local wheel (for local development) - WITH TTFT TRACKING
# COPY ai_energy_benchmarks/dist/ai_energy_benchmarks-*.whl /tmp/
# RUN pip install /tmp/ai_energy_benchmarks-*.whl && rm -rf /tmp/*.whl

COPY entrypoint.sh /entrypoint.sh
COPY *.py /
COPY oct_2025_models.csv /oct_2025_models.csv
COPY text_generation.yaml /optimum-benchmark/energy_star/text_generation.yaml
RUN chmod +x /entrypoint.sh
RUN chmod +x /summarize_gpu_wh.py
RUN chmod +x /run_ai_energy_benchmark.py

ENTRYPOINT ["/entrypoint.sh"]
