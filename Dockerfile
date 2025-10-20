FROM pytorch/pytorch:2.8.0-cuda12.9-cudnn9-runtime 

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

COPY AIEnergyScore/requirements.txt requirements.txt
RUN pip install -r requirements.txt

# Install optimum-benchmark (default backend)
RUN git clone https://github.com/huggingface/optimum-benchmark.git /optimum-benchmark && cd /optimum-benchmark && git checkout reasoning_test && pip install -e .

# Install ai_energy_benchmarks (optional backend)
# Option A: Install from TestPyPI (for ppe testing)
ARG AI_ENERGY_BENCHMARKS_VERSION=0.0.1rc1
RUN pip install --index-url https://test.pypi.org/simple/ \
    --extra-index-url https://pypi.org/simple/ \
    ai_energy_benchmarks==${AI_ENERGY_BENCHMARKS_VERSION}

# Option B: Install from local wheel (for local development)
# COPY ai_energy_benchmarks/dist/ai_energy_benchmarks-*.whl /tmp/
# RUN pip install /tmp/ai_energy_benchmarks-*.whl && rm -rf /tmp/*.whl

# Option C: Install from production PyPI (future)
# ARG AI_ENERGY_BENCHMARKS_VERSION=0.0.1
# RUN pip install ai_energy_benchmarks==${AI_ENERGY_BENCHMARKS_VERSION}

COPY AIEnergyScore/check_h100.py /check_h100.py
COPY AIEnergyScore/entrypoint.sh /entrypoint.sh
COPY AIEnergyScore/summarize_gpu_wh.py /summarize_gpu_wh.py
COPY AIEnergyScore/run_ai_energy_benchmark.py /run_ai_energy_benchmark.py
COPY AIEnergyScore/text_generation.yaml /optimum-benchmark/energy_star/text_generation.yaml
COPY AIEnergyScore/text_generation_gptoss.yaml /optimum-benchmark/energy_star/text_generation_gptoss.yaml
RUN chmod +x /entrypoint.sh
RUN chmod +x /summarize_gpu_wh.py
RUN chmod +x /run_ai_energy_benchmark.py

ENTRYPOINT ["/entrypoint.sh"]
