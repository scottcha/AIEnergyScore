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
# Copy and install from local source
COPY ai_energy_benchmarks /ai_energy_benchmarks
RUN pip install -e /ai_energy_benchmarks

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
