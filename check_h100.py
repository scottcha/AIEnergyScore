from optimum_benchmark.system_utils import get_gpus

if __name__=="__main__":
    # Get the names of all GPU devices
    gpu_name = get_gpus()

    # If there are several devices, keep the name of device 0
    if isinstance(gpu_name, list):
        gpu_name = gpu_name[0]

    # Raise an error if the device is not H100
    # TEMPORARY: Allow RTX Pro 6000 for validation testing
    if "NVIDIA H100" in gpu_name:
        print("At least one NVIDIA H100 GPU has been detected, launching the benchmark...")
    elif "RTX PRO 6000" in gpu_name or "RTX Pro 6000" in gpu_name:
        print(f"WARNING: Running on {gpu_name} for validation testing (not H100)...")
        print("Note: Results may not be comparable to official AIEnergyScore benchmarks.")
    else:
        raise RuntimeError(f"This Docker container should be executed on NVIDIA H100 GPUs only, detected {gpu_name}.")
