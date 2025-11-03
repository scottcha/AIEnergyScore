![AI Energy Score](/logo.png)

Welcome to AI Energy Score! This is an initiative to establish comparable energy efficiency ratings for AI models, helping the industry make informed decisions about sustainability in AI development.

## Key Links
- [Leaderboard](https://huggingface.co/spaces/AIEnergyScore/Leaderboard)
- [FAQ](https://huggingface.github.io/AIEnergyScore/#faq)
- [Documentation](https://huggingface.github.io/AIEnergyScore/#documentation)
- [Label Generator](https://huggingface.co/spaces/AIEnergyScore/Label)


## Evaluating a Proprietary Model
### Hardware

The Dockerfile provided in this repository is made to be used on the NVIDIA H100-80GB GPU.
If you would like to run benchmarks on other types of hardware, we invite you to take a look at [these configuration examples](https://github.com/huggingface/optimum-benchmark/tree/main/energy_star) that can be run directly with [Optimum Benchmark](https://github.com/huggingface/optimum-benchmark/). However, evaluations completed on other hardware would not be currently compatable and comparable with the rest of the AI Energy Score data.


### Usage

You can build the Docker image with:

```
docker build -t energy_star .
```

Then you can run your benchmark with:

```
docker run --gpus all --shm-size 1g energy_star --config-name my_task backend.model=my_model backend.processor=my_processor
```
where `my_task` is the name of a task with a configuration [here](https://github.com/huggingface/optimum-benchmark/tree/main/energy_star), `my_model` is the name of your model that you want to test (which needs to be compatible with either the Transformers or the Diffusers libraries) and `my_processor` is the name of the tokenizer/processor you want to use. In most cases, `backend.model` and `backend.processor` wil lbe identical, except in cases where a model is using another model's tokenizer (e.g. from a LLaMa model).

The rest of the configuration is explained [here](https://github.com/huggingface/optimum-benchmark/)

> [!WARNING]
> It is essential to adhere to the following GPU usage guidelines:
> - If the model being tested is classified as a Class A or Class B model (generally models with fewer than 66B parameters, depending on quantization and precision settings), testing must be conducted on a single GPU.
> - Running tests on multiple GPUs for these model types will invalidate the results, as it may introduce inconsistencies and misrepresent the model’s actual performance under standard conditions.

Once the benchmarking has been completed, the zipped log files should be uploaded to the [Submission Portal](https://huggingface.co/spaces/AIEnergyScore/submission_portal). The following terms and conditions will need to be accepted upon upload:

*By checking the box below and submitting your energy score data, you confirm and agree to the following:*

1. ***Public Data Sharing**: You consent to the public sharing of the energy performance data derived from your submission. No additional information related to this model including proprietary configurations will be disclosed.*
2. ***Data Integrity**: You validate that the log files submitted are accurate, unaltered, and generated directly from testing your model as per the specified benchmarking procedures.*
3. ***Model Representation**: You verify that the model tested and submitted is representative of the production-level version of the model, including its level of quantization and any other relevant characteristics impacting energy efficiency and performance.*

