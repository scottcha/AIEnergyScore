![AI Energy Score](/AIEnergyScore_LightBG.png)
A repository for the AI Energy Score Project, aiming to establish energy efficiency ratings for AI models.

## Key Links
- [Leaderboard](https://huggingface.co/spaces/AIEnergyScore/Leaderboard)
- [FAQ] (https://huggingface.github.io/AIEnergyScore/#faq)
- [Documentation] (https://huggingface.github.io/AIEnergyScore/#documentation)
-   - [Evaluating a Closed Model](https://huggingface.github.io/AIEnergyScore/#evaluating-a-closed-model)).
- [Label Generator] (https://huggingface.co/spaces/AIEnergyScore/Label)


> [!NOTE]
> This is still a work in progress.


## FAQ
### What is the goal of this project?
The goal of AI Energy Score is to establish standardized benchmarks for evaluating the energy efficiency of AI model inference. By focusing on controlled and comparable metrics, by focusing on specific tasks and hardware (e.g., NVIDIA H100 GPUs), we aim to provide actionable insights for researchers, developers, and organizations.
We hope that the establishment of this easy-to-use and recognizable system can encourage customers to advocate for AI model developers to disclose energy efficiency data. Additionally, we want product developers and end users to easily identify and select the most energy-efficient models for their needs. Furthermore, policymakers and enterprises can use these benchmarks as procurement criteria and a foundation for developing regulatory frameworks. Ultimately, our mission is to drive awareness, encourage transparency, and foster sustainability in AI deployments.

### What do the star ratings mean?
The star ratings represent the relative energy efficiency of an AI model for a specific task (and class for Text Generation tasks) on a particular leaderboard (e.g., February 2025). It is important to understand the timing of the leaderboard when interpreting the star rating, as it reflects the model's performance relative to other models evaluated at that time.
While the numerical GPU Energy (measured in Watt-hours) value remains consistent as long as the hardware, model, and task don't change, the relative star rating is recalibrated every six months. This ensures the ratings stay up-to-date with improvements in energy efficiency across the AI landscape.
- 5 Stars: Most energy efficient
- 1 Star: Least energy efficient
The star rating system was chosen for its simplicity and familiarity, and the 5-star range provides sufficient granularity. The ratings are calculated by grouping the GPU Energy results into 20% intervals, ranking models based on their relative performance within these bands.
This system makes it easier for users to quickly compare models and choose those with optimal energy efficiency for their needs.

### Why are you focused only on inference?
The methodology for calculating energy consumption during training is generally well understood, as it typically involves discreet and measurable computation (verifying training energy consumption, however, is challenging because it often depends on trusting data provided by the AI model developers).
Inference energy consumption, on the other hand, presents a much more complex challenge. It is influenced by a wide range of variables such as hardware configurations, model optimizations, deployment scenarios, and usage patterns. These complexities make inference energy harder to measure and compare reliably. By focusing on inference, we aim to address this gap and establish standardized benchmarks that can bring more clarity and comparability to this critical aspect of AI model deployment.

### What about performance?
While the AI Energy Score primarily focuses on energy efficiency, model performance is not overlooked. Users are encouraged to consider energy efficiency alongside key performance metrics, such as throughput, accuracy, and latency, to make balanced decisions when selecting models. By providing a clear and transparent efficiency rating, the AI Energy Score enables stakeholders to weigh these factors effectively based on their specific requirements.

### Given that there are many variables that affect AI model inference efficiency, what steps have you taken to ensure comparability?
To ensure comparability, we have taken several steps to control key variables that impact AI model inference efficiency. These include:
- Standardized Task: Uniform datasets for each task ensure that we are verifying comparable, real-world performance.
- Standardized Hardware: All benchmarks are conducted exclusively on NVIDIA H100 GPUs, and the score is focused solely on the GPU, eliminating variability introduced by different hardware setups.
- Energy Focus: By measuring energy consumption rather than emissions, we avoid discrepancies caused by the carbon intensity of the energy grid at different physical locations.
- Consistent Configuration: Models are tested using their default configurations, mimicking realistic production scenarios.
- Controlled Batching: We use the same batching strategies across tests to ensure consistency in workload management.
By standardizing these factors, we have attempted to create a controlled environment that allows for fair and reliable comparisons of energy efficiency across different AI models.

### How does the AI Energy Score account for hardware differences?
The AI Energy Score standardizes evaluations by conducting all benchmarks on NVIDIA H100 GPUs, ensuring consistent hardware conditions across all tested models. This approach allows for "apples-to-apples" comparisons by isolating GPU energy consumption under equivalent scenarios. While the score primarily focuses on results from the H100 hardware, users who wish to benchmark on different hardware can use the configuration examples and instructions provided in the associated Optimum Benchmark repository.
However, it’s important to note that results obtained on alternative hardware may not align directly with the standardized GPU-specific metrics used in this study. To ensure clarity, benchmarks performed on different hardware are not included in the official leaderboard but can serve as valuable insights for internal comparisons or research purposes.

### What is the timeline for updates?
The AI Energy Score leaderboard is updated biannually, approximately every six months. During each update, new models are added, and existing ratings are recalibrated to reflect advancements in the field. This regular update cycle ensures that the leaderboard remains a current and reliable resource for evaluating energy efficiency.

### Why include closed-source models?
Closed-source models are integral to ensuring that the AI Energy Score provides a comprehensive and representative benchmark. Many industry-leading AI systems are proprietary, and excluding them would leave significant gaps in the evaluation landscape. By incorporating both open-source and closed-source models, the AI Energy Score encourages industry-wide transparency.
### How are closed-source model ratings verified?
Closed-source model ratings are verified through a process designed to ensure integrity and transparency while respecting confidentiality. When log files are uploaded for evaluation, the uploader must agree to the following attestation, which meets the typical bar for energy efficiency ratings:
- Public Data Sharing: You consent to the public sharing of the energy performance data derived from your submission. No additional information related to this model, including proprietary configurations, will be disclosed.
- Data Integrity: You validate that the log files submitted are accurate, unaltered, and generated directly from testing your model as per the specified benchmarking procedures.
- Model Representation: You verify that the model tested and submitted is representative of the production-level version of the model, including its level of quantization and any other relevant characteristics impacting energy efficiency and performance.
This attestation ensures that submitted results are accurate, reproducible, and aligned with the project’s benchmarking standards.

### Is this data public?
Yes, the benchmarking results are made publicly available via the AI Energy Score leaderboard hosted on Hugging Face. However, strict guidelines are in place to ensure that sensitive data from closed-source models remains confidential. Only aggregate metrics and anonymized results are shared, ensuring privacy while promoting transparency.
### What is the rationale for selecting the initial set of tasks?
The initial set of tasks was chosen to represent a broad spectrum of commonly used machine learning applications across multiple modalities, ensuring relevance and coverage for a wide range of AI models. These tasks include text generation, image classification, object detection, and others, reflecting both the popularity of these tasks and their significance in real-world applications.
The selection was guided by data from the Hugging Face Hub, which tracks model downloads in real time, highlighting tasks with the highest demand. Each task corresponds to a pipeline within the Transformers library, facilitating standardized model loading and evaluation. By focusing on well-established and frequently used tasks, the AI Energy Score ensures that its benchmarks are practical, widely applicable, and reflective of current trends in AI development.
Future iterations aim to expand this list to include emerging tasks and modalities, maintaining the framework’s relevance as the AI landscape evolves.

### How is this different from existing projects?
The AI Energy Score builds on existing initiatives like MLPerf, Zeus, and Ecologits by focusing solely on standardized energy efficiency benchmarking for AI inference. Unlike MLPerf, which prioritizes performance with optional energy metrics, or Zeus and Ecologits, which may be limited by open-source constraints or estimation methods, the AI Energy Score provides a unified framework that evaluates both open-source and proprietary models consistently.
With a simple, transparent rating system and a public leaderboard, it enables clear model comparisons, filling critical gaps in scalability and applicability left by other projects.

## Documentation

## Evaluating a Closed Model
### Hardware

The Dockerfile provided in this repository is made to be used on NVIDIA H100 GPUs.
If you would like to run benchmarks on other types of hardware, we invite you to take a look at [these configuration examples](https://github.com/huggingface/optimum-benchmark/tree/energy_star_dev/examples/energy_star) that can be run directly with [Optimum Benchmark](https://github.com/huggingface/optimum-benchmark/tree/energy_star_dev).


### Usage

You can build the Docker image with:

```
docker build -t energy_star .
```

Then you can run your benchmark with:

```
docker run --gpus all --shm-size 1g energy_star --config-name my_task backend.model=my_model backend.processor=my_processor 
```
where `my_task` is the name of a task with a configuration here: https://github.com/huggingface/optimum-benchmark/tree/energy_star_dev/examples/energy_star, `my_model` is the name of your model that you want to test (which needs to be compatible with either the Transformers or the Diffusers libraries) and `my_processor` is the name of the tokenizer/processor you want to use. In most cases, `backend.model` and `backend.processor` wil lbe identical, except in cases where a model is using another model's tokenizer (e.g. from a LLaMa model).

The rest of the configuration is explained here: https://github.com/huggingface/optimum-benchmark/tree/energy_star_dev?tab=readme-ov-file#configuration-overrides-%EF%B8%8F
