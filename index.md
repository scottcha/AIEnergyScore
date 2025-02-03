<nav>
  <ul>
    <li><a href="{{ '/' | relative_url }}">Home</a></li>
    <li><a href="{{ '/about/' | relative_url }}">About</a></li>
    <li><a href="{{ '/contact/' | relative_url }}">Contact</a></li>
  </ul>
</nav>
Option 3: Using a Jekyll _config.yml Navigation
If using a Jekyll theme, define navigation in _config.yml:

yml
Copy
Edit
navigation:
  - title: Home
    url: /
  - title: About
    url: /about/
  - title: Contact
    url: /contact/
Then, render it in _layouts/default.html:

html
Copy
Edit
<nav>
  {% for item in site.navigation %}
    <a href="{{ item.url | relative_url }}">{{ item.title }}</a>
  {% endfor %}
</nav>
Option 4: Using a Jekyll Theme (Minimal Mistakes, Just the Docs)
If using a Jekyll theme like Minimal Mistakes or Just the Docs, navigation can be handled via _config.yml.

Would you like help setting up a theme or custom styling for the navbar?






You said:
I'm using option 1.  How do I format the links to spread them across the top evenly, make them black, bigger font.  Also is there a way to remove the github repo name from the very top of the site? current code <nav>
  <a href="https://huggingface.co/spaces/AIEnergyScore/Leaderboard">Leaderboard</a> |
  <a href="https://huggingface.github.io/AIEnergyScore/#faq">FAQ</a> |
  <a href="https://huggingface.github.io/AIEnergyScore/#documentation">Documentation</a>
</nav>
![AI Energy Score](/AIEnergyScore_LightBG.png)
ChatGPT said:
ChatGPT
To style your navigation links so they are spread evenly across the top, black, and have a bigger font, you can add inline CSS within the Markdown file.

Updated Code with Styling
md
Copy
Edit
<style>
  nav {
    display: flex;
    justify-content: center; /* Centers the links */
    gap: 40px; /* Adds space between links */
    background-color: white; /* Background color if needed */
    padding: 10px 0; /* Padding for spacing */
  }

  nav a {
    text-decoration: none; /* Removes underline */
    color: black; /* Makes links black */
    font-size: 20px; /* Increases font size */
    font-weight: bold; /* Optional: makes text bold */
  }

  nav a:hover {
    text-decoration: underline; /* Underlines on hover */
  }

  body > header {
    display: none; /* Hides GitHub repo name */
  }
</style>

<nav>
  <a href="https://huggingface.co/spaces/AIEnergyScore/Leaderboard">Leaderboard</a>
  <a href="https://huggingface.github.io/AIEnergyScore/#faq">FAQ</a>
  <a href="https://huggingface.github.io/AIEnergyScore/#documentation">Documentation</a>
</nav>


![AI Energy Score](/AIEnergyScore_LightBG.png)

Welcome to AI Energy Score! This is an initiative to establish comparable energy efficiency ratings for AI models, helping the industry make informed decisions about sustainability in AI development.

> [!NOTE]
> This is still a work in progress.

## Key Links
- [Leaderboard](https://huggingface.co/spaces/AIEnergyScore/Leaderboard)
- [FAQ](https://huggingface.github.io/AIEnergyScore/#faq)
- [Documentation](https://huggingface.github.io/AIEnergyScore/#documentation)
   - [Evaluating Proprietary Models](https://huggingface.github.io/AIEnergyScore/#evaluating-a-closed-model)
- [Label Generator](https://huggingface.co/spaces/AIEnergyScore/Label)


# FAQ
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

# Documentation
## Introduction {#introduction}

With the increased ubiquity of large artificial intelligence (AI) models in recent years, the environmental impacts of this technology have become increasingly evident. The latest figures regarding the electricity consumption of data centres is that they consumed 460 terawatt-hours (TWh) in 2022, representing almost 2% of total global electricity demand, and are projected to more than double by 2026, [reaching 1 000 TWh](https://www.iea.org/reports/electricity-2024). AI's contribution towards this number is hard to delineate, mainly given the lack of transparency around the topic \- however, in the recent year alone, both [Microsoft](https://www.theverge.com/2024/5/15/24157496/microsoft-ai-carbon-footprint-greenhouse-gas-emissions-grow-climate-pledge) and [Google](http://www.bloomberg.com/news/articles/2024-07-02/google-s-emissions-shot-up-48-over-five-years-due-to-ai) have announced that they have missed their own climate targets due, at least in part, to the rising energy demands of AI. The growing energy demand of AI can be attributed to multiple factors: on the one hand, the popularity of AI models, especially large language models (LLMs), that are [more energy-intensive](https://arxiv.org/abs/2311.16863) than previous generations of AI approaches as well as the [rapid adoption](https://www.theverge.com/2024/12/4/24313097/chatgpt-300-million-weekly-users) of user-facing AI applications.

Despite these conditions, there is a general lack of consensus regarding what the term "AI" encompasses and therefore how to properly account for all of its direct and indirect environmental impacts. While most of the existing work has focused on AI's energy usage and the ensuing emissions related to its generation, some have proposed to extend this to [AI's broader life cycle](https://arxiv.org/abs/2211.02001) and even to its indirect [impacts on other industries](https://www.nature.com/articles/s41558-022-01377-7). What is largely agreed-upon is that we are currently lacking ways to meaningfully assess the environmental impacts of [individual AI systems](https://aclanthology.org/2021.sustainlp-1.2/) as well as how the field at large contributes towards global warming and the [race to net-zero emissions](https://www.nature.com/articles/d41586-024-01137-x). 

This project aims to establish a standard for reporting AI models' energy efficiency – thereby improving transparency in the field at large –  as well as to define a set of popular tasks to better circumscribe the field of AI – thereby helping make its definition more concrete and tangible. We propose to develop a set of efficiency ratings for AI models inspired by the United States Environmental Protection Agency (EPA)'s Energy Star ratings for different categories of products, and similar rating systems. Given the multitude of downstream functions that AI models can serve, we put forward a set of 10 tasks spanning multiple modalities as a starting point for benchmarking and comparing model efficiency. Based on these tasks, we test hundreds of open- and closed-source models and propose a leaderboard for tracking progress over time. We also suggest ways in which stakeholders from AI developers to policymakers can adopt and adapt AI Energy Score for their own use, in order to factor in efficiency when developing, deploying and selecting models. We conclude by outlining future steps to maintain and evolve the proposed rating system, ensuring its continued relevance over time. 

## Related Work {#related-work}

Our work builds off of past scholarship focused on measuring the environmental impacts of AI as well as benchmarking and comparing AI's energy efficiency. We describe the relevant work below.

### Environmental Impacts of AI {#environmental-impacts-of-ai}

The first article aiming to quantify the environmental impacts of training an AI model was the [seminal study by Strubell et al.](https://arxiv.org/abs/1906.02243), which estimated the carbon emissions of training a Transformer-type model with Neural Architecture Search. Since then, numerous studies have aimed to build upon this work, expanding our knowledge of the factors that influence the carbon emissions of model training ([Patterson 2021](https://arxiv.org/abs/2104.10350), [Patterson 2022](https://arxiv.org/abs/2204.05149), [Luccioni 2023](https://arxiv.org/abs/2302.08476)). Recent years have also seen proposals to expand the scope of these estimates to include the full model life cycle, from the embodied emissions of manufacturing computing hardware to the recurring impacts of model deployment ([Ligozat 2021](https://arxiv.org/abs/2110.11822), [Luccioni 2022](https://arxiv.org/abs/2211.02001)). However, there are still many missing pieces of information that preclude an exact estimate of life cycle impacts. 

### AI Efficiency Benchmarking {#ai-efficiency-benchmarking}

Benchmarking initiatives play a crucial role in evaluating and enhancing the performance and efficiency of AI models. Two notable efforts in this domain are MLPerf and Zeus, each contributing uniquely to AI efficiency benchmarking.

* [MLPerf Inference Benchmark](https://arxiv.org/pdf/1911.02549): Established as an industry-standard benchmark suite, MLPerf evaluates the performance of hardware, software, and services across diverse applications. It encompasses a range of tasks, including image classification, object detection, and language processing, providing a comprehensive assessment of AI system capabilities. While MLPerf’s primary focus has been on performance metrics like throughput, latency, and accuracy, it has started to include [power measurements](https://arxiv.org/abs/2410.12032) as part of its benchmarking suite. The MLPerf Power Measurement benchmark allows participants to optionally report energy usage during training and inference, including metrics such as energy consumed per inference and inferences per joule which provide valuable insights into energy efficiency. However, energy consumption remains an optional evaluation criterion, with performance continuing to be the primary emphasis.  
    
* [Zeus](https://ml.energy/zeus/) is an open-source library developed as part of the [ML.ENERGY](https://ml.energy/) initiative to measure and optimize the energy consumption of deep learning (DL) workloads. It provides both programmatic and command-line interfaces for tracking energy usage with precision and minimal overhead. Zeus supports energy measurement for both GPUs and CPUs, synchronizing their operations to ensure accurate results, and is compatible with NVIDIA and AMD GPUs. However, its functionality is currently limited to open-source models, restricting its applicability for closed-source or proprietary systems. Additionally, energy measurements must be conducted directly by researchers using the library, which may limit scalability for broader adoption or seamless integration into automated workflows. Results are disclosed in the [ML.ENERGY Leaderboard](https://ml.energy/leaderboard/?__theme=light).  
    
* [Ecologits](https://huggingface.co/spaces/genai-impact/ecologits-calculator) has taken steps to address the gap in evaluating closed-source models. This initiative provides approximate energy, carbon, and resource consumption estimates for proprietary models by analyzing the relationship between open-source model energy use and parameter counts, as well as rumored parameter counts for closed-source models. Ecologits extends beyond energy to include carbon emissions and abiotic resource consumption, offering a more holistic view of environmental impact. Furthermore, its [methodology](https://ecologits.ai/latest/) incorporates supply chain considerations, providing insights into the upstream impacts of AI model development and deployment.

The AI Energy Score builds on this existing work by establishing a standardized, standalone energy efficiency benchmark for AI models. Unlike existing initiatives, the AI Energy Score focuses on creating a unified framework that evaluates both open-source and closed-source models in a consistent, transparent, and scalable manner. This includes a comprehensible rating system that distills complex energy measurements into a single, easy-to-understand relative metric, enabling stakeholders to compare models effectively. The project also introduces a public leaderboard to showcase model results, fostering transparency and encouraging innovation in sustainable AI development.

## Methodology {#methodology}

The AI Energy Score methodology combines insights from prior benchmarking efforts to deliver a unified framework for evaluating the energy efficiency of AI models. This section outlines the steps taken to ensure consistency, precision, and scalability in measuring and reporting energy consumption, offering a standardized approach that supports both open and closed-source models. Below, we detail the task definitions, dataset creation, model selection, and experimental setup that underpin this project.

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
