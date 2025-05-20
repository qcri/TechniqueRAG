# TechniqueRAG
TechniqueRAG: Retrieval Augmented Generation for Adversarial Technique Annotation in Cyber Threat Intelligence Text

# Citation

```
@misc{lekssays2025techniqueragretrievalaugmentedgeneration,
      title={TechniqueRAG: Retrieval Augmented Generation for Adversarial Technique Annotation in Cyber Threat Intelligence Text}, 
      author={Ahmed Lekssays and Utsav Shukla and Husrev Taha Sencar and Md Rizwan Parvez},
      year={2025},
      eprint={2505.11988},
      archivePrefix={arXiv},
      primaryClass={cs.CR},
      url={https://arxiv.org/abs/2505.11988}, 
}
```

# Getting started

- Install dependencies: `pip install -r requirements.txt`

# Finetuning (Models and Datasets)

### Datasets

The datasets will be stored in the `./datasets` folder. They are hosted on HuggingFace for easy access:

- [TechniqueRAG-Datasets](https://huggingface.co/datasets/qcri-cs/TechniqueRAG-Datasets)

You can download them by running `python3 download_datasets.py`

### Finetuned Models

We host our finetuned models on HuggingFace for better reproducibility of our work:

- [TechniqueRAG-FS-Ministral-8B](https://huggingface.co/qcri-cs/TechniqueRAG-FS-Ministral-8B): Our main models based on the few shot examples retrieved by our RankGPT pipeline
- [TechniqueRAG-ZS-Ministral-8B](https://huggingface.co/qcri-cs/TechniqueRAG-ZS-Ministral-8B): The zeroshot model
- [TechniqueRAG-Reflection-Ministral-8B](https://huggingface.co/qcri-cs/TechniqueRAG-Reflection-Ministral-8B): The reflection model

### Finetuning

For finetuning, we used LLaMa-Factory. The YAML configuration files are available in the `./finetuning` folder.

# Run the pipeline

This tool runs inference using LLMs to identify MITRE ATT&CK techniques from cybersecurity threat descriptions. It processes test datasets and saves prediction results that can later be evaluated using the evaluation script.

## Usage

```bash
# Run inference with a local model
python main.py --name dataset_name --model model_name

# Use a custom API endpoint
python main.py --name dataset_name --model model_name --base_url "http://your-api-endpoint:9003/v1"
```

### Required Arguments

- `--name`: Name of the dataset file (without .json extension) stored in `./datasets/test/`.
- `--model`: Name of the model to use for inference

### Optional Arguments

- `--base_url`: Base URL of the vLLM hosted model with OpenAI-compatible API (default: "http://localhost:9003/v1")

### Input Format

The script expects dataset files in `datasets/test/{name}.json` with the following structure:

```json
[
  {
    "instruction": "You are a cybersecurity expert specializing in the MITRE ATT&CK framework...",
    "input": "The exploit also contains an additional check that ATMFD.dll is of the exact version...",
    "gold": ["T1518", "T1203"]
  },
  ...
]
```

### Output

The script generates result files in the `./results/` directory named `{dataset_name}_results.json` containing the original data plus predictions:

```json
[
  {
    "instruction": "...",
    "input": "...",
    "gold": ["T1518", "T1203"],
    "predicted": ["T1082"]
  },
  ...
]
```


## Our IntelEx Implementation

This is our own implementation of the following work:

> Xu, M., Wang, H., Liu, J., Lin, Y., Liu, C. X. Y., Lim, H. W., & Dong, J. S. (2024). IntelEX: A LLM-driven Attack-level Threat Intelligence Extraction Framework. arXiv preprint arXiv:2412.10872.


IntelEX processes cyber threat intelligence data to extract and validate MITRE ATT&CK techniques using LLM-based validation. The framework includes:

1. **Inference Pipeline** - Extracts MITRE ATT&CK techniques from threat descriptions
2. **Evaluation Script** - Measures the accuracy of technique identification across models

## Pipeline Usage

```bash
python intelex_pipeline.py --input <input_file> --output <output_file> [options]
```

### Configuration

Use environment variables or command line arguments to configure API access:

- `OPENAI_API_KEY`: OpenAI API key
- `AZURE_OPENAI_KEY`: Azure OpenAI API key
- `AZURE_OPENAI_API_ENDPOINT`: Azure OpenAI endpoint URL
- `LOCAL_LLM_URL`: URL for local LLM server (if used)


# Evaluation

This script evaluates how well different models identify MITRE ATT&CK techniques from cybersecurity threat descriptions. It calculates precision, recall, F1 score, and Mean Reciprocal Rank (MRR) metrics across multiple models.

### Usage

```bash
python evaluate.py --mode {technique, subtechnique}
```

### Input Format

The script expects result files in the `./results` directory with names ending in `_results.json`. Each file should contain a list of examples with the following structure:

```json
[
  {
    "instruction": "...",
    "input": "...",
    "gold": ["T1518", "T1203"],
    "predicted": ["T1082"]
  },
  ...
]
```

### Output

The script generates:
1. A markdown table comparing model performance
2. A file `model_comparison_results.md` with the same table
3. Analysis files in `./analysis/{model_name}_analysis.jsonl` for detailed error analysis


