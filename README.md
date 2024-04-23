# MLLM Evaluation Suite (GPU)

## Overview
This repo provides an evaluation pipeline for MLLMs on GPU. The benchmark is designed to be model agnostic. To evaluate your model, you should implement a model interface in `model_eval/model_interface.py` (an example is already provided in it). You can run evaluation on multiple benchmarks through a single command.

## Install

We assume our model is based on the original LLaVA codebase, so you should first install LLaVA and additionally install the following packages.
```
pip install datasets shortuuid openpyxl pyarrow pandas openai==0.28.0
```
Add the path of LLaVA to PYTHONPATH.

### GCP CLI
<details>
<summary>Install the `gcloud` CLI without sudo</summary>
    
Source: <a href="https://www.perplexity.ai/search/install-gcloud-cli-BuLeoh1QSR.FluoCX8XH8w#0">Perplexity</a>

To install the Google Cloud SDK (gcloud CLI) without using sudo, you can follow these steps:
Download the Google Cloud SDK installer:

```bash
curl https://sdk.cloud.google.com | bash
```

This will download and extract the SDK to your home directory in a folder called google-cloud-sdk.
Initialize the SDK:

```bash
~/google-cloud-sdk/bin/gcloud init
```

This will prompt you to log in to your Google account and select the project you want to use.
Add the SDK bin directory to your PATH:

```bash
export PATH=$PATH:~/google-cloud-sdk/bin
```

You can add this line to your shell startup file (e.g. `~/.bashrc`, `~/.zshrc`) to make the change permanent.
</details>

## Convert TPU weights for Eval

To consolidate the checkpoints, run `consolidate.py`

Then you need to convert the consolidated weight to hf model by running `convert_hf_model.py`.

## Benchmark dataset

Cassio A6000 machine: `/home/pw2436/penghaowu_workspace/evaluation/benchmarks`

Local A100 69.30.0.74: `/home/penghao/penghao_workspace/evaluation/benchmarks`

## Benchmark Suite Evaluation
For running multiple benchmarks, you need to first modify the configuration file `config.json` which has the following structure.
```javascript
{   
    "model_module": "llava_hf",  // The class name of the model interface defined in model_eval/model_interface.py
    "model_path": "llava-hf/llava-1.5-7b-hf", // Path to model weight
    "data_folder": "/home/pw2436/penghaowu_workspace/evaluation/benchmarks",  // Path to the benchmark data folder
    "output_name": "llava1.5_hf_7b", // Name of the output for your model evaluation
    "temperature": 0,
    "top_p": null,
    "num_beams": 1,
    "max_length": 1024, // You can override the above four parameters for each benchmark by providing their value in the dict for each benchmark below
    "benchmarks": { // A dict of configs for different benchmarks, the 'eval' item indicates whether this benchmark would be evaluated
        "mmbench": {
            "eval": true
        },
        ...
    }
}
```
To launch the evaluation, run
```python launch_evaluation.py```
By default, it will utilize all GPUs to run the evaluation distributedly, but you can also decide how many GPUs to use by adding `CUDA_VISIBLE_DEVICES=0,1,2,3` before the command.

The predictions will be stored under a `results` folder.

**Note:**   
For evaluation on LLaVA-Bench and mathvista, remember to provide the openai-api-key in the config file. Also note that previous methods like LLaVA use `gpt-4-0314` which is deprecated now, so we currently use `gpt-4-0613`.   
The SEED-Bench video question part is updated compared with the version used by LLaVA.
