# Preamble
This is an attempt to fine tune the [music generator model](https://github.com/facebookresearch/audiocraft) developed by Meta.  This example references https://github.com/chavinlo/musicgen_trainer.git

MUSICGEN is an autoregressive transformer-based decoder conditioned on text or melodic representation. In this example, we will fine tune the decoder in a multi-gpu single node environment with huggingface's accelerate library and experiment with either audio conditioned or conditionless generation of music.

This notebook was run on AWS using g5.12xlarge (4x A10 GPU).

## Prerequisites
Prepare audio files (.wav) to be used for training and store them in the same directory.  
Prepare accelerate config file
```bash
accelerate config
```

For a basic setup using deepspeed with zero_stage=2, gradient_accumulation_steps=2 on a single machine with 4 gpus, the config (~/.cache/huggingface/accelerate/default_config.yaml) will look like below.
```bash
compute_environment: LOCAL_MACHINE
debug: false
deepspeed_config:
  gradient_accumulation_steps: 2
  offload_optimizer_device: none
  offload_param_device: none
  zero3_init_flag: false
  zero_stage: 2
distributed_type: DEEPSPEED
downcast_bf16: 'no'
machine_rank: 0
main_training_function: main
mixed_precision: 'no'
num_machines: 1
num_processes: 4
rdzv_backend: static
same_network: true
tpu_env: []
tpu_use_cluster: false
tpu_use_sudo: false
use_cpu: false
```

## Environment Setup

```bash
python3 -m venv venv # create virtual environment
source venv/bin/activate
pip install requirements.txt # install dependencies
```

## Run
### Train

```bash
python main.py train
```

### Generate

```bash
python main.py gen
```