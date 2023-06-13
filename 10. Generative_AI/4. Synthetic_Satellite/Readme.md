# Using Stable Diffusion to generate dataset examples
We explore using a fine-tuned version of Stable Diffusion to generate examples of satellite images. Tested using SageMaker Studio, PyTorch 2.0 Python 3.10 GPU kernel on g4dn.xlarge.

In this tutorial, we go through using LoRA and Dreambooth to fine-tune Stable Diffusion, a text-to-image generation model, to generate additional satellite images to augment your dataset.