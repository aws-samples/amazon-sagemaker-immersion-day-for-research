# What's the use case?
In this lab we will generate synthetic satellite images. These images can be used for research or as input data for building your computer vision models.

# Stable Diffusion
Stable Diffusion is a Generative AI model that makes it easy for you to generate images from text. Besides image generation though, stable diffusion also has a host of other features such as generating an image based on another image and a prompt (image to image), in-painting (modifying part of an image , out-painting (extending the image) and upscaling (increasing the resolution of an image). 

## Why fine tune stable diffusion?
Although Stable diffusion is great at generating images, the quality of images that specialise in a particular are may not be great. For example, in this notebook we aim to generate satellite images. However, the default satellite images that are generated do show some of the features (such as highways) very well. To improve the quality of satellite images with highways, we fine-tune stable diffusion using real satellite images.

## How do we fine-tune
To fine-tune stable diffusion we use a method called DreamBooth which is described [here](https://dreambooth.github.io/). 