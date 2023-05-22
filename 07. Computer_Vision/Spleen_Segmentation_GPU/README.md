# Spleen Segmentation
In this example, we will demonstrate a 3d segmentation example of integrating the [MONAI](http://monai.io) framework into Amazon SageMaker with the SageMaker Pytorch managed containers, using NIfTI format images as input. This is an adaptation of the [spleen segmentation tutorial](https://github.com/Project-MONAI/tutorials/blob/master/3d_segmentation/spleen_segmentation_3d.ipynb) from the MONAI project. The [dataset](https://registry.opendata.aws/msd/) was obtained from Memorial Sloan Kettering Cancer Center and each image has a different number of slices. A total of 40 images are used for training (31 training, 9 validation), with another 1 labeled test image and 20 unlabeled test images. The training process optimizes to reduce [Dice Loss](https://docs.monai.io/en/stable/losses.html) and the inference script can perform inference on a single slice, a selection of slices or all slices in the image.

+ In `MONAI_BYOS_spleen_segmentation_3D.ipynb` notebook, you can see a typical ML workflow in SageMaker which covers data preparation --> model training --> model deployment --> Inference. 

# Citation
Medical Segmentation Decathlon was accessed on 11 April 2022 from https://registry.opendata.aws/msd. 
