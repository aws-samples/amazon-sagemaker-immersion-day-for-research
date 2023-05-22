## Immersion day - Manage research workloads using Amazon sagemaker

This repository contains workshops and notebooks that you can use to learn more about Amazon SageMaker. If you are in an immersion day run by AWS, then follow the instructions specified by your host. If you would like to run the workshop on your own then you would need an AWS account and access to Amazon SageMaker. 



1) [No Code Machine Learning Using Amazon SageMaker Canvas](1.%20No_Code_ML_Using_SageMaker_Canvas/Readme.md) : This first workshop should be the first point in your journey. It walks you through how you can use SageMaker Canvas to perform no-code machine learning. SageMaker Canvas makes it easy for your to do a quick proof of concept and then generate results that you can use to see value in the use case and possibly use the results as data points for your funding application.
2) [Low code Feature Engineering using Amazon SageMaker Data Wrangler](2.%20AutoML_Using_SageMaker_Pilot/README.md) : A lot of Machine learning is about Feature Engineering. Use this workshop to learn how you can use Amazon SageMaker Data Wrangler to perform transformations on your data. 
   
3) [AutoML using SageMaker autopilot]( 3.%20Low_Code_Feature_Engineering_Using_Amazon_Data_Wrangler/README.md )  : If you want to automatically pass your model through hundreds of models and hyperparemets and find the best model automatically then this workshop is for you. It also gives you the actual notebook that it uses to try out the various models so that you can use that notebook and start running your own experiments!
4) [Predict average hospital spending using SageMaker's built-in algorithm](4.%20Hiv_Inhibitor_Prediction_DGL/../README.md) - Amazon SageMaker has built in algorithms that you can use for common use cases. In this workshop, we look at how you can predict average hospital spending using the [linear-learner](https://docs.aws.amazon.com/sagemaker/latest/dg/linear-learner.html) algorithm.
5) [Customer SK Learn Random Forest](5.%20Custom_SKLearn_Random_Forest/../README.md) : In this notebook we show how to use Amazon SageMaker to develop, train, tune and deploy a Random Forest model based using the popular ML framework [Scikit-Learn](https://scikit-learn.org/stable/index.html).
6) [HIV Inhibitor prediction using GNN (Bring your own algorithm to Sagemaker)](6.%20Computer_Vision/) : This example notebook focuses on training multiple Graph neural network models using Deep Graph Librar and deploying it using Amazon SageMaker
7) 
   - [Computer vision 1 : 3D segmentation example of integrating the MONAI framework into Amazon SageMaker](7.%20Computer_Vision/Spleen_Segmentation_GPU/README.md) : This tutorial shows how to run SageMaker managed training using MONAI for 3D Segmentation and SageMaker managed inference after model training. This notebook needs access to GPU instances for training.

   - [Computer vision 2 : Training a Tensorflow Model on MNIST](7.%20Computer_Vision/mnist_cpu/get_started_mnist_train.ipynb) : This notebook shows how to train and deploy a tensorflow model to classify MNIST images. The notebooks can run on CPU.
8) 
   - [Natural Language processing 1: Train a Medical Specialty Detector on SageMaker Using HuggingFace Transformers](8.%20Natural_Language_Processing/Classify_Medical_Specialty_NLP_Huggingface_Transformers_GPU/1_sagemaker_medical_specialty_using_transfomers.ipynb) : In this workshop, we will show how you can train an NLP classifier using trainsformers from [HuggingFace](https://huggingface.co/).  we will use the SageMaker HuggingFace supplied container to train an algorithm that will distinguish between physician notes that are either part of the General Medicine (encoded as 0), or Radiology (encoded as 1) medical specialties. 

   - [Natural Language Processing 2: Fine tune a PyTorch BERT model](8.%20Natural_Language_Processing/Bert_NLP_CPU/bert-sm-python-SDK.ipynb) : The notebook demonstrates how to use Amazon SageMaker to fine tune a PyTorch BERT model and deploy it with Elastic Inference. We walk through our dataset, the training process, and finally model deployment.

9)  
   - [ GeoSpatial 1: semantic segmentation model on a SpaceNet dataset](9.%20Geospatial/amazon-sagemaker-satellite-imagery-segmentation/README.md) : In this workshop, we will demonstrate how to train and host a semantic segmentation model to detect buildings in satellite images. In order to train this model, we will be using the Semantic Segmentation built in-algorithm on the SpaceNet dataset with the Deeplab-v3 backbone.
    
   - [GeoSpatial 2: Deep Learning on AWS Open Data Registry: Automatic Building and Road Extraction from Satellite and LiDAR](9.%20Geospatial/aws-open-data-satellite-lidar-tutorial/README.md) : This section has two notebooks that use Deep Learning on AWS Open Data Registry to perform Automatic Building and Road Extraction from Satellite and LiDAR.

10) Generative AI

       - Medical Fine Tuning (coming soon)
       - Education QnA (coming soon)
       - XRay_Upscaler (coming soon)
       - Synthetic_Satellite (coming soon)

If you are at an event with an AWS event engine provided account then start [here](0.%20Setup/Readme.md)    


## License

This library is licensed under the MIT-0 License. See the LICENSE file.

