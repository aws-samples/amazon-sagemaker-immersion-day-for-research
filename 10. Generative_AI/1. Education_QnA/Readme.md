Welcome to the first lab for Large Language models. In this lab, we will first deploy a large language model using SageMaker Jumpstart and then use that model to try out multiple education related use cases.

The notebooks for this lab are at https://github.com/aws-samples/amazon-sagemaker-immersion-day-for-research/tree/main/10.%20Generative_AI/1.%20Education_QnA

Use a ml.t3.medium instance and Data Science kernel for this notebook.

## SageMaker Jumpstart for Large Language Models
SageMaker JumpStart provides pretrained, open-source models for a wide range of problem types to help you get started with machine learning. You can access the pretrained models, solution templates, and examples through the JumpStart landing page in Amazon SageMaker Studio. You can also access JumpStart models using the SageMaker Python SDK.

![SageMaker Jumpstart](./jumpstart.png)

Amazon SageMaker JumpStart offers state-of-the-art, built-in foundation models for use cases such as content writing, code generation, question answering, copywriting, summarization, classification, information retrieval, and more. Use JumpStart foundation models to build your own generative AI solutions and integrate custom solutions with additional SageMaker features. 

In the accompanying notebook, you will first deploy a model using the jumpstart python SDK and then run the following  use cases using multiple sources:

1. Keyword generation
2. Summarization 
3. Checking for correct answers
4. Generation of QnA pairs
5. ask questions such as 
   1. What is the storyline?
   2.  who is the main character? 
   3.  what happens in the end?
   4.  What is the main gist of the paper?
   5.  What is the problem being solved?
   6.  What is the conclusion of the paper?
