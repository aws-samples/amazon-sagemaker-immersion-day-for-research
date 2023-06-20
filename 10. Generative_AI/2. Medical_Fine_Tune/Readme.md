# Text Summarization for healthcare
## Background and Motivation
Generative AI has a lot of potential uses in healthcare, including patient care, medical research and supporting doctors.
While many challenges need to be addressed such as responsibility of its output, quality control of production models, the benefits of generative AI in healthcare might be big.

One of usecases is text summarization which is a Natural Language Processing task for healthcare.
In this sample, we will use MeQSum Dataset[1] for demonstrating the task.
The dataset is for summarizing consumer health questions.

## Text Summerization for healthcare using Large Language Model (LLM)
In this sample, we will show how you can fine-tune LLM using healthcare data to summarize it.
There are two parts for this section.

### Part1. fine-tuning flan-t5 in notebook local
In this part, we will use [HuggingFace](https://huggingface.co/) to train flan-t5-small in the notebook (summerizing_medical_text-flan-t5.ipynb).

The notebook show how you can train flan-t5-small using your own dataset on the local notebook instance. When you open the notebook, select ml.g4dn.xlarge as the instance and Data Science 3.0 as the kernel.

### Part2. fine-tuning flan-t5 using SageMaker Training
In this part, we will use the SageMaker HuggingFace supplied container to train flan-t5-base in the notebook(summerizing_medical_text_flan-t5_sagemaker.ipynb).

The notebook show how you can train flan-t5-base using [SageMake Training](https://docs.aws.amazon.com/sagemaker/latest/dg/how-it-works-training.html).

When you open the notebook, select ml.t3.medium as the instance and Data Science 3.0 as the kernel.

SageMaker Training allows you to train larger data and models like flan-t5-large if needed. 

## Reference
[1] MeQSum Dataset: "On the Summarization of Consumer Health Questions". Asma Ben Abacha and Dina Demner-Fushman. In Proceedings of the 57th Annual Meeting of the Association for Computational Linguistics, ACL 2019. (https://www.aclweb.org/anthology/P19-1215)  
@Inproceedings{MeQSum, author = {Asma {Ben Abacha} and Dina Demner-Fushman}, title = {On the Summarization of Consumer Health Questions}, booktitle = {Proceedings of the 57th Annual Meeting of the Association for Computational Linguistics, ACL 2019, Florence, Italy, July 28th - August 2}, year = {2019}, abstract = {Question understanding is one of the main challenges in question answering. In real world applications, users often submit natural language questions that are longer than needed and include peripheral information that increases the complexity of the question, leading to substantially more false positives in answer retrieval. In this paper, we study neural abstractive models for medical question summarization. We introduce the MeQSum corpus of 1,000 summarized consumer health questions. We explore data augmentation methods and evaluate state-of-the-art neural abstractive models on this new task. In particular, we show that semantic augmentation from question datasets improves the overall performance, and that pointer-generator networks outperform sequence-to-sequence attentional models on this task, with a ROUGE-1 score of 44.16%. We also present a detailed error analysis and discuss directions for improvement that are specific to question summarization. }}
