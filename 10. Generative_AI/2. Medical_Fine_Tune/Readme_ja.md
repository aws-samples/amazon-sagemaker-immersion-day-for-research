# ヘルスケアのためのテキスト要約
## 背景と動機
生成系 AI は、患者のケア、医学研究、医師のサポートなど、ヘルスケアにおける多くの用途が期待されています。
その出力に対する責任やモデルの品質管理など、多くの課題を解決する必要がありますが、ヘルスケアにおける生成系 AI のメリットは大きいかもしれません。

そのひとつが、医療における自然言語処理タスクであるテキスト要約です。
このサンプルでは、MeQSumデータセット[1]を使って、このタスクを実演します。
このデータセットは、患者の健康に関する質問を要約するためのものです。

## 大規模言語モデル(LLM)を用いたヘルスケアのためのテキスト要約
このサンプルでは、ヘルスケアのデータを使ってLLMをFine-tuningし、テキスト要約する方法を紹介します。
このセクションは2つのパートに分かれています。

### Part1. Notebook 上で flan-t5 を Fine-tuning する
このパートでは、ノートブック(summerizing_medical_text-flan-t5_ja.ipynb)を使います。
[HuggingFace](https://huggingface.co/)を用いて、flan-t5-small を学習します。

このノートブックでは、ノートブックインスタンス上で自分のデータセットを使って flan-t5-small を学習させる方法を紹介しています。インスタンスには ml.g4dn.xlarge を、カーネルには Data Science 3.0 を選択します。

### Part2. SageMaker Training を使って flan-t5 を Fine-tuning する
このパートでは、ノートブック(summerizing_medical_text_flan-t5_sagemaker_ja.ipynb)を使います。
SageMaker の HuggingFace コンテナを使用して、の flan-t5-base を学習します。

このノートブックでは、[SageMake Training](https://docs.aws.amazon.com/sagemaker/latest/dg/how-it-works-training.html) を使って、flan-t5-baseをトレーニングする方法が書かれています。

ノートブックを開くと、インスタンスとして ml.t3.medium を、カーネルには Data Science 3.0 を選択します。

SageMaker Training では、必要に応じてより大きなデータや flan-t5-large のようなモデルの学習が可能です。

## リファレンス
[1] MeQSum Dataset: "On the Summarization of Consumer Health Questions". Asma Ben Abacha and Dina Demner-Fushman. In Proceedings of the 57th Annual Meeting of the Association for Computational Linguistics, ACL 2019. (https://www.aclweb.org/anthology/P19-1215)  
@Inproceedings{MeQSum, author = {Asma {Ben Abacha} and Dina Demner-Fushman}, title = {On the Summarization of Consumer Health Questions}, booktitle = {Proceedings of the 57th Annual Meeting of the Association for Computational Linguistics, ACL 2019, Florence, Italy, July 28th - August 2}, year = {2019}, abstract = {Question understanding is one of the main challenges in question answering. In real world applications, users often submit natural language questions that are longer than needed and include peripheral information that increases the complexity of the question, leading to substantially more false positives in answer retrieval. In this paper, we study neural abstractive models for medical question summarization. We introduce the MeQSum corpus of 1,000 summarized consumer health questions. We explore data augmentation methods and evaluate state-of-the-art neural abstractive models on this new task. In particular, we show that semantic augmentation from question datasets improves the overall performance, and that pointer-generator networks outperform sequence-to-sequence attentional models on this task, with a ROUGE-1 score of 44.16%. We also present a detailed error analysis and discuss directions for improvement that are specific to question summarization. }}
