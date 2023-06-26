# Stable Diffusion を使ってデータセットのサンプルを生成する
衛星画像のサンプルを生成するために、Fine-tuning された Stable Diffusion を使用してみます。
SageMaker Studioでは、PyTorch 2.0 Python 3.10 GPU カーネル、g4dn.xlarge インスタンスでテストしています。

このチュートリアルでは、LoRA と Dreambooth を使って、テキストから画像への生成モデルである Stable Diffusion を Fine-tuning し、データセットを補強するための衛星画像を生成するところまで説明します。