# Stable Diffusion を使ってデータセットのサンプルを生成する
衛星画像のサンプルを生成するために、Fine-tuning された Stable Diffusion を使用してみます。
SageMaker Studioでは、PyTorch 2.0 Python 3.10 GPU カーネル、g4dn.xlarge インスタンスでテストしています。

このチュートリアルでは、LoRA と Dreambooth を使って、テキストから画像への生成モデルである Stable Diffusion を Fine-tuning し、データセットを補強するための衛星画像を生成するところまで説明します。


# ユースケース
このラボでは、合成衛星画像を生成します。これらの画像は研究のために使われたり、画像認識モデルを作成するときの入力画像として利用されます。

# Stable Diffusion
Stable Diffusion を利用すると、簡単にテキストから画像を生成することができます。画像生成以外にも、別の画像とプロンプトに基づいて画像を生成したり（image to image）、インペインティング（画像の一部を編集）、アウトペインティング（画像を拡張）、アップスケーリング（画像の解像度を大きくする）などの機能を備えています。

## なぜ Stable Diffusion を Fine-tune するのか？
Stable Diffusion は画像生成において優れていますが、特定の分野に特化した画像の質はあまり高くないかもしれません。たとえば、このノートブックでは、衛星画像を生成しようとします。デフォルトで生成される衛生画像は、いくつかの特徴（高速道路など）をよく表していますが、高速道路を含む衛生画像の品質を向上させるために、実際の衛生画像を用いて Stable Diffusion を Fine-tuning します。

## How do we fine-tune
Stable Diffusion を Fine-tune するために、[こちら](https://dreambooth.github.io/) で説明のある DtreamBooth という方法を使います。