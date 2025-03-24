# Whisperモデルの自動音声認識のためのファインチューニングと評価

このリポジトリには、huggingface 🤗で利用可能な様々な設定のWhisperモデルのファインチューニングと評価に必要なスクリプトが含まれています。

このリポジトリのスクリプトは、huggingfaceで利用できないカスタムデータセットを使用してモデルのファインチューニングをサポートしています。
これらのスクリプトを使用して学習・評価されたモデルの一部は[huggingfaceで確認できます](https://huggingface.co/vasista22)。

また、異なる設定のwhisperモデルの各層から関連する埋め込みを抽出するためのコードスニペットも提供されています。

## 目次

- [セットアップ](#セットアップ)
- [カスタムデータセットのデータ準備](#カスタムデータセットのデータ準備)
- [ハイパーパラメータの調整](#ハイパーパラメータの調整)
- [huggingfaceのデータセットでのファインチューニング](#huggingfaceのデータセットでのファインチューニング)
- [カスタムデータセットでのファインチューニング](#カスタムデータセットでのファインチューニング)
- [huggingfaceのデータセットでの評価](#huggingfaceのデータセットでの評価)
- [カスタムデータセットでの評価](#カスタムデータセットでの評価)
- [単一の音声ファイルの文字起こし](#単一の音声ファイルの文字起こし)
- [whisper-jaxを使用した高速評価](#whisper-jaxを使用した高速評価)
- [whisperモデルからの埋め込みの抽出](#whisperモデルからの埋め込みの抽出)
- [Whisperに関する興味深い研究](#whisperに関する興味深い研究)

## セットアップ

これらのスクリプトはPython 3.11とCUDA 12.4でテストされています。

インストール目的で仮想環境をセットアップし、その中で作業することをお勧めします。以下のコマンドセットで仮想環境のセットアップとインストールが完了します：

```bash
python3.11 -m venv env_whisper-finetune
source env_whisper-finetune/bin/activate

python3 -m pip install --upgrade pip
pip install -r requirements.txt
```

huggingfaceにモデルをプッシュするには、コマンドラインインターフェースを使用してログインする必要があります。また、大きなモデルファイルをプッシュするには`git-lfs`のインストールが必要です。以下のコマンドを実行することで対応できます：

```bash
sudo apt-get install git-lfs
huggingface-cli login
```

## カスタムデータセットのデータ準備

**注意:** このセクションの内容は、huggingfaceで利用できないデータセットを使用する場合にのみ関連します。該当しない場合は、READMEの後半のセクションに進んでください。

huggingfaceで利用できないデータセットで作業したい場合があります。
そのようなデータセットでwhisperモデルのファインチューニングや評価を行うには、huggingfaceのシーケンス間学習パイプラインと互換性のある形式に変換するための予備的なデータ準備が必要です。

データセットを必要な形式に変換するスクリプトは、`text`と`audio_paths`という2つのファイルを期待します。

`audio_paths`ファイルには、ファインチューニングや評価プロセスで使用する各音声ファイルへの絶対パスが含まれている必要があります。また、ファイル内の各エントリは一意の発話IDでインデックス付けされている必要があります。ファイルの内容は以下のように整理する必要があります：

```bash
<unique-id> <absolute path to the audio file-1>
<unique-id> <absolute path to the audio file-2>
...
<unique-id> <absolute path to the audio file-N>
```

`text`ファイルには、`audio_paths`ファイルで言及されている各音声ファイルに対応する文字起こしが含まれている必要があります。また、ファイル内の各エントリは一意の発話IDでインデックス付けされている必要があります。`text`と`audio_paths`ファイルの一意の発話IDの順序は一致している必要があります。`text`ファイルの内容は以下のように整理する必要があります：

```bash
<unique-id> <Transcription (ground truth) corresponding to the audio file-1>
<unique-id> <Transcription (ground truth) corresponding to the audio file-2>
...
<unique-id> <Transcription (ground truth) corresponding to the audio file-N>
```

このリポジトリの`sample_data`フォルダは、これらの2つのファイルをどのように整理すべきかの参考例を提供しています。

データが上記の方法で整理されたら、`custom_data/data_prep.py`というスクリプトを使用して、huggingfaceのシーケンス間パイプラインが期待する形式にデータを変換できます。

以下は、データを目的の形式に変換するためのサンプルコマンドです：

```bash
# source_data_directoryは`text`と`audio_paths`ファイルを含むディレクトリへのパス
# output_data_directoryはフォーマットされたデータが保存される場所

python3 custom_data/data_prep.py \
--source_data_dir source_data_directory \
--output_data_dir output_data_directory
```

使用方法の詳細については`python3 custom_data/data_prep.py -h`コマンドを使用してください。

## ハイパーパラメータの調整

学習率は、特にWhisperのような大量のデータで事前学習されたモデルを適応/ファインチューニングする際の最も重要なハイパーパラメータの1つです。

Whisper論文の著者の1人であるJong Wook Kimによると、ファインチューニング時に考慮すべき実用的な学習率は、事前学習で使用された値の40分の1の値で、学習期間中に線形的にゼロに減衰させることです。（[この発言がなされたDiscordスレッド](https://discord.com/channels/879548962464493619/1050020275250548836/1050369079111856168)）

以下の表は、異なるモデル設定のファインチューニング実験のための推奨学習率を示しています：

| モデルサイズ | 最大学習率（論文） | 推奨ファインチューニング学習率（40分の1） |
|   :---:    |           :---:           |                      :---:                        |
|   tiny     |      $1.5$ x $10^{-3}$    |                  $3.75$ x $10^{-5}$               |
|   base     |      $1$ x $10^{-3}$      |                  $2.5$ x $10^{-5}$                |
|   small    |      $5$ x $10^{-4}$      |                  $1.25$ x $10^{-5}$               |
|   medium   |      $2.5$ x $10^{-4}$    |                  $6.25$ x $10^{-6}$               |
|   large    |      $1.75$ x $10^{-4}$   |                  $4.375$ x $10^{-6}$              |
|   large-v2 |      $2.0$ x $10^{-4}$    |                  $5$ x $10^{-6}$                  |

## huggingfaceのデータセットでのファインチューニング

huggingfaceで利用可能なデータセットでWhisperモデルのファインチューニングを行うには、`train/fine-tune_on_hf_dataset.py`ファイルを使用できます。

以下は実行のサンプルコマンドです：

```bash
ngpu=4  # 分散学習に使用するGPUの数

torchrun --nproc_per_node=${ngpu} train/fine-tune_on_hf_dataset.py \
--model_name vasista22/whisper-hindi-base \
--language Hindi \
--sampling_rate 16000 \
--num_proc 2 \
--train_strategy steps \
--learning_rate 3e-3 \
--warmup 1000 \
--train_batchsize 16 \
--eval_batchsize 8 \
--num_steps 10000 \
--resume_from_ckpt None \
--output_dir op_dir_steps \
--train_datasets mozilla-foundation/common_voice_11_0 mozilla-foundation/common_voice_11_0 \
--train_dataset_configs hi hi \
--train_dataset_splits train validation \
--train_dataset_text_columns sentence sentence \
--eval_datasets "google/fleurs" \
--eval_dataset_configs hi_in \
--eval_dataset_splits test \
--eval_dataset_text_columns transcription
```

ファインチューニングプロセスの一部として複数のデータセットを使用できます。これらのデータセットは、データセット準備時に連結されシャッフルされます。
`train_datasets`、`train_dataset_configs`、`train_dataset_splits`、`train_dataset_text_columns`引数を介して渡されるパラメータの数は同じである必要があり、これらの引数間のパラメータの順序は一致している必要があることに注意してください。同様のことが`eval_datasets`、`eval_dataset_configs`、`eval_dataset_splits`、`eval_dataset_text_columns`引数にも当てはまります。

使用方法の詳細については`python3 train/fine-tune_on_hf_dataset.py -h`コマンドを使用してください。

すべての引数はデフォルトオプションで設定されていますが、利用可能なデータ量と使用するモデルのサイズに合わせて学習ハイパーパラメータをカスタマイズすることをお勧めします。

## カスタムデータセットでのファインチューニング

カスタムデータセットでWhisperモデルのファインチューニングを行うには、`train/fine-tune_on_custom_dataset.py`ファイルを使用できます。

以下は実行のサンプルコマンドです：

```bash
ngpu=4  # 分散学習に使用するGPUの数

torchrun --nproc_per_node=${ngpu} train/fine-tune_on_custom_dataset.py \
--model_name vasista22/whisper-telugu-base \
--language Telugu \
--sampling_rate 16000 \
--num_proc 2 \
--train_strategy epoch \
--learning_rate 3e-3 \
--warmup 1000 \
--train_batchsize 16 \
--eval_batchsize 8 \
--num_epochs 20 \
--resume_from_ckpt None \
--output_dir op_dir_epoch \
--train_datasets output_data_directory/train_dataset_1 output_data_directory/train_dataset_2 \
--eval_datasets output_data_directory/eval_dataset_1 output_data_directory/eval_dataset_2 output_data_directory/eval_dataset_3
```

`train_datasets`と`eval_datasets`引数を介してパラメータとして渡されるデータセットは、データ準備段階で生成された出力ディレクトリからのものである必要があります。
ファインチューニングプロセスの一部として複数のデータセットを使用できます。これらのデータセットは、データセット準備時に連結されシャッフルされます。

使用方法の詳細については`python3 train/fine-tune_on_custom_dataset.py -h`コマンドを使用してください。

すべての引数はデフォルトオプションで設定されていますが、利用可能なデータ量と使用するモデルのサイズに合わせて学習ハイパーパラメータをカスタマイズすることをお勧めします。

## huggingfaceのデータセットでの評価

`evaluate/evaluate_on_hf_dataset.py`ファイルを使用して、huggingfaceで利用可能なデータセットでモデルを評価できます。評価対象のモデルは、huggingfaceのWhisperモデルまたはファインチューニング段階で生成されたローカルのWhisperチェックポイントのいずれかです。

以下は実行のサンプルコマンドです：

```bash
python3 evaluate/evaluate_on_hf_dataset.py \
--is_public_repo False \
--ckpt_dir "op_dir_epoch/checkpoint-394" \
--temp_ckpt_folder "temp" \
--language gu \
--dataset "google/fleurs" \
--config gu_in \
--split test \
--device 0 \
--batch_size 16 \
--output_dir predictions_dir
```

`is_public_repo`引数はブール値を取り、評価対象のモデルがhuggingfaceのモデルかローカルチェックポイントかを指定します。上記のコマンドはローカルチェックポイントをhuggingfaceのデータセットで評価します。また、`ckpt_dir`と`temp_ckpt_folder`引数はローカルチェックポイントを評価する場合にのみ関連します。

huggingfaceのモデルを評価するには、`is_public_repo`を`True`に設定し、モデルIDを`hf_model`引数で渡す必要があります。以下は実行のサンプルコマンドです：

```bash
python3 evaluate/evaluate_on_hf_dataset.py \
--is_public_repo True \
--hf_model vasista22/whisper-kannada-small \
--language kn \
--dataset "google/fleurs" \
--config kn_in \
--split test \
--device 0 \
--batch_size 16 \
--output_dir predictions_dir
```

実行が成功すると、`--output_dir`にはデータセットごとに1つの結果ファイルが含まれ、各ファイルには単語エラー率と文字エラー率の結果、およびデータセット内の各発話の参照（REF）とモデルが生成した仮説（HYP）が含まれます。これらの結果ファイルは、モデルの名前と評価対象のデータセットの名前に基づいて命名されます。

使用方法の詳細については`python3 evaluate/evaluate_on_hf_dataset.py -h`コマンドを使用してください。

すべての引数はデフォルトオプションで設定されていますが、引数をカスタマイズすることをお勧めします。例えば、CPU推論では`device`引数を`-1`に設定する必要があります。

## カスタムデータセットでの評価

`evaluate/evaluate_on_custom_dataset.py`ファイルを使用して、上記のデータ準備段階で準備したカスタムデータセットでモデルを評価できます。評価対象のモデルは、huggingfaceのWhisperモデルまたはファインチューニング段階で生成されたローカルのWhisperチェックポイントのいずれかです。

以下は実行のサンプルコマンドです：

```bash
python3 evaluate/evaluate_on_custom_dataset.py \
--is_public_repo False \
--ckpt_dir "op_dir_epoch/checkpoint-394" \
--temp_ckpt_folder "temp" \
--language gu \
--eval_datasets output_data_directory/eval_dataset_1 output_data_directory/eval_dataset_2 \
--device 0 \
--batch_size 16 \
--output_dir predictions_dir
```

モデルは複数のデータセットで評価でき、上記のコマンドのように渡すことができます。各データセットの結果は`--output_dir`内の個別のファイルに保存されます。

`is_public_repo`引数はブール値を取り、評価対象のモデルがhuggingfaceのモデルかローカルチェックポイントかを指定します。上記のコマンドはローカルチェックポイントをhuggingfaceのデータセットで評価します。また、`ckpt_dir`と`temp_ckpt_folder`引数はローカルチェックポイントを評価する場合にのみ関連します。

huggingfaceのモデルを評価するには、`is_public_repo`を`True`に設定し、モデルIDを`hf_model`引数で渡す必要があります。以下は実行のサンプルコマンドです：

```bash
python3 evaluate/evaluate_on_custom_dataset.py \
--is_public_repo True \
--hf_model vasista22/whisper-kannada-small \
--language kn \
--eval_datasets output_data_directory/eval_dataset_1 output_data_directory/eval_dataset_2 \
--device 0 \
--batch_size 16 \
--output_dir predictions_dir
```

実行が成功すると、`--output_dir`にはデータセットごとに1つの結果ファイルが含まれ、各ファイルには単語エラー率と文字エラー率の結果、およびデータセット内の各発話の参照（REF）とモデルが生成した仮説（HYP）が含まれます。これらの結果ファイルは、モデルの名前と評価対象のデータセットの名前に基づいて命名されます。

使用方法の詳細については`python3 evaluate/evaluate_on_custom_dataset.py -h`コマンドを使用してください。

すべての引数はデフォルトオプションで設定されていますが、引数をカスタマイズすることをお勧めします。例えば、CPU推論では`device`引数を`-1`に設定する必要があります。

## 単一の音声ファイルの文字起こし

`transcribe_audio.py`ファイルを使用して、単一の音声ファイルの文字起こしを取得できます。文字起こしに使用するモデルは、huggingfaceのWhisperモデルまたはファインチューニング段階で生成されたローカルのWhisperチェックポイントのいずれかです。

以下は実行のサンプルコマンドです：

```bash
python3 transcribe_audio.py \
--is_public_repo False \
--ckpt_dir "op_dir_epoch/checkpoint-1254" \
--temp_ckpt_folder "temp" \
--path_to_audio /path/to/audio/file.wav \
--language ta \
--device 0
```

`is_public_repo`引数はブール値を取り、使用するモデルがhuggingfaceのモデルかローカルチェックポイントかを指定します。上記のコマンドはローカルチェックポイントを使用して音声を文字起こしします。また、`ckpt_dir`と`temp_ckpt_folder`引数はローカルチェックポイントを使用する場合にのみ関連します。

huggingfaceのモデルを使用するには、`is_public_repo`を`True`に設定し、モデルIDを`hf_model`引数で渡す必要があります。以下は実行のサンプルコマンドです：

```bash
python3 transcribe_audio.py \
--is_public_repo True \
--hf_model vasista22/whisper-tamil-base \
--path_to_audio /path/to/audio/file.wav \
--language ta \
--device 0
```

使用方法の詳細については`python3 transcribe_audio.py -h`コマンドを使用してください。

ほとんどの引数はデフォルトオプションで設定されていますが、引数をカスタマイズすることをお勧めします。例えば、CPU推論では`device`引数を`-1`に設定する必要があります。

## whisper-jaxを使用した高速評価

[whisper-jax](https://github.com/sanchit-gandhi/whisper-jax)はwhisperモデルの推論を高速化するのに役立ちます。`evaluate/jax_evaluate_on_hf_dataset.py`と`evaluate/jax_evaluate_on_custom_dataset.py`ファイルは、whisper-jaxを使用してhuggingfaceのデータセットとカスタムデータセットでの評価を高速化します。

この高速評価を使用するには、[whisper-jax](https://github.com/sanchit-gandhi/whisper-jax)リポジトリで提案されている必要な依存関係をインストールしてください。CUDA 11を使用している場合、以下のコマンドで安全にインストールを完了できます：

```bash
pip install --upgrade pip
pip install --upgrade "jax[cpu]"  # jaxのCPUインストール
pip install --upgrade "jax[cuda11_pip]" -f https://storage.googleapis.com/jax-releases/jax_cuda_releases.html  # GPU用のjax

pip install git+https://github.com/sanchit-gandhi/whisper-jax.git
```

**注意:** whisper-jaxは、huggingfaceでflaxの重みも利用可能なモデルでのみ使用できます。既存のモデルのflax重みをプッシュするには、[ここ](https://github.com/sanchit-gandhi/whisper-jax#available-models-and-languages)の指示に従ってください。

以下は、huggingfaceのデータセットでモデルを評価するサンプルコマンドです：

```bash
python3 evaluate/jax_evaluate_on_hf_dataset.py \
--hf_model vasista22/whisper-telugu-small \
--language te \
--dataset "google/fleurs" \
--config te_in \
--split test \
--device 0 \
--batch_size 16 \
--output_dir jax_predictions_dir \
--half_precision True
```

同様に、以下はカスタムデータセットでモデルを評価するサンプルコマンドです：

```bash
python3 evaluate/jax_evaluate_on_custom_dataset.py \
--hf_model openai/whisper-base \
--language hi \
--eval_datasets output_data_directory/eval_dataset_1 output_data_directory/eval_dataset_2 \
--device 0 \
--batch_size 16 \
--output_dir jax_predictions_dir \
--half_precision True
```

`--half_precision`引数を`True`に設定することで、モデルの計算を半精度で実行できます。これにより、計算をさらに高速化できます。

whisper-jaxを使用して推論を実行する際に、`Failed to determine best cudnn convolution algorithm/No GPU/TPU found`というエラーメッセージが表示された場合、[提案されている](https://github.com/google/jax/issues/8746#issuecomment-1327919319)解決策は以下のコマンドをエクスポートすることです：

```bash
export XLA_PYTHON_CLIENT_PREALLOCATE=false
export XLA_FLAGS="--xla_gpu_strict_conv_algorithm_picker=false --xla_gpu_force_compilation_parallelism=1"
```

単一の音声ファイルの文字起こしにwhisper-jaxを使用するには、`jax_transcribe_audio.py`コマンドを使用できます。以下は使用例です：

```bash
python3 jax_transcribe_audio.py \
--hf_model vasista22/whisper-tamil-base \
--path_to_audio /path/to/audio/file.wav \
--language ta \
--device 0 \
--half_precision True \
--batch_size 16
```

## whisperモデルからの埋め込みの抽出

Whisperモデル（オリジナル/ファインチューニング済み）は大量の音声データで学習されているため、これらのモデルの埋め込みは自動音声認識（ASR）以外の音声の下流タスクにも使用できます。

以下の表は、異なるモデルサイズのエンコーダーとデコーダーの埋め込みの次元を示しています：

| モデルサイズ | 埋め込み次元 | レイヤー数 |
|   :---:    |        :---:        |       :---:      |
|   tiny     |        384          |         4        |
|   base     |        512          |         6        |
|   small    |        768          |        12        |
|   medium   |        1024         |        24        |
|   large    |        1280         |        32        |
|   large-v2 |        1280         |        32        |

whisper Seq2Seqモデル出力から利用可能な異なる埋め込みは以下の通りです：

- `encoder_last_hidden_state` - レイヤー正規化後のエンコーダーの最後のレイヤーの出力
- `encoder_hidden_states` - エンコーダーの各レイヤーからの埋め込みのリスト。例えば、whisper tinyモデルではこのリストに5つの埋め込みがあります。このリストのインデックス0から3は、エンコーダーのレイヤー1から4の埋め込みです。このリストのインデックス4（5番目の埋め込み）は`encoder_last_hidden_state`と同じです。つまり、レイヤー正規化が適用された最終エンコーダーレイヤーの埋め込みに対応します。
- `last_hidden_state` - レイヤー正規化後のデコーダーの最後のレイヤーの出力
- `decoder_hidden_states` - デコーダーの各レイヤーからの埋め込みのリスト。例えば、whisper tinyモデルではこのリストに5つの埋め込みがあります。このリストのインデックス0から3は、デコーダーのレイヤー1から4の埋め込みです。このリストのインデックス4（5番目の埋め込み）は`last_hidden_state`と同じです。つまり、レイヤー正規化が適用された最終デコーダーレイヤーの埋め込みに対応します。

エンコーダーからの埋め込みは、話者検証、話者ダイアライゼーション、音声強調などの、話者関連の情報がより関連する下流タスクに使用できます。

キーワードスポッティング、音素認識などの、データの意味論により関連する下流タスクについては、デコーダーからの埋め込みがより適している可能性があります。

以下のコードスニペットを使用して、上記で説明した異なる埋め込みを抽出できます。

**注意:**
- 渡される音声セグメントが30秒を超えないようにしてください。これは、whisperの位置埋め込みなどが30秒以下の音声セグメントを処理するように設計されているためです。より長い音声の特徴は切り捨てられ、より短い音声の特徴はパディングされます。[WhisperConfig](https://github.com/huggingface/transformers/blob/main/src/transformers/models/whisper/configuration_whisper.py#L62)クラスは`max_source_positions`引数の定義で、`1500`が「このモデルが使用される可能性のあるログメルフィルターバンク特徴の最大シーケンス長」であることを指定しています。これは時間の観点から30秒に相当します。
- 任意のレイヤーの埋め込みの平均を使用して、その特定のレイヤーの出力を単一の埋め込みで表現できます。

```python

import torch
from datasets import Dataset, Audio
from transformers import AutoFeatureExtractor, WhisperModel

audio_segment_path="/path/to/the/audio_file"  # 音声セグメントへのパスをここに渡す（<= 30秒）

model = WhisperModel.from_pretrained("vasista22/whisper-kannada-small")  # 使用するモデルIDはここで変更可能
feature_extractor = AutoFeatureExtractor.from_pretrained("vasista22/whisper-kannada-small")  # 使用するモデルIDはここで変更可能
model.eval()

# 音声セグメントの特徴を抽出するための疑似データセットを作成
audio_read = Dataset.from_dict({"audio": [audio_segment_path]}).cast_column("audio", Audio(sampling_rate=16_000))
inputs = feature_extractor(audio_read['audio'][0]['array'], sampling_rate=16_000, return_tensors="pt")
input_features = inputs.input_features

model.config.output_hidden_states=True  # 個々のレイヤーの埋め込みを取得するため
decoder_input_ids = torch.tensor([[1, 1]]) * model.config.decoder_start_token_id

whisper_embeddings = model(input_features, decoder_input_ids=decoder_input_ids)

print('\n レイヤー正規化後のwhisperエンコーダーの最後のレイヤーの埋め込み: ', whisper_embeddings.encoder_last_hidden_state)
print('\n レイヤー正規化後のwhisperエンコーダーの最後のレイヤーの埋め込みの平均: ', torch.mean(whisper_embeddings.encoder_last_hidden_state, dim=1))
print('\n 8番目のエンコーダーレイヤーの埋め込み: ', whisper_embeddings.encoder_hidden_states[7])
print('\n 8番目のエンコーダーレイヤーの埋め込みの平均: ', torch.mean(whisper_embeddings.encoder_hidden_states[7], dim=1))
print('\n レイヤー正規化後のwhisperデコーダーの最後のレイヤーの埋め込み: ', whisper_embeddings.last_hidden_state)
print('\n レイヤー正規化後のwhisperデコーダーの最後のレイヤーの埋め込みの平均: ', torch.mean(whisper_embeddings.last_hidden_state, dim=1))
print('\n 8番目のデコーダーレイヤーの埋め込み: ', whisper_embeddings.decoder_hidden_states[7])
print('\n 8番目のデコーダーレイヤーの埋め込みの平均: ', torch.mean(whisper_embeddings.decoder_hidden_states[7], dim=1))

```

## Whisperに関する興味深い研究

OpenAIからWhisperモデルとコードがリリースされて以来、これらのモデルの機能を引き出し強化するためのいくつかの開発が行われています。以下は研究者や開発者にとって潜在的に有用となる可能性のある研究の一部です：

- 効率的な推論
    - [whisper-jax](https://github.com/sanchit-gandhi/whisper-jax)
    - [faster-whisper](https://github.com/guillaumekln/faster-whisper)
    - [whisper.cpp](https://github.com/ggerganov/whisper.cpp)
- 正確なタイムスタンプ
    - [whisper-timestamped](https://github.com/linto-ai/whisper-timestamped)
    - [stable-ts](https://github.com/jianfch/stable-ts)
- 外部の音素ベースASRモデルを使用した強制アライメント
    - [whisperX](https://github.com/m-bain/whisperX)
- パラメータ効率的なファインチューニング
    - [fast-whisper-finetuning](https://github.com/Vaibhavs10/fast-whisper-finetuning) 