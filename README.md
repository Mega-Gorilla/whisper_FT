# Whisper Fine-tuning Docker Project

このプロジェクトは、[vasistalodagala/whisper-finetune](https://github.com/vasistalodagala/whisper-finetune)をベースに、Docker化したWhisperモデルのファインチューニング環境を提供します。

## 概要

このプロジェクトは、Whisperモデルのファインチューニングと評価を行うためのDocker化された環境を提供します。元のプロジェクトの機能を保持しながら、Dockerコンテナ内で簡単に実行できるように環境を整備しています。

## プロジェクト構成

```
.
├── Dockerfile          # Dockerイメージの定義
├── docker-compose.yml  # Docker Compose設定
├── .devcontainer/      # VSCode DevContainer設定
├── .github/           # GitHub関連設定
├── data/              # データディレクトリ
├── models/            # モデル保存ディレクトリ
├── output/            # 出力ディレクトリ
└── whisper-finetune/  # 元のwhisper-finetuneリポジトリ（サブモジュール）
```

## セットアップ

### 必要条件

- Docker
- Docker Compose
- NVIDIA GPU (推奨)

### 環境構築

1. リポジトリのクローン:
```bash
git clone [your-repository-url]
cd whisper_FT
```

2. Dockerイメージのビルドと起動:
```bash
docker-compose up -d
```

## 使用方法

詳細な使用方法については、[whisper-finetune](https://github.com/vasistalodagala/whisper-finetune)のREADMEを参照してください。

主な機能：

- カスタムデータセットでのファインチューニング
- Hugging Faceデータセットでのファインチューニング
- モデルの評価
- 音声ファイルの文字起こし
- whisper-jaxを使用した高速な評価

## ライセンス

このプロジェクトは、[vasistalodagala/whisper-finetune](https://github.com/vasistalodagala/whisper-finetune)のMITライセンスに基づいています。 