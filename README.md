# whisper-realtime

whisper.cpp を使用したリアルタイム音声文字起こし CLI ツール。
Apple Silicon (M1/M2/M3) Mac 向けに最適化されています。

## 機能

- リアルタイム音声文字起こし（テキストが随時更新される体験）
- 複数の音声入力ソース対応
  - マイク入力
  - システム音声（BlackHole経由）
  - 両方同時
- モデル選択可能（リアルタイム性 vs 精度のトレードオフ）
- 話者分離機能（簡易版 / pyannote.audio連携）
- CLIベースのシンプルなインターフェース

## 必要環境

- macOS (Apple Silicon)
- Python 3.10+
- [uv](https://docs.astral.sh/uv/) (パッケージマネージャー)
- C++ コンパイラ (Xcode Command Line Tools)

## セットアップ

```bash
# リポジトリをクローン
git clone <repository-url>
cd whisper-cpp-test

# セットアップスクリプトを実行
chmod +x setup.sh
./setup.sh
```

セットアップスクリプトは以下を行います：
1. whisper.cpp のクローンとビルド（Apple Silicon最適化）
2. デフォルトモデル（tiny, base）のダウンロード
3. Python依存パッケージのインストール（uv使用）

### 追加モデルのダウンロード

```bash
# より高精度なモデルを追加
./setup.sh --skip-whisper --skip-python --model small
./setup.sh --skip-whisper --skip-python --model large-v3-turbo
```

### システム音声キャプチャの設定

システム音声（ブラウザの音声、Zoomの相手の声など）をキャプチャするには、BlackHoleが必要です：

```bash
brew install blackhole-2ch
```

インストール後：
1. 「Audio MIDI設定」を開く（Spotlight で検索）
2. 左下の「+」→「複数出力装置を作成」
3. スピーカーとBlackHole 2chの両方にチェック
4. この複数出力装置をシステムの出力デバイスに設定

## 使い方

### 基本コマンド

```bash
# ヘルプを表示
uv run whisper-realtime --help

# マイク入力でリアルタイム文字起こし開始
uv run whisper-realtime start

# システム音声をキャプチャ
uv run whisper-realtime start -s system

# マイクとシステム音声の両方
uv run whisper-realtime start -s both

# デバイス一覧を表示
uv run whisper-realtime devices

# 利用可能なモデル一覧
uv run whisper-realtime models
```

### 詳細オプション

```bash
# モデルを指定（リアルタイム性重視）
uv run whisper-realtime start -m tiny

# プロファイルを使用
uv run whisper-realtime start -p realtime  # 最速
uv run whisper-realtime start -p balanced  # バランス
uv run whisper-realtime start -p quality   # 高精度

# 言語を指定
uv run whisper-realtime start -l ja  # 日本語（デフォルト）
uv run whisper-realtime start -l en  # 英語
uv run whisper-realtime start -l auto  # 自動検出

# 話者分離を有効化
uv run whisper-realtime start --speaker

# 特定のデバイスを使用
uv run whisper-realtime start -d 2  # デバイスID 2

# 結果をファイルに保存
uv run whisper-realtime start -o output.txt

# 処理パラメータを調整
uv run whisper-realtime start --step 300 --length 3000
```

### whisper.cpp stream モード

whisper.cpp の stream バイナリを直接使用するモード：

```bash
uv run whisper-realtime stream -m base
```

## モデル一覧

| モデル | サイズ | 用途 |
|--------|--------|------|
| tiny | ~75MB | 最速、リアルタイム向け |
| base | ~142MB | バランス型（デフォルト） |
| small | ~466MB | 高精度 |
| medium | ~1.5GB | より高精度 |
| large-v3 | ~2.9GB | 最高精度 |
| large-v3-turbo | ~1.5GB | 高精度 + 高速 |

リアルタイム性が重要な場合は `tiny` または `base` を推奨。

## オプション詳細

| オプション | 説明 |
|------------|------|
| `-s, --source` | 音声ソース: `mic`, `system`, `both` |
| `-m, --model` | Whisperモデル名 |
| `-p, --profile` | プリセット: `realtime`, `balanced`, `quality`, `best` |
| `-l, --language` | 言語コード (ja, en, auto) |
| `-d, --device` | マイクデバイスID |
| `--system-device` | システム音声デバイスID |
| `--speaker/--no-speaker` | 話者分離の有効/無効 |
| `--translate/--no-translate` | 英語への翻訳 |
| `-o, --output` | 出力ファイルパス |
| `--step` | 処理ステップ（ミリ秒） |
| `--length` | 処理窓の長さ（ミリ秒） |
| `--vad/--no-vad` | 音声区間検出 |

## トラブルシューティング

### whisper.cpp のビルドに失敗する

Xcode Command Line Tools がインストールされているか確認：

```bash
xcode-select --install
```

### モデルが見つからない

```bash
./setup.sh --skip-whisper --skip-python
```

### BlackHoleが認識されない

Audio MIDI設定で複数出力装置が正しく設定されているか確認してください。

### 音声が認識されない

1. システム環境設定でマイクのアクセス許可を確認
2. ターミナルアプリにマイクアクセスを許可

## ライセンス

MIT License
