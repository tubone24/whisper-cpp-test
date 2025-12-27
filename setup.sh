#!/bin/bash
# whisper.cpp リアルタイム文字起こしセットアップスクリプト
# Apple Silicon (M1/M2/M3) 向け最適化

set -e

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
WHISPER_DIR="$SCRIPT_DIR/whisper.cpp"
MODELS_DIR="$SCRIPT_DIR/models"

echo "=== whisper.cpp リアルタイム文字起こし セットアップ ==="
echo ""

# whisper.cppのクローンとビルド
setup_whisper_cpp() {
    echo "[1/4] whisper.cpp のセットアップ..."

    if [ -d "$WHISPER_DIR" ]; then
        echo "whisper.cpp は既にクローン済みです。更新します..."
        cd "$WHISPER_DIR"
        git pull
    else
        echo "whisper.cpp をクローン中..."
        git clone https://github.com/ggerganov/whisper.cpp.git "$WHISPER_DIR"
        cd "$WHISPER_DIR"
    fi

    echo "whisper.cpp をビルド中 (Apple Silicon 最適化)..."
    # Apple Silicon向けに最適化してビルド
    make clean 2>/dev/null || true

    # mainとstreamの両方をビルド
    WHISPER_COREML=1 make -j$(sysctl -n hw.ncpu) main stream

    echo "whisper.cpp のビルド完了!"
}

# モデルのダウンロード
download_models() {
    echo ""
    echo "[2/4] モデルのダウンロード..."

    mkdir -p "$MODELS_DIR"
    cd "$WHISPER_DIR"

    # 利用可能なモデル一覧
    # tiny.en, tiny, base.en, base, small.en, small, medium.en, medium, large-v1, large-v2, large-v3, large-v3-turbo

    # デフォルトでbaseモデルをダウンロード（リアルタイム向けバランス型）
    if [ ! -f "$MODELS_DIR/ggml-base.bin" ]; then
        echo "ggml-base モデルをダウンロード中..."
        bash models/download-ggml-model.sh base
        cp models/ggml-base.bin "$MODELS_DIR/"
    else
        echo "ggml-base モデルは既にダウンロード済みです"
    fi

    # tinyモデルもダウンロード（最速、リアルタイム重視の場合）
    if [ ! -f "$MODELS_DIR/ggml-tiny.bin" ]; then
        echo "ggml-tiny モデルをダウンロード中..."
        bash models/download-ggml-model.sh tiny
        cp models/ggml-tiny.bin "$MODELS_DIR/"
    else
        echo "ggml-tiny モデルは既にダウンロード済みです"
    fi

    echo "モデルのダウンロード完了!"
}

# Python環境のセットアップ (uv使用)
setup_python() {
    echo ""
    echo "[3/4] Python環境のセットアップ (uv)..."

    cd "$SCRIPT_DIR"

    # uvがインストールされているか確認
    if ! command -v uv &> /dev/null; then
        echo "uv がインストールされていません。インストール中..."
        curl -LsSf https://astral.sh/uv/install.sh | sh
        # シェルを再読み込み
        export PATH="$HOME/.local/bin:$PATH"
    fi

    echo "uv でプロジェクトをセットアップ中..."
    uv sync

    echo "Python環境のセットアップ完了!"
}

# システム音声キャプチャの説明
show_audio_setup_info() {
    echo ""
    echo "[4/4] オーディオ設定について"
    echo ""
    echo "■ マイク入力:"
    echo "  → 追加設定不要。そのまま使用できます。"
    echo ""
    echo "■ システム音声キャプチャ (macOS):"
    echo "  BlackHole (仮想オーディオデバイス) のインストールが必要です:"
    echo ""
    echo "  brew install blackhole-2ch"
    echo ""
    echo "  インストール後、Audio MIDI設定で「複数出力装置」を作成し、"
    echo "  スピーカーとBlackHoleを両方追加してください。"
    echo ""
    echo "  詳細: https://github.com/ExistentialAudio/BlackHole"
    echo ""
}

# メイン処理
main() {
    echo "対象ディレクトリ: $SCRIPT_DIR"
    echo ""

    # 引数でスキップするステップを指定可能
    SKIP_WHISPER=false
    SKIP_MODELS=false
    SKIP_PYTHON=false

    while [[ $# -gt 0 ]]; do
        case $1 in
            --skip-whisper)
                SKIP_WHISPER=true
                shift
                ;;
            --skip-models)
                SKIP_MODELS=true
                shift
                ;;
            --skip-python)
                SKIP_PYTHON=true
                shift
                ;;
            --model)
                EXTRA_MODEL="$2"
                shift 2
                ;;
            --help)
                echo "使い方: ./setup.sh [オプション]"
                echo ""
                echo "オプション:"
                echo "  --skip-whisper    whisper.cpp のビルドをスキップ"
                echo "  --skip-models     モデルのダウンロードをスキップ"
                echo "  --skip-python     Python環境のセットアップをスキップ"
                echo "  --model <name>    追加モデルをダウンロード"
                echo "                    (tiny, base, small, medium, large-v3, large-v3-turbo)"
                echo "  --help            このヘルプを表示"
                exit 0
                ;;
            *)
                echo "不明なオプション: $1"
                echo "ヘルプ: ./setup.sh --help"
                exit 1
                ;;
        esac
    done

    if [ "$SKIP_WHISPER" = false ]; then
        setup_whisper_cpp
    fi

    if [ "$SKIP_MODELS" = false ]; then
        download_models
    fi

    # 追加モデルのダウンロード
    if [ -n "$EXTRA_MODEL" ]; then
        echo "追加モデル ($EXTRA_MODEL) をダウンロード中..."
        cd "$WHISPER_DIR"
        bash models/download-ggml-model.sh "$EXTRA_MODEL"
        cp "models/ggml-$EXTRA_MODEL.bin" "$MODELS_DIR/"
    fi

    if [ "$SKIP_PYTHON" = false ]; then
        setup_python
    fi

    show_audio_setup_info

    echo "=== セットアップ完了 ==="
    echo ""
    echo "使用方法:"
    echo "  uv run whisper-realtime --help"
    echo ""
    echo "または:"
    echo "  uv run whisper-realtime start           # マイク入力で開始"
    echo "  uv run whisper-realtime start -s system # システム音声で開始"
    echo "  uv run whisper-realtime devices         # デバイス一覧"
    echo "  uv run whisper-realtime models          # モデル一覧"
    echo ""
}

main "$@"
