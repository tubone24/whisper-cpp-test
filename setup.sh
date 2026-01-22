#!/bin/bash
# whisper.cpp Real-time Transcription Setup Script
# Optimized for Apple Silicon (M1/M2/M3/M4)

set -e

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
WHISPER_DIR="$SCRIPT_DIR/whisper.cpp"
MODELS_DIR="$SCRIPT_DIR/models"

echo "=== whisper.cpp Real-time Transcription Setup ==="
echo ""

# Clone and build whisper.cpp
setup_whisper_cpp() {
    echo "[1/6] Setting up whisper.cpp..."

    if [ -d "$WHISPER_DIR" ]; then
        echo "whisper.cpp already cloned. Updating..."
        cd "$WHISPER_DIR"
        git pull
    else
        echo "Cloning whisper.cpp..."
        git clone https://github.com/ggerganov/whisper.cpp.git "$WHISPER_DIR"
        cd "$WHISPER_DIR"
    fi

    echo "Building whisper.cpp (Apple Silicon optimized)..."

    # SDL2 is required for stream command
    if ! brew list sdl2 &>/dev/null; then
        echo "Installing SDL2..."
        brew install sdl2
    fi

    # Clean build directory
    rm -rf build 2>/dev/null || true

    # Build with CMake (Apple Silicon + Metal GPU + SDL2 + examples)
    # Note: Core ML is disabled because it requires generating .mlmodelc files
    # Metal GPU acceleration is used instead (automatically enabled on Apple Silicon)
    cmake -B build \
        -DWHISPER_METAL=ON \
        -DWHISPER_SDL2=ON \
        -DWHISPER_BUILD_EXAMPLES=ON \
        -DCMAKE_BUILD_TYPE=Release

    cmake --build build -j$(sysctl -n hw.ncpu) --config Release

    echo ""
    echo "Built binaries:"
    ls -la build/bin/ 2>/dev/null || echo "  (no bin directory)"

    echo "whisper.cpp build complete!"
}

# Download models
download_models() {
    echo ""
    echo "[2/6] Downloading models..."

    mkdir -p "$MODELS_DIR"
    cd "$WHISPER_DIR"

    # Available models:
    # tiny.en, tiny, base.en, base, small.en, small, medium.en, medium, large-v1, large-v2, large-v3, large-v3-turbo
    # Quantized: large-v3-turbo-q8_0, large-v3-turbo-q5_0

    # Download large-v3-turbo model (Streaming最適化では量子化より元モデルが推奨)
    # Streaming処理はデコード部分がボトルネックのため、量子化による速度改善は限定的
    if [ ! -f "$MODELS_DIR/ggml-large-v3-turbo.bin" ]; then
        echo "Downloading ggml-large-v3-turbo model (recommended for streaming)..."
        bash models/download-ggml-model.sh large-v3-turbo
        cp models/ggml-large-v3-turbo.bin "$MODELS_DIR/"
    else
        echo "ggml-large-v3-turbo model already downloaded"
    fi

    # Also download base model (fallback, smaller)
    if [ ! -f "$MODELS_DIR/ggml-base.bin" ]; then
        echo "Downloading ggml-base model (fallback)..."
        bash models/download-ggml-model.sh base
        cp models/ggml-base.bin "$MODELS_DIR/"
    else
        echo "ggml-base model already downloaded"
    fi

    echo "Model download complete!"
}

# Setup Python environment using uv
setup_python() {
    echo ""
    echo "[3/6] Setting up Python environment (uv)..."

    cd "$SCRIPT_DIR"

    # Check if uv is installed
    if ! command -v uv &> /dev/null; then
        echo "uv is not installed. Installing..."
        curl -LsSf https://astral.sh/uv/install.sh | sh
        # Reload shell path
        export PATH="$HOME/.local/bin:$PATH"
    fi

    echo "Setting up project with uv..."
    uv sync

    echo "Python environment setup complete!"
}

# Setup textlint for proofreading (オプション、重いため非推奨)
# Python軽量校正がデフォルトで有効なため、textlintは通常不要
setup_textlint() {
    echo ""
    echo "[4/6] Skipping textlint setup (Python軽量校正を使用)..."
    echo "  textlintを使用したい場合は --with-textlint オプションを使用してください"
    echo "  例: ./setup.sh --with-textlint"
}

# textlintをインストールする関数（オプショナル）
setup_textlint_optional() {
    echo ""
    echo "[4/6] Setting up textlint (optional, 重い処理)..."

    # Check if Node.js is installed
    if ! command -v node &> /dev/null; then
        echo "Node.js is not installed. Installing via Homebrew..."
        if command -v brew &> /dev/null; then
            brew install node
        else
            echo "Warning: Homebrew not found. Please install Node.js manually."
            echo "  https://nodejs.org/"
            return
        fi
    fi

    # Check if textlint is installed globally
    if ! command -v textlint &> /dev/null; then
        echo "Installing textlint and Japanese rules..."
        npm install -g textlint textlint-rule-preset-ja-technical-writing textlint-rule-preset-japanese
    else
        echo "textlint already installed"
    fi

    # Create default config if not exists
    CONFIG_DIR="$HOME/.config/whisper-realtime"
    CONFIG_FILE="$CONFIG_DIR/.textlintrc.json"
    mkdir -p "$CONFIG_DIR"

    if [ ! -f "$CONFIG_FILE" ]; then
        echo "Creating default textlint config..."
        cat > "$CONFIG_FILE" << 'EOF'
{
  "rules": {
    "preset-ja-technical-writing": {
      "sentence-length": {
        "max": 100
      },
      "max-ten": {
        "max": 3
      },
      "no-doubled-joshi": {
        "strict": false
      }
    }
  }
}
EOF
    fi

    echo "textlint setup complete!"
}

# Build WhisperMenuBar
build_menubar_app() {
    echo ""
    echo "[5/6] Building WhisperMenuBar..."

    MENUBAR_DIR="$SCRIPT_DIR/WhisperMenuBar"

    if [ ! -d "$MENUBAR_DIR" ]; then
        echo "WhisperMenuBar directory not found. Skipping..."
        return
    fi

    cd "$MENUBAR_DIR"

    # Build using Swift Package Manager
    echo "Building with Swift Package Manager..."
    swift build -c debug

    if [ -f ".build/debug/WhisperMenuBar" ]; then
        echo "WhisperMenuBar build complete!"
        echo "  Binary: $MENUBAR_DIR/.build/debug/WhisperMenuBar"
    else
        echo "Warning: WhisperMenuBar build may have failed"
    fi
}

# Show audio setup information
show_audio_setup_info() {
    echo ""
    echo "[6/6] Audio Configuration"
    echo ""
    echo "■ Microphone Input:"
    echo "  → No additional setup required. Ready to use."
    echo ""
    echo "■ System Audio Capture (macOS):"
    echo ""
    echo "  [Method 1] ScreenCaptureKit (macOS 13+ recommended)"
    echo "    Install additional packages:"
    echo "    uv pip install -e '.[macos]'"
    echo ""
    echo "    Note: Screen Recording permission is required on first run"
    echo "    System Settings → Privacy & Security → Screen Recording"
    echo ""
    echo "  [Method 2] BlackHole (Virtual Audio Device)"
    echo "    brew install blackhole-2ch"
    echo ""
    echo "    After installation, open Audio MIDI Setup and create"
    echo "    a 'Multi-Output Device' with both speakers and BlackHole."
    echo ""
    echo "    Details: https://github.com/ExistentialAudio/BlackHole"
    echo ""
}

# Main function
main() {
    echo "Target directory: $SCRIPT_DIR"
    echo ""

    # Arguments to skip steps
    SKIP_WHISPER=false
    SKIP_MODELS=false
    SKIP_PYTHON=false
    SKIP_MENUBAR=false
    WITH_TEXTLINT=false

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
            --skip-menubar)
                SKIP_MENUBAR=true
                shift
                ;;
            --with-textlint)
                WITH_TEXTLINT=true
                shift
                ;;
            --model)
                EXTRA_MODEL="$2"
                shift 2
                ;;
            --help)
                echo "Usage: ./setup.sh [options]"
                echo ""
                echo "Options:"
                echo "  --skip-whisper    Skip whisper.cpp build"
                echo "  --skip-models     Skip model download"
                echo "  --skip-python     Skip Python environment setup"
                echo "  --skip-menubar    Skip WhisperMenuBar build"
                echo "  --with-textlint   Install textlint (optional, 重い処理)"
                echo "  --model <name>    Download additional model"
                echo "                    (tiny, base, small, medium, large-v3, large-v3-turbo,"
                echo "                     large-v3-turbo-q8_0, large-v3-turbo-q5_0)"
                echo "  --help            Show this help"
                exit 0
                ;;
            *)
                echo "Unknown option: $1"
                echo "Help: ./setup.sh --help"
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

    # Download additional model if specified
    if [ -n "$EXTRA_MODEL" ]; then
        echo "Downloading additional model ($EXTRA_MODEL)..."
        cd "$WHISPER_DIR"
        bash models/download-ggml-model.sh "$EXTRA_MODEL"
        cp "models/ggml-$EXTRA_MODEL.bin" "$MODELS_DIR/"
    fi

    if [ "$SKIP_PYTHON" = false ]; then
        setup_python
    fi

    # textlintはオプション（--with-textlintを指定した場合のみ）
    if [ "$WITH_TEXTLINT" = true ]; then
        setup_textlint_optional
    else
        setup_textlint
    fi

    if [ "$SKIP_MENUBAR" = false ]; then
        build_menubar_app
    fi

    show_audio_setup_info

    echo "=== Setup Complete ==="
    echo ""
    echo "Usage:"
    echo "  uv run whisper-realtime --help"
    echo ""
    echo "Quick start:"
    echo "  uv run whisper-realtime start           # Start with microphone"
    echo "  uv run whisper-realtime start -s system # Start with system audio"
    echo "  uv run whisper-realtime voice --menubar # Menubar mode (Python)"
    echo ""
    echo "WhisperMenuBar (Native Swift app):"
    echo "  ./WhisperMenuBar/.build/debug/WhisperMenuBar"
    echo ""
    echo "Models:"
    echo "  Default: large-v3-turbo (best for streaming with optimized params)"
    echo "  uv run whisper-realtime models          # List available models"
    echo ""
}

main "$@"
