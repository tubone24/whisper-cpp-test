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
    echo "[1/4] Setting up whisper.cpp..."

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
    echo "[2/4] Downloading models..."

    mkdir -p "$MODELS_DIR"
    cd "$WHISPER_DIR"

    # Available models:
    # tiny.en, tiny, base.en, base, small.en, small, medium.en, medium, large-v1, large-v2, large-v3, large-v3-turbo

    # Download base model by default (balanced for real-time)
    if [ ! -f "$MODELS_DIR/ggml-base.bin" ]; then
        echo "Downloading ggml-base model..."
        bash models/download-ggml-model.sh base
        cp models/ggml-base.bin "$MODELS_DIR/"
    else
        echo "ggml-base model already downloaded"
    fi

    # Also download tiny model (fastest, for real-time priority)
    if [ ! -f "$MODELS_DIR/ggml-tiny.bin" ]; then
        echo "Downloading ggml-tiny model..."
        bash models/download-ggml-model.sh tiny
        cp models/ggml-tiny.bin "$MODELS_DIR/"
    else
        echo "ggml-tiny model already downloaded"
    fi

    echo "Model download complete!"
}

# Setup Python environment using uv
setup_python() {
    echo ""
    echo "[3/4] Setting up Python environment (uv)..."

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

# Show audio setup information
show_audio_setup_info() {
    echo ""
    echo "[4/4] Audio Configuration"
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
                echo "Usage: ./setup.sh [options]"
                echo ""
                echo "Options:"
                echo "  --skip-whisper    Skip whisper.cpp build"
                echo "  --skip-models     Skip model download"
                echo "  --skip-python     Skip Python environment setup"
                echo "  --model <name>    Download additional model"
                echo "                    (tiny, base, small, medium, large-v3, large-v3-turbo)"
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

    show_audio_setup_info

    echo "=== Setup Complete ==="
    echo ""
    echo "Usage:"
    echo "  uv run whisper-realtime --help"
    echo ""
    echo "Quick start:"
    echo "  uv run whisper-realtime start           # Start with microphone"
    echo "  uv run whisper-realtime start -s system # Start with system audio"
    echo "  uv run whisper-realtime devices         # List audio devices"
    echo "  uv run whisper-realtime models          # List available models"
    echo ""
}

main "$@"
