#!/bin/bash
# WhisperMenuBar ビルドスクリプト

set -e

SCRIPT_DIR="$(cd "$(dirname "$0")" && pwd)"
PROJECT_DIR="$(dirname "$SCRIPT_DIR")"
APP_NAME="WhisperMenuBar"
BUILD_DIR="$SCRIPT_DIR/build"

echo "🔨 Building $APP_NAME..."

# ビルドディレクトリ作成
mkdir -p "$BUILD_DIR"

# Swiftファイルをコンパイル
swiftc \
    -o "$BUILD_DIR/$APP_NAME" \
    -framework Cocoa \
    -framework Carbon \
    "$SCRIPT_DIR/WhisperMenuBar.swift"

echo "✅ Build complete: $BUILD_DIR/$APP_NAME"
echo ""
echo "実行方法:"
echo "  $BUILD_DIR/$APP_NAME"
echo ""
echo "または、アプリバンドルを作成する場合:"
echo "  ./bundle.sh"
