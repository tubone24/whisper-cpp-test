"""
whisper.cpp 文字起こしエンジン
リアルタイム処理と訂正機能を提供
"""

import os
import subprocess
import tempfile
import threading
import time
from dataclasses import dataclass, field
from enum import Enum
from pathlib import Path
from typing import Callable, Optional

import numpy as np
from scipy.io import wavfile


class WhisperModel(Enum):
    """利用可能なWhisperモデル"""
    TINY = "tiny"
    TINY_EN = "tiny.en"
    BASE = "base"
    BASE_EN = "base.en"
    SMALL = "small"
    SMALL_EN = "small.en"
    MEDIUM = "medium"
    MEDIUM_EN = "medium.en"
    LARGE_V1 = "large-v1"
    LARGE_V2 = "large-v2"
    LARGE_V3 = "large-v3"
    LARGE_V3_TURBO = "large-v3-turbo"


# モデルの推奨設定（リアルタイム性 vs 精度）
MODEL_PROFILES = {
    "realtime": WhisperModel.TINY,        # 最速、低精度
    "balanced": WhisperModel.BASE,        # バランス
    "quality": WhisperModel.SMALL,        # 高精度、やや遅い
    "best": WhisperModel.LARGE_V3_TURBO,  # 最高精度、要GPU
}


@dataclass
class TranscriptionResult:
    """文字起こし結果"""
    text: str
    start_time: float
    end_time: float
    is_final: bool = False
    confidence: float = 0.0
    language: str = ""


@dataclass
class WhisperConfig:
    """Whisperエンジン設定"""
    model: WhisperModel = WhisperModel.BASE
    language: str = "ja"  # 言語コード (ja, en, auto)
    translate: bool = False  # 英語に翻訳
    threads: int = 4
    processors: int = 1
    # リアルタイム設定
    step_ms: int = 500  # 処理ステップ（ミリ秒）
    length_ms: int = 5000  # 処理窓の長さ（ミリ秒）
    keep_ms: int = 200  # コンテキスト保持（ミリ秒）
    # パス設定
    whisper_cpp_path: Optional[Path] = None
    models_path: Optional[Path] = None


class WhisperEngine:
    """whisper.cpp ベースの文字起こしエンジン"""

    def __init__(self, config: Optional[WhisperConfig] = None):
        self.config = config or WhisperConfig()
        self._setup_paths()
        self._validate_setup()

        self.is_running = False
        self._buffer: list[np.ndarray] = []
        self._buffer_lock = threading.Lock()
        self._last_result = ""
        self._results: list[TranscriptionResult] = []
        self._callback: Optional[Callable[[TranscriptionResult], None]] = None

    def _setup_paths(self):
        """パスの設定"""
        base_path = Path(__file__).parent.parent

        if self.config.whisper_cpp_path is None:
            self.config.whisper_cpp_path = base_path / "whisper.cpp"

        if self.config.models_path is None:
            self.config.models_path = base_path / "models"

        # CMakeビルド後のバイナリパス
        build_bin = self.config.whisper_cpp_path / "build" / "bin"
        self._stream_binary = build_bin / "whisper-stream"
        self._main_binary = build_bin / "whisper-cli"

        # フォールバック: 古いMakefileビルドのパス
        if not self._main_binary.exists():
            self._main_binary = self.config.whisper_cpp_path / "main"
        if not self._stream_binary.exists():
            self._stream_binary = self.config.whisper_cpp_path / "stream"

    def _validate_setup(self):
        """セットアップの検証"""
        # streamバイナリの存在確認
        if not self._stream_binary.exists():
            if not self._main_binary.exists():
                raise FileNotFoundError(
                    f"whisper.cpp バイナリが見つかりません。\n"
                    f"setup.sh を実行してください。"
                )

        # モデルファイルの確認
        model_file = self._get_model_path()
        if not model_file.exists():
            raise FileNotFoundError(
                f"モデルファイルが見つかりません: {model_file}\n"
                f"setup.sh でモデルをダウンロードしてください。"
            )

    def _get_model_path(self) -> Path:
        """モデルファイルのパスを取得"""
        model_name = self.config.model.value
        return self.config.models_path / f"ggml-{model_name}.bin"

    def set_callback(self, callback: Callable[[TranscriptionResult], None]):
        """結果コールバックを設定"""
        self._callback = callback

    def add_audio(self, audio: np.ndarray):
        """音声データをバッファに追加"""
        with self._buffer_lock:
            self._buffer.append(audio)

    def get_buffer_duration(self) -> float:
        """バッファの長さ（秒）を取得"""
        with self._buffer_lock:
            if not self._buffer:
                return 0.0
            total_samples = sum(len(chunk) for chunk in self._buffer)
            return total_samples / 16000  # 16kHz想定

    def _get_buffer_audio(self) -> np.ndarray:
        """バッファの音声を取得"""
        with self._buffer_lock:
            if not self._buffer:
                return np.array([], dtype=np.float32)
            audio = np.concatenate(self._buffer)
            return audio

    def _clear_buffer(self, keep_samples: int = 0):
        """バッファをクリア（一部保持可能）"""
        with self._buffer_lock:
            if keep_samples > 0 and self._buffer:
                full_audio = np.concatenate(self._buffer)
                if len(full_audio) > keep_samples:
                    self._buffer = [full_audio[-keep_samples:]]
                # それ以外はそのまま
            else:
                self._buffer.clear()

    def transcribe_audio(self, audio: np.ndarray) -> Optional[TranscriptionResult]:
        """音声データを文字起こし"""
        if len(audio) == 0:
            return None

        # 一時WAVファイルに保存
        with tempfile.NamedTemporaryFile(suffix=".wav", delete=False) as f:
            temp_path = f.name
            # int16に変換して保存
            audio_int16 = (audio * 32767).astype(np.int16)
            wavfile.write(temp_path, 16000, audio_int16)

        try:
            # whisper.cpp main を呼び出し
            cmd = [
                str(self._main_binary),
                "-m", str(self._get_model_path()),
                "-f", temp_path,
                "-l", self.config.language if self.config.language != "auto" else "auto",
                "-t", str(self.config.threads),
                "-p", str(self.config.processors),
                "--no-timestamps",
                "-nt",  # no prints
            ]

            if self.config.translate:
                cmd.append("--translate")

            result = subprocess.run(
                cmd,
                capture_output=True,
                text=True,
                timeout=30,
            )

            if result.returncode == 0:
                text = result.stdout.strip()
                # 空白や改行を整理
                text = " ".join(text.split())
                if text:
                    return TranscriptionResult(
                        text=text,
                        start_time=time.time(),
                        end_time=time.time(),
                        is_final=True,
                    )
        except subprocess.TimeoutExpired:
            pass
        except Exception as e:
            print(f"Transcription error: {e}")
        finally:
            os.unlink(temp_path)

        return None

    def process_realtime(self) -> Optional[TranscriptionResult]:
        """リアルタイム処理を実行"""
        # 十分な音声が溜まったら処理
        buffer_duration = self.get_buffer_duration()
        min_duration = self.config.length_ms / 1000

        if buffer_duration < min_duration:
            return None

        audio = self._get_buffer_audio()
        result = self.transcribe_audio(audio)

        if result:
            # 前回と比較して差分を検出（訂正機能）
            if self._last_result and result.text.startswith(self._last_result):
                # 追加のみ
                result.is_final = False
            else:
                # 訂正あり
                result.is_final = False

            self._last_result = result.text
            self._results.append(result)

            if self._callback:
                self._callback(result)

        # バッファの一部を保持（コンテキスト用）
        keep_samples = int(16000 * self.config.keep_ms / 1000)
        self._clear_buffer(keep_samples)

        return result

    def finalize(self) -> Optional[TranscriptionResult]:
        """残りのバッファを処理して最終結果を返す"""
        audio = self._get_buffer_audio()
        if len(audio) > 0:
            result = self.transcribe_audio(audio)
            if result:
                result.is_final = True
                self._results.append(result)
                if self._callback:
                    self._callback(result)
                return result
        return None

    def get_full_transcript(self) -> str:
        """全ての結果を結合したテキストを取得"""
        return " ".join(r.text for r in self._results if r.text)


class StreamingWhisperEngine:
    """whisper.cpp stream を使用したストリーミングエンジン"""

    def __init__(self, config: Optional[WhisperConfig] = None):
        self.config = config or WhisperConfig()
        self._setup_paths()
        self._process: Optional[subprocess.Popen] = None
        self._callback: Optional[Callable[[str, bool], None]] = None
        self._output_thread: Optional[threading.Thread] = None
        self.is_running = False

    def _setup_paths(self):
        """パスの設定"""
        base_path = Path(__file__).parent.parent

        if self.config.whisper_cpp_path is None:
            self.config.whisper_cpp_path = base_path / "whisper.cpp"

        if self.config.models_path is None:
            self.config.models_path = base_path / "models"

        # CMakeビルド後のバイナリパス
        build_bin = self.config.whisper_cpp_path / "build" / "bin"
        self._stream_binary = build_bin / "whisper-stream"

        # フォールバック: 古いMakefileビルドのパス
        if not self._stream_binary.exists():
            self._stream_binary = self.config.whisper_cpp_path / "stream"

    def _get_model_path(self) -> Path:
        """モデルファイルのパスを取得"""
        model_name = self.config.model.value
        return self.config.models_path / f"ggml-{model_name}.bin"

    def set_callback(self, callback: Callable[[str, bool], None]):
        """
        結果コールバックを設定

        Args:
            callback: (text, is_partial) -> None
                text: 文字起こしテキスト
                is_partial: 部分結果かどうか（訂正される可能性あり）
        """
        self._callback = callback

    def start(self, capture_id: Optional[int] = None):
        """ストリーミング開始"""
        if self.is_running:
            return

        model_path = self._get_model_path()
        if not model_path.exists():
            raise FileNotFoundError(f"モデルが見つかりません: {model_path}")

        cmd = [
            str(self._stream_binary),
            "-m", str(model_path),
            "-l", self.config.language if self.config.language != "auto" else "auto",
            "-t", str(self.config.threads),
            "--step", str(self.config.step_ms),
            "--length", str(self.config.length_ms),
            "--keep", str(self.config.keep_ms),
            "-vth", "0.6",  # VAD threshold
        ]

        if capture_id is not None:
            cmd.extend(["-c", str(capture_id)])

        if self.config.translate:
            cmd.append("--translate")

        self._process = subprocess.Popen(
            cmd,
            stdout=subprocess.PIPE,
            stderr=subprocess.PIPE,
            text=True,
            bufsize=1,
        )

        self.is_running = True
        self._output_thread = threading.Thread(target=self._read_output, daemon=True)
        self._output_thread.start()

    def _read_output(self):
        """出力を読み取ってコールバックを呼び出す"""
        if not self._process or not self._process.stdout:
            return

        current_text = ""

        for line in self._process.stdout:
            line = line.strip()
            if not line:
                continue

            # whisper.cpp stream の出力形式を解析
            # 通常: "[timestamp] text" または単純なテキスト
            if line.startswith("["):
                # タイムスタンプ付き
                try:
                    end_bracket = line.index("]")
                    text = line[end_bracket + 1:].strip()
                except ValueError:
                    text = line
            else:
                text = line

            if text:
                # 部分結果 vs 確定結果の判定
                # whisper.cpp stream は改行で区切られた結果を出力
                is_partial = not line.endswith(".")

                if self._callback:
                    self._callback(text, is_partial)

        self.is_running = False

    def stop(self):
        """ストリーミング停止"""
        if self._process:
            self._process.terminate()
            try:
                self._process.wait(timeout=2)
            except subprocess.TimeoutExpired:
                self._process.kill()
            self._process = None
        self.is_running = False

    def __enter__(self):
        self.start()
        return self

    def __exit__(self, exc_type, exc_val, exc_tb):
        self.stop()
