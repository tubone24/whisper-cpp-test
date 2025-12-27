"""
音声キャプチャモジュール
マイク、システム音声、または両方からの音声入力を処理
"""

import queue
import sys
import threading
from dataclasses import dataclass
from enum import Enum
from typing import Callable, Optional

import numpy as np
import sounddevice as sd


class AudioSource(Enum):
    """音声入力ソースの種類"""
    MICROPHONE = "microphone"
    SYSTEM = "system"
    BOTH = "both"


@dataclass
class AudioConfig:
    """音声設定"""
    sample_rate: int = 16000  # whisper.cppは16kHzを要求
    channels: int = 1
    chunk_duration: float = 0.5  # チャンクの長さ（秒）
    dtype: str = "float32"
    use_screencapturekit: bool = True  # ScreenCaptureKitを優先使用


class AudioCapture:
    """音声キャプチャクラス"""

    def __init__(
        self,
        config: Optional[AudioConfig] = None,
        source: AudioSource = AudioSource.MICROPHONE,
        device_id: Optional[int] = None,
        system_device_id: Optional[int] = None,
    ):
        self.config = config or AudioConfig()
        self.source = source
        self.device_id = device_id
        self.system_device_id = system_device_id

        self.audio_queue: queue.Queue = queue.Queue()
        self.is_running = False
        self._stream: Optional[sd.InputStream] = None
        self._system_stream: Optional[sd.InputStream] = None
        self._sck_capture = None  # ScreenCaptureKit capture
        self._sck_thread = None  # ScreenCaptureKit audio forwarding thread
        self._use_sck = False
        self._lock = threading.Lock()

    @staticmethod
    def list_devices() -> list[dict]:
        """利用可能なオーディオデバイスを一覧表示"""
        devices = sd.query_devices()
        result = []
        for i, device in enumerate(devices):
            if device["max_input_channels"] > 0:
                result.append({
                    "id": i,
                    "name": device["name"],
                    "channels": device["max_input_channels"],
                    "sample_rate": device["default_samplerate"],
                    "is_default": i == sd.default.device[0],
                })
        return result

    @staticmethod
    def find_blackhole_device() -> Optional[int]:
        """BlackHole（システム音声キャプチャ用）デバイスを検索"""
        devices = sd.query_devices()
        for i, device in enumerate(devices):
            if "blackhole" in device["name"].lower() and device["max_input_channels"] > 0:
                return i
        return None

    def _audio_callback(self, indata: np.ndarray, frames: int, time_info, status):
        """音声コールバック"""
        if status:
            print(f"Audio status: {status}")
        # モノラルに変換
        if indata.shape[1] > 1:
            audio = np.mean(indata, axis=1)
        else:
            audio = indata[:, 0]
        self.audio_queue.put(audio.copy())

    def _create_stream(self, device_id: Optional[int]) -> sd.InputStream:
        """ストリームを作成"""
        chunk_size = int(self.config.sample_rate * self.config.chunk_duration)
        return sd.InputStream(
            device=device_id,
            samplerate=self.config.sample_rate,
            channels=self.config.channels,
            dtype=self.config.dtype,
            blocksize=chunk_size,
            callback=self._audio_callback,
        )

    def _try_screencapturekit(self) -> bool:
        """ScreenCaptureKitでのシステム音声キャプチャを試みる"""
        if sys.platform != "darwin" or not self.config.use_screencapturekit:
            return False

        try:
            from .system_audio_capture import (
                ScreenCaptureKitAudioCapture,
                is_screencapturekit_available,
            )

            if not is_screencapturekit_available():
                return False

            self._sck_capture = ScreenCaptureKitAudioCapture(
                sample_rate=self.config.sample_rate,
                chunk_duration=self.config.chunk_duration,
            )
            self._sck_capture.start()
            self._use_sck = True

            # ScreenCaptureKitからの音声をメインキューに転送するスレッド
            def forward_audio():
                while self._use_sck and self._sck_capture and self._sck_capture.is_running:
                    audio = self._sck_capture.get_audio(timeout=0.5)
                    if audio is not None and len(audio) > 0:
                        self.audio_queue.put(audio)

            self._sck_thread = threading.Thread(target=forward_audio, daemon=True)
            self._sck_thread.start()

            return True

        except Exception as e:
            print(f"[ScreenCaptureKit] 初期化エラー: {e}")
            self._sck_capture = None
            self._use_sck = False
            return False

    def start(self):
        """音声キャプチャを開始"""
        with self._lock:
            if self.is_running:
                return

            if self.source == AudioSource.MICROPHONE:
                self._stream = self._create_stream(self.device_id)
                self._stream.start()

            elif self.source == AudioSource.SYSTEM:
                # まずScreenCaptureKitを試みる（macOS 13.0+、BlackHole不要）
                if self._try_screencapturekit():
                    print("[ScreenCaptureKit] BlackHoleなしでシステム音声をキャプチャ中")
                else:
                    # フォールバック: BlackHoleなどの仮想デバイス
                    device = self.system_device_id or self.find_blackhole_device()
                    if device is not None:
                        self._stream = self._create_stream(device)
                        self._stream.start()
                    else:
                        raise RuntimeError(
                            "システム音声キャプチャを開始できません。\n\n"
                            "macOS 13.0以上の場合:\n"
                            "  1. uv pip install 'whisper-realtime[macos]'\n"
                            "  2. システム設定 > プライバシー > 画面収録 で許可\n\n"
                            "または BlackHole をインストール:\n"
                            "  1. brew install blackhole-2ch\n"
                            "  2. Audio MIDI設定.app を開く\n"
                            "  3. 左下の「+」→「複数出力装置を作成」\n"
                            "  4. 「BlackHole 2ch」と使用するスピーカーをチェック\n"
                            "  5. システム設定 > サウンド > 出力 で「複数出力装置」を選択"
                        )

            elif self.source == AudioSource.BOTH:
                # マイクストリームを開始
                self._stream = self._create_stream(self.device_id)
                self._stream.start()

                # システム音声: ScreenCaptureKitを優先
                if self._try_screencapturekit():
                    print("[ScreenCaptureKit] マイク + システム音声をキャプチャ中")
                else:
                    # フォールバック: BlackHoleなどの仮想デバイス
                    system_device = self.system_device_id or self.find_blackhole_device()
                    if system_device is not None:
                        self._system_stream = self._create_stream(system_device)
                        self._system_stream.start()
                        print("[BlackHole] マイク + システム音声をキャプチャ中")
                    else:
                        # システム音声なしで続行
                        print("[警告] システム音声キャプチャが利用できません。マイクのみで続行します。")
                        print("ScreenCaptureKit を使用するには:")
                        print("  uv pip install 'whisper-realtime[macos]'")
                        print("  システム設定 > プライバシー > 画面収録 で許可")

            self.is_running = True

    def stop(self):
        """音声キャプチャを停止"""
        with self._lock:
            if not self.is_running:
                return

            if self._stream:
                self._stream.stop()
                self._stream.close()
                self._stream = None

            if self._system_stream:
                self._system_stream.stop()
                self._system_stream.close()
                self._system_stream = None

            # ScreenCaptureKitの停止
            self._use_sck = False  # スレッドに停止を通知
            if self._sck_capture:
                self._sck_capture.stop()
                self._sck_capture = None

            # 転送スレッドの終了を待機
            if self._sck_thread:
                self._sck_thread.join(timeout=2.0)
                self._sck_thread = None

            self.is_running = False

    def get_audio(self, timeout: float = 1.0) -> Optional[np.ndarray]:
        """音声データを取得"""
        try:
            return self.audio_queue.get(timeout=timeout)
        except queue.Empty:
            return None

    def get_all_audio(self) -> np.ndarray:
        """キューにある全ての音声データを取得"""
        chunks = []
        while not self.audio_queue.empty():
            try:
                chunks.append(self.audio_queue.get_nowait())
            except queue.Empty:
                break
        if chunks:
            return np.concatenate(chunks)
        return np.array([], dtype=np.float32)

    def __enter__(self):
        self.start()
        return self

    def __exit__(self, exc_type, exc_val, exc_tb):
        self.stop()


class VADFilter:
    """音声区間検出（VAD）フィルタ"""

    def __init__(self, sample_rate: int = 16000, frame_duration_ms: int = 30, aggressiveness: int = 2):
        """
        Args:
            sample_rate: サンプルレート (8000, 16000, 32000, 48000)
            frame_duration_ms: フレーム長 (10, 20, 30)
            aggressiveness: 検出の積極性 (0-3, 高いほど厳格)
        """
        self.sample_rate = sample_rate
        self.frame_duration_ms = frame_duration_ms
        self.frame_size = int(sample_rate * frame_duration_ms / 1000)
        self.enabled = False
        self.vad = None

        try:
            import webrtcvad
            self.vad = webrtcvad.Vad(aggressiveness)
            self.enabled = True
        except ImportError:
            # webrtcvadはオプション - Apple Siliconでのビルドが難しい場合がある
            pass
        except Exception as e:
            # webrtcvadのロードエラー
            pass

    def is_speech(self, audio: np.ndarray) -> bool:
        """音声データに発話が含まれているかチェック"""
        if not self.enabled:
            return True

        # float32をint16に変換
        audio_int16 = (audio * 32767).astype(np.int16)

        # フレームごとにチェック
        speech_frames = 0
        total_frames = 0

        for i in range(0, len(audio_int16) - self.frame_size, self.frame_size):
            frame = audio_int16[i:i + self.frame_size]
            if len(frame) == self.frame_size:
                total_frames += 1
                if self.vad.is_speech(frame.tobytes(), self.sample_rate):
                    speech_frames += 1

        # 30%以上のフレームで発話検出されたらTrue
        if total_frames == 0:
            return False
        return (speech_frames / total_frames) >= 0.3

    def filter_audio(self, audio: np.ndarray) -> Optional[np.ndarray]:
        """発話が含まれている場合のみ音声を返す"""
        if self.is_speech(audio):
            return audio
        return None
