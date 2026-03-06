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
        if sys.platform != "darwin":
            print("[ScreenCaptureKit] macOS以外では利用不可")
            return False

        if not self.config.use_screencapturekit:
            print("[ScreenCaptureKit] 設定で無効化されています")
            return False

        try:
            from .system_audio_capture import (
                ScreenCaptureKitAudioCapture,
                is_screencapturekit_available,
                get_screencapturekit_error,
            )

            if not is_screencapturekit_available():
                error_msg = get_screencapturekit_error()
                print(f"[ScreenCaptureKit] 利用不可: {error_msg}")
                return False

            print("[ScreenCaptureKit] 初期化中...")
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

        except ImportError as e:
            print(f"[ScreenCaptureKit] インポートエラー: {e}")
            print("  インストール: uv pip install 'whisper-realtime[macos]'")
            self._sck_capture = None
            self._use_sck = False
            return False
        except Exception as e:
            print(f"[ScreenCaptureKit] 初期化エラー: {e}")
            import traceback
            traceback.print_exc()
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
    """音声区間検出（VAD）フィルタ - Silero VAD使用（高精度）"""

    def __init__(
        self,
        sample_rate: int = 16000,
        threshold: float = 0.5,
        min_speech_ratio: float = 0.3,
        use_silero: bool = True,
    ):
        """
        Args:
            sample_rate: サンプルレート (16000推奨)
            threshold: Silero VADの検出閾値 (0.0-1.0、高いほど厳格)
            min_speech_ratio: 発話判定に必要な最小フレーム比率
            use_silero: Silero VADを使用 (Falseの場合webrtcvadにフォールバック)
        """
        self.sample_rate = sample_rate
        self.threshold = threshold
        self.min_speech_ratio = min_speech_ratio
        self.enabled = False
        self.use_silero = False
        self.silero_model = None
        self.silero_utils = None
        self.vad = None  # webrtcvad fallback
        self._chunk_size = 512 if sample_rate == 16000 else 256  # Silero要件

        # Silero VADを試みる（高精度）
        if use_silero:
            try:
                import torch
                # Silero VADをロード（キャッシュされる）
                self.silero_model, utils = torch.hub.load(
                    repo_or_dir='snakers4/silero-vad',
                    model='silero_vad',
                    trust_repo=True,
                    verbose=False,
                )
                self.silero_utils = utils
                self.use_silero = True
                self.enabled = True
                # 初期状態リセット
                self.silero_model.reset_states()
                print("[VAD] Silero VAD loaded (high accuracy)")
            except Exception as e:
                print(f"[VAD] Silero VAD not available: {e}")
                print("[VAD] Falling back to webrtcvad...")

        # webrtcvadにフォールバック
        if not self.use_silero:
            try:
                import webrtcvad
                self.vad = webrtcvad.Vad(2)  # aggressiveness=2
                self.frame_duration_ms = 30
                self.frame_size = int(sample_rate * self.frame_duration_ms / 1000)
                self.enabled = True
                print("[VAD] webrtcvad loaded (fallback)")
            except ImportError:
                print("[VAD] No VAD available - all audio will be processed")
            except Exception as e:
                print(f"[VAD] webrtcvad error: {e}")

    def reset_states(self):
        """Silero VADの内部状態をリセット（新しい音声ストリーム開始時に呼ぶ）"""
        if self.use_silero and self.silero_model is not None:
            try:
                self.silero_model.reset_states()
                self._states_reset = True
            except Exception as e:
                print(f"[VAD] Failed to reset states: {e}")

    def is_speech(self, audio: np.ndarray) -> bool:
        """音声データに発話が含まれているかチェック"""
        if not self.enabled:
            return True

        if self.use_silero:
            return self._is_speech_silero(audio)
        else:
            return self._is_speech_webrtc(audio)

    def _is_speech_silero(self, audio: np.ndarray) -> bool:
        """Silero VADで発話検出"""
        import torch

        # float32の1D配列を確保
        audio = np.asarray(audio, dtype=np.float32).flatten()

        # Silero VADは512サンプル（16kHz）または256サンプル（8kHz）単位で処理
        chunk_size = self._chunk_size

        if len(audio) < chunk_size:
            # サンプル数が足りない場合はゼロパディング
            padded = np.zeros(chunk_size, dtype=np.float32)
            padded[:len(audio)] = audio
            audio = padded

        # チャンクごとに処理して、発話フレームの割合を計算
        speech_frames = 0
        total_frames = 0

        try:
            # 明示的に正確なチャンクサイズで分割
            num_chunks = len(audio) // chunk_size

            with torch.no_grad():
                for chunk_idx in range(num_chunks):
                    start = chunk_idx * chunk_size
                    end = start + chunk_size
                    chunk = audio[start:end]

                    # チャンクサイズを厳密に検証
                    if len(chunk) != chunk_size:
                        print(f"[VAD] Skip: chunk size {len(chunk)} != {chunk_size}")
                        continue

                    # 正確に chunk_size 要素の1D tensorを作成
                    chunk_tensor = torch.tensor(chunk, dtype=torch.float32)

                    # モデルに渡す前に形状を確認（デバッグ）
                    if chunk_tensor.shape[0] != chunk_size:
                        print(f"[VAD] Error: tensor shape {chunk_tensor.shape} != ({chunk_size},)")
                        continue

                    # サンプルレートも確認
                    sr = self.sample_rate

                    speech_prob = self.silero_model(chunk_tensor, sr).item()
                    if speech_prob >= self.threshold:
                        speech_frames += 1
                    total_frames += 1

        except Exception as e:
            # エラー時はフォールバック（RMSベース）
            print(f"[VAD] Silero error: {e}, falling back to RMS")
            rms = np.sqrt(np.mean(audio ** 2))
            return rms > 0.003

        # 発話フレームの割合が min_speech_ratio 以上なら発話と判定
        if total_frames == 0:
            return False
        return (speech_frames / total_frames) >= self.min_speech_ratio

    def _is_speech_webrtc(self, audio: np.ndarray) -> bool:
        """webrtcvadで発話検出（フォールバック）"""
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

        if total_frames == 0:
            return False
        return (speech_frames / total_frames) >= self.min_speech_ratio

    def filter_audio(self, audio: np.ndarray) -> Optional[np.ndarray]:
        """発話が含まれている場合のみ音声を返す"""
        if self.is_speech(audio):
            return audio
        return None

    def get_speech_timestamps(self, audio: np.ndarray) -> list[dict]:
        """音声データから発話区間のタイムスタンプを取得（Silero VADのみ）"""
        if not self.use_silero:
            # webrtcvadではタイムスタンプ取得非対応
            return [{"start": 0, "end": len(audio)}] if self.is_speech(audio) else []

        import torch
        get_speech_timestamps = self.silero_utils[0]

        audio_tensor = torch.from_numpy(audio.astype(np.float32))
        timestamps = get_speech_timestamps(
            audio_tensor,
            self.silero_model,
            sampling_rate=self.sample_rate,
            threshold=self.threshold,
        )
        return timestamps

    def extract_speech(self, audio: np.ndarray) -> np.ndarray:
        """音声データから発話部分のみを抽出"""
        timestamps = self.get_speech_timestamps(audio)
        if not timestamps:
            return np.array([], dtype=np.float32)

        segments = []
        for ts in timestamps:
            segments.append(audio[ts["start"]:ts["end"]])

        return np.concatenate(segments) if segments else np.array([], dtype=np.float32)


class NoiseReducer:
    """ノイズ除去フィルタ - DeepFilterNet使用（高品質・リアルタイム対応）"""

    def __init__(self, sample_rate: int = 16000):
        """
        Args:
            sample_rate: 入力サンプルレート (16000推奨、内部で48kHzに変換)
        """
        self.input_sample_rate = sample_rate
        self.df_sample_rate = 48000  # DeepFilterNetは48kHz
        self.enabled = False
        self.model = None
        self.df_state = None

        try:
            from df import enhance, init_df
            self.model, self.df_state, _ = init_df()
            self._enhance = enhance
            self.enabled = True
            print("[NoiseReducer] DeepFilterNet loaded (high quality)")
        except ImportError:
            print("[NoiseReducer] DeepFilterNet not available")
            print("  Install: uv pip install 'whisper-realtime[enhanced]'")
        except Exception as e:
            print(f"[NoiseReducer] DeepFilterNet error: {e}")

    def reduce_noise(self, audio: np.ndarray) -> np.ndarray:
        """音声データからノイズを除去"""
        if not self.enabled:
            return audio

        import torch
        from scipy import signal

        # 16kHz -> 48kHz にアップサンプリング
        if self.input_sample_rate != self.df_sample_rate:
            num_samples = int(len(audio) * self.df_sample_rate / self.input_sample_rate)
            audio_48k = signal.resample(audio, num_samples)
        else:
            audio_48k = audio

        # tensorに変換
        audio_tensor = torch.from_numpy(audio_48k.astype(np.float32))

        # ノイズ除去
        enhanced = self._enhance(self.model, self.df_state, audio_tensor)

        # numpy配列に戻す
        if isinstance(enhanced, torch.Tensor):
            enhanced = enhanced.numpy()

        # 48kHz -> 16kHz にダウンサンプリング
        if self.input_sample_rate != self.df_sample_rate:
            num_samples = int(len(enhanced) * self.input_sample_rate / self.df_sample_rate)
            enhanced = signal.resample(enhanced, num_samples)

        return enhanced.astype(np.float32)


class AudioPreprocessor:
    """音声前処理パイプライン（VAD + ノイズ除去）"""

    def __init__(
        self,
        sample_rate: int = 16000,
        use_vad: bool = True,
        use_noise_reduction: bool = False,
        vad_threshold: float = 0.5,
    ):
        """
        Args:
            sample_rate: サンプルレート
            use_vad: VADを使用するか
            use_noise_reduction: ノイズ除去を使用するか（高ノイズ環境向け）
            vad_threshold: VAD検出閾値
        """
        self.sample_rate = sample_rate
        self.vad = VADFilter(sample_rate=sample_rate, threshold=vad_threshold) if use_vad else None
        self.noise_reducer = NoiseReducer(sample_rate=sample_rate) if use_noise_reduction else None

    def process(self, audio: np.ndarray) -> Optional[np.ndarray]:
        """音声データを前処理

        Returns:
            処理後の音声データ。発話が検出されない場合はNone。
        """
        # 1. VADで発話チェック
        if self.vad and not self.vad.is_speech(audio):
            return None

        # 2. ノイズ除去（オプション）
        if self.noise_reducer and self.noise_reducer.enabled:
            audio = self.noise_reducer.reduce_noise(audio)

        return audio

    def extract_speech_with_denoise(self, audio: np.ndarray) -> np.ndarray:
        """音声データから発話部分を抽出し、ノイズ除去を適用"""
        # 1. VADで発話区間を抽出
        if self.vad:
            audio = self.vad.extract_speech(audio)
            if len(audio) == 0:
                return audio

        # 2. ノイズ除去
        if self.noise_reducer and self.noise_reducer.enabled:
            audio = self.noise_reducer.reduce_noise(audio)

        return audio
