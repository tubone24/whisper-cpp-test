"""
Push-to-Talk 音声入力モジュール
キーを押している間録音し、離すと確定してペースト
"""

import logging
import os
import subprocess
import sys
import threading
import time
from dataclasses import dataclass
from enum import Enum
from pathlib import Path
from typing import Callable, Optional

import numpy as np

# デバッグログ設定（stderrに出力）
logging.basicConfig(
    level=logging.DEBUG,
    format='[%(asctime)s.%(msecs)03d] %(levelname)s %(name)s: %(message)s',
    datefmt='%H:%M:%S',
    stream=sys.stderr
)
logger = logging.getLogger('voice_input')

from .audio_capture import AudioCapture, AudioConfig, AudioSource, VADFilter, AudioPreprocessor
from .dictionary import Dictionary, load_or_create_dictionary
from .phonetic_corrector import PhoneticCorrector, PhoneticCorrectorConfig
from .whisper_engine import TranscriptionResult, WhisperConfig, WhisperEngine, WhisperModel


class OutputMode(Enum):
    """出力モード"""
    CLIPBOARD = "clipboard"  # クリップボードにコピーのみ
    TYPE = "type"  # キーボード入力としてタイプ
    BOTH = "both"  # 両方


class HotkeyType(Enum):
    """ホットキーの種類"""
    # 修飾キー
    CTRL_RIGHT = "ctrl_r"
    CTRL_LEFT = "ctrl_l"
    ALT_RIGHT = "alt_r"
    ALT_LEFT = "alt_l"
    SHIFT_RIGHT = "shift_r"
    SHIFT_LEFT = "shift_l"
    # ファンクションキー
    F1 = "f1"
    F2 = "f2"
    F3 = "f3"
    F4 = "f4"
    F5 = "f5"
    F6 = "f6"
    F7 = "f7"
    F8 = "f8"
    F9 = "f9"
    F10 = "f10"
    F11 = "f11"
    F12 = "f12"
    # その他
    CAPS_LOCK = "caps_lock"
    SCROLL_LOCK = "scroll_lock"
    PAUSE = "pause"


@dataclass
class VoiceInputConfig:
    """音声入力設定"""
    # ホットキー設定
    hotkey: HotkeyType = HotkeyType.CTRL_RIGHT  # 右Ctrl がデフォルト
    # 出力設定
    output_mode: OutputMode = OutputMode.CLIPBOARD  # クリップボードにコピー
    # 辞書設定
    dictionary_path: Optional[Path] = None
    use_dictionary: bool = True
    # 音声認識誤り訂正設定
    use_phonetic_correction: bool = True  # 音声認識誤り訂正を使用
    # Whisper設定
    model: WhisperModel = WhisperModel.BASE
    language: str = "ja"
    # 音声設定
    device_id: Optional[int] = None
    use_vad: bool = True
    # UI設定
    show_realtime: bool = True  # リアルタイム表示
    sound_feedback: bool = True  # 開始/終了音
    # Streaming処理設定（長い文章の精度向上用）
    step_ms: int = 500  # 処理ステップ間隔（ms）
    length_ms: int = 10000  # 処理窓の長さ（ms）- デフォルト10秒で長文対応
    keep_ms: int = 500  # コンテキスト保持時間（ms）
    max_tokens: int = 128  # 最大トークン数 - 長文対応のため増加
    # Two-Pass処理設定（Zoom/Google Meet方式）
    two_pass: bool = True  # 2段階処理を有効化
    partial_step_ms: int = 100  # Partial用：処理間隔（100ms固定）
    partial_window_sec: float = 5.0  # Partial用：最新N秒のみ処理
    partial_beam_size: int = 1  # Partial用：greedy search（高速）
    final_beam_size: int = 5  # Final用：beam search（高精度）
    # 音声前処理設定
    use_silero_vad: bool = True  # Silero VADを使用（高精度）
    vad_threshold: float = 0.5  # VAD検出閾値（0.0-1.0）
    use_noise_reduction: bool = False  # DeepFilterNetノイズ除去（高ノイズ環境向け）
    # Utterance検出設定（発話単位での確定）
    utterance_silence_sec: float = 0.8  # 発話終了とみなす無音時間（秒）
    min_utterance_sec: float = 0.3  # 最小発話時間（秒）- これ以下は無視


def compute_spectrum(audio: np.ndarray, num_bands: int = 8) -> list[float]:
    """
    音声データからFFTを計算し、周波数帯域ごとのレベルを返す

    Args:
        audio: 16kHz音声データ（float32）
        num_bands: 出力する周波数帯域の数

    Returns:
        各周波数帯域のレベル（0.0-1.0の範囲）
    """
    # サンプル数が足りない場合はゼロパディング
    min_samples = 512
    if len(audio) < min_samples:
        audio = np.pad(audio, (0, min_samples - len(audio)), mode='constant')

    # ハミング窓を適用
    window = np.hamming(len(audio))
    windowed = audio * window

    # FFT実行
    fft = np.fft.rfft(windowed)
    magnitude = np.abs(fft)

    # 周波数帯域に分割（対数スケールで分割）
    # 人間の聴覚に近い対数スケールで分割
    sample_rate = 16000
    freq_bins = np.fft.rfftfreq(len(audio), 1.0 / sample_rate)

    # 周波数範囲: 20Hz - 8000Hz（Nyquist以下）
    min_freq = 20
    max_freq = min(8000, sample_rate // 2)

    # 対数スケールで周波数帯域を分割
    log_min = np.log10(min_freq)
    log_max = np.log10(max_freq)
    band_edges = np.logspace(log_min, log_max, num_bands + 1)

    levels = []
    for i in range(num_bands):
        # この帯域の周波数範囲に該当するビンを選択
        low_freq = band_edges[i]
        high_freq = band_edges[i + 1]

        # 周波数ビンのインデックスを取得
        mask = (freq_bins >= low_freq) & (freq_bins < high_freq)

        if np.any(mask):
            # 帯域内のRMS値を計算
            band_magnitude = magnitude[mask]
            band_level = np.sqrt(np.mean(band_magnitude ** 2))
        else:
            band_level = 0.0

        levels.append(band_level)

    # 正規化（0-1の範囲に）
    max_level = max(levels) if levels else 1.0
    if max_level > 0:
        # 全体の最大値で正規化し、さらに見やすいようにスケーリング
        levels = [min(l / max_level * 1.5, 1.0) for l in levels]
    else:
        levels = [0.0] * num_bands

    return levels


class ClipboardManager:
    """クリップボード操作"""

    @staticmethod
    def copy(text: str) -> bool:
        """テキストをクリップボードにコピー"""
        try:
            # Linux: xclip または xsel を使用
            if sys.platform == "linux":
                # xclip を試す
                try:
                    proc = subprocess.Popen(
                        ["xclip", "-selection", "clipboard"],
                        stdin=subprocess.PIPE,
                    )
                    proc.communicate(text.encode("utf-8"))
                    return proc.returncode == 0
                except FileNotFoundError:
                    pass

                # xsel を試す
                try:
                    proc = subprocess.Popen(
                        ["xsel", "--clipboard", "--input"],
                        stdin=subprocess.PIPE,
                    )
                    proc.communicate(text.encode("utf-8"))
                    return proc.returncode == 0
                except FileNotFoundError:
                    pass

                # wl-copy (Wayland) を試す
                try:
                    proc = subprocess.Popen(
                        ["wl-copy"],
                        stdin=subprocess.PIPE,
                    )
                    proc.communicate(text.encode("utf-8"))
                    return proc.returncode == 0
                except FileNotFoundError:
                    pass

            # macOS
            elif sys.platform == "darwin":
                proc = subprocess.Popen(
                    ["pbcopy"],
                    stdin=subprocess.PIPE,
                )
                proc.communicate(text.encode("utf-8"))
                return proc.returncode == 0

            return False
        except Exception:
            return False

    @staticmethod
    def paste() -> Optional[str]:
        """クリップボードからペースト"""
        try:
            if sys.platform == "linux":
                try:
                    result = subprocess.run(
                        ["xclip", "-selection", "clipboard", "-o"],
                        capture_output=True,
                        text=True,
                    )
                    if result.returncode == 0:
                        return result.stdout
                except FileNotFoundError:
                    pass

                try:
                    result = subprocess.run(
                        ["xsel", "--clipboard", "--output"],
                        capture_output=True,
                        text=True,
                    )
                    if result.returncode == 0:
                        return result.stdout
                except FileNotFoundError:
                    pass

            elif sys.platform == "darwin":
                result = subprocess.run(["pbpaste"], capture_output=True, text=True)
                if result.returncode == 0:
                    return result.stdout

            return None
        except Exception:
            return None


class KeyboardTyper:
    """キーボード入力シミュレーション"""

    @staticmethod
    def type_text(text: str) -> bool:
        """テキストをキーボード入力としてタイプ"""
        try:
            if sys.platform == "linux":
                # xdotool を使用
                try:
                    result = subprocess.run(
                        ["xdotool", "type", "--clearmodifiers", "--", text],
                        capture_output=True,
                    )
                    return result.returncode == 0
                except FileNotFoundError:
                    pass

                # ydotool を試す (Wayland)
                try:
                    result = subprocess.run(
                        ["ydotool", "type", "--", text],
                        capture_output=True,
                    )
                    return result.returncode == 0
                except FileNotFoundError:
                    pass

                # wtype を試す (Wayland)
                try:
                    result = subprocess.run(
                        ["wtype", text],
                        capture_output=True,
                    )
                    return result.returncode == 0
                except FileNotFoundError:
                    pass

            elif sys.platform == "darwin":
                # macOS: クリップボードにコピーしてCmd+Vでペースト
                # より信頼性が高い方法
                ClipboardManager.copy(text)
                time.sleep(0.05)  # クリップボード反映待ち

                # Cmd+V を送信
                script = '''
                tell application "System Events"
                    keystroke "v" using command down
                end tell
                '''
                result = subprocess.run(
                    ["osascript", "-e", script],
                    capture_output=True,
                )
                return result.returncode == 0

            return False
        except Exception:
            return False


class VoiceInputSession:
    """単一の音声入力セッション"""

    def __init__(
        self,
        config: VoiceInputConfig,
        on_partial: Optional[Callable[[str], None]] = None,
        on_final: Optional[Callable[[str], None]] = None,
        on_level: Optional[Callable[[float], None]] = None,
        on_spectrum: Optional[Callable[[list[float]], None]] = None,
    ):
        self.config = config
        self.on_partial = on_partial
        self.on_final = on_final
        self.on_level = on_level
        self.on_spectrum = on_spectrum

        self._is_recording = False
        self._audio_buffer: list[np.ndarray] = []
        self._partial_text = ""
        self._engine: Optional[WhisperEngine] = None
        self._capture: Optional[AudioCapture] = None
        self._record_thread: Optional[threading.Thread] = None
        self._dictionary: Optional[Dictionary] = None
        self._phonetic_corrector: Optional[PhoneticCorrector] = None
        self._process_thread: Optional[threading.Thread] = None
        self._last_processed_samples = 0
        self._preprocessor: Optional[AudioPreprocessor] = None
        # Utterance（発話）ベースの確定処理
        self._confirmed_samples = 0  # 確定済み音声サンプル数
        self._confirmed_text = ""  # 確定済みテキスト
        self._utterance_start_samples = 0  # 現在のutteranceの開始位置
        self._is_speaking = False  # 現在発話中かどうか
        self._silence_start_time: Optional[float] = None  # 無音開始時刻
        self._last_vad_speech = False  # 前回のVAD結果
        self._last_audio_chunk_time: Optional[float] = None  # 最後に音声チャンクが追加された時刻

        # 辞書読み込み
        if config.use_dictionary:
            self._dictionary = load_or_create_dictionary(config.dictionary_path)

        # 音声認識誤り訂正機能初期化
        if config.use_phonetic_correction:
            self._phonetic_corrector = PhoneticCorrector()

        # 音声前処理（Silero VAD + DeepFilterNet）
        if config.use_vad or config.use_noise_reduction:
            self._preprocessor = AudioPreprocessor(
                sample_rate=16000,
                use_vad=config.use_vad and config.use_silero_vad,
                use_noise_reduction=config.use_noise_reduction,
                vad_threshold=config.vad_threshold,
            )
            # VAD状態をリセット（新しいセッション用）
            if self._preprocessor.vad:
                self._preprocessor.vad.reset_states()

    def start(self):
        """録音開始"""
        if self._is_recording:
            return

        logger.info("=== Recording session START ===")
        logger.info(f"  Model: {self.config.model.value}")
        logger.info(f"  Language: {self.config.language}")
        logger.info(f"  Dictionary: {self.config.use_dictionary}")
        logger.info(f"  PhoneticCorrection: {self.config.use_phonetic_correction}")

        # VAD状態をリセット（新しいセッション開始前）
        if self._preprocessor and self._preprocessor.vad:
            self._preprocessor.vad.reset_states()
            logger.debug("VAD states reset")

        self._is_recording = True
        self._audio_buffer = []
        self._partial_text = ""
        self._last_processed_samples = 0

        # Whisperエンジン初期化（Streaming最適化パラメータ）
        logger.debug("Initializing WhisperEngine...")
        logger.info(f"  step_ms: {self.config.step_ms}")
        logger.info(f"  length_ms: {self.config.length_ms}")
        logger.info(f"  keep_ms: {self.config.keep_ms}")
        logger.info(f"  max_tokens: {self.config.max_tokens}")
        whisper_config = WhisperConfig(
            model=self.config.model,
            language=self.config.language,
            # Streaming最適化設定（configから取得）
            step_ms=self.config.step_ms,
            length_ms=self.config.length_ms,
            keep_ms=self.config.keep_ms,
            beam_size=1,  # Greedy search（高速）
            max_tokens=self.config.max_tokens,
            use_flash_attn=True,
            no_timestamps=True,
        )
        self._engine = WhisperEngine(whisper_config)
        logger.debug("WhisperEngine initialized")

        # 音声キャプチャ初期化
        audio_config = AudioConfig(
            sample_rate=16000,
            chunk_duration=0.5,
        )
        self._capture = AudioCapture(
            config=audio_config,
            source=AudioSource.MICROPHONE,
            device_id=self.config.device_id,
        )

        # 録音スレッド開始
        self._record_thread = threading.Thread(target=self._record_loop, daemon=True)
        self._record_thread.start()

        # リアルタイム処理スレッド開始
        self._process_thread = threading.Thread(target=self._process_loop, daemon=True)
        self._process_thread.start()

    def _record_loop(self):
        """録音ループ - 音声をバッファに蓄積"""
        logger.debug("_record_loop started")
        chunk_count = 0

        # AudioPreprocessorを使用（Silero VAD + DeepFilterNet）
        # セッション初期化時に作成済みのpreprocessorを使用
        preprocessor = self._preprocessor

        try:
            with self._capture:
                while self._is_recording:
                    audio = self._capture.get_audio(timeout=0.1)
                    if audio is not None and len(audio) > 0:
                        # 音声レベルを計算してコールバック（前処理前の生データで計算）
                        if self.on_level:
                            rms = float(np.sqrt(np.mean(audio ** 2)))
                            # 0-1の範囲に正規化（一般的なマイク入力の場合）
                            level = min(rms * 10, 1.0)
                            self.on_level(level)

                        # スペクトラムを計算してコールバック
                        if self.on_spectrum:
                            spectrum = compute_spectrum(audio, num_bands=8)
                            self.on_spectrum(spectrum)

                        # 音声前処理（VAD + ノイズ除去）
                        try:
                            if preprocessor:
                                processed = preprocessor.process(audio)
                                if processed is not None:
                                    self._audio_buffer.append(processed)
                                    self._last_audio_chunk_time = time.time()  # 音声追加時刻を記録
                                    chunk_count += 1
                                # VADでフィルタされた場合は時刻を更新しない（無音検出用）
                            else:
                                self._audio_buffer.append(audio)
                                self._last_audio_chunk_time = time.time()
                                chunk_count += 1
                        except Exception as proc_err:
                            # 前処理エラーは警告を出して生データを使用
                            logger.warning(f"Preprocessing error: {proc_err}, using raw audio")
                            self._audio_buffer.append(audio)
                            self._last_audio_chunk_time = time.time()
                            chunk_count += 1

                        if chunk_count % 10 == 0:
                            total_samples = sum(len(a) for a in self._audio_buffer)
                            logger.debug(f"Audio: {chunk_count} chunks, {total_samples} samples ({total_samples/16000:.1f}s)")
        except Exception as e:
            logger.error(f"Recording error: {e}")
            import traceback
            traceback.print_exc()

        logger.debug(f"_record_loop ended, total chunks: {chunk_count}")

    def _process_loop(self):
        """リアルタイム処理ループ - Utterance（発話）ベースの確定処理"""
        logger.debug("_process_loop started")
        process_count = 0

        # Two-Pass有効時は高速な100ms間隔、無効時は従来のstep_ms
        if self.config.two_pass:
            step_sec = self.config.partial_step_ms / 1000.0
            partial_beam = self.config.partial_beam_size  # PARTIAL用: greedy (beam=1)
            final_beam = self.config.final_beam_size  # FINAL用: beam search (beam=5)
            logger.info(f"[Two-Pass Utterance] step={self.config.partial_step_ms}ms, partial_beam={partial_beam}, final_beam={final_beam}")
            logger.info(f"[Two-Pass Utterance] silence_threshold={self.config.utterance_silence_sec}s, min_utterance={self.config.min_utterance_sec}s")
        else:
            step_sec = self.config.step_ms / 1000.0
            partial_beam = None
            final_beam = None
            logger.info(f"[Single-Pass] step={self.config.step_ms}ms")

        # Utterance検出用の設定
        silence_threshold_sec = self.config.utterance_silence_sec
        min_utterance_samples = int(self.config.min_utterance_sec * 16000)

        while self._is_recording:
            time.sleep(step_sec)

            current_time = time.time()
            process_count += 1

            # バッファが空の場合
            if not self._audio_buffer:
                # 発話中なら無音として扱う
                if self._is_speaking and self._last_audio_chunk_time is not None:
                    time_since_last_audio = current_time - self._last_audio_chunk_time
                    if time_since_last_audio >= silence_threshold_sec:
                        logger.debug(f"[SILENCE #{process_count}] No audio for {time_since_last_audio:.1f}s, triggering FINAL")
                        # 発話終了処理（バッファが空なのでスキップ）
                        self._is_speaking = False
                        self._silence_start_time = None
                        self._partial_text = ""
                continue

            # 現在のバッファ全体を結合
            current_audio = np.concatenate(self._audio_buffer)
            current_samples = len(current_audio)

            # 新しい音声があるかチェック
            has_new_audio = current_samples > self._last_processed_samples + 1600  # 0.1秒以上の新規音声

            # 無音検出（VADでフィルタされて新しい音声がない場合）
            if not has_new_audio and self._is_speaking:
                if self._last_audio_chunk_time is not None:
                    time_since_last_audio = current_time - self._last_audio_chunk_time
                    logger.debug(f"[SILENCE #{process_count}] No new audio, time_since_last={time_since_last_audio:.2f}s, threshold={silence_threshold_sec}s")

                    # 無音が閾値を超えたらFINAL処理
                    if time_since_last_audio >= silence_threshold_sec:
                        logger.info(f"[FINAL #{process_count}] 🎯 Silence detected for {time_since_last_audio:.1f}s")

                        # Utteranceの音声範囲を特定
                        utterance_duration_samples = current_samples - self._utterance_start_samples

                        if utterance_duration_samples >= min_utterance_samples:
                            # Utteranceを高精度処理（beam search）
                            utterance_audio = current_audio[self._utterance_start_samples:]
                            logger.info(f"[FINAL #{process_count}] Processing utterance: {len(utterance_audio)/16000:.1f}s")

                            t0 = time.time()
                            # 確定テキストをコンテキストとして渡す（精度向上）
                            prompt = self._confirmed_text if self._confirmed_text else None
                            result = self._engine.transcribe_audio(utterance_audio, beam_size=final_beam, initial_prompt=prompt)
                            whisper_time = time.time() - t0

                            if result and result.text.strip():
                                text = self._apply_corrections(result.text.strip(), process_count, "FINAL")
                                # 確定テキストに追加
                                if self._confirmed_text:
                                    self._confirmed_text += text
                                else:
                                    self._confirmed_text = text
                                logger.info(f"[FINAL #{process_count}] ✅ '{text}' ({whisper_time:.2f}s)")
                                if self.on_final:
                                    self.on_final(self._confirmed_text)

                            # 確定済みサンプル数を更新
                            self._confirmed_samples = current_samples
                        else:
                            logger.debug(f"[FINAL #{process_count}] ⏭️ Too short ({utterance_duration_samples/16000:.2f}s), skipping")

                        # Utterance状態をリセット
                        self._is_speaking = False
                        self._silence_start_time = None
                        self._partial_text = ""

                continue

            # 新しい音声がある場合の処理
            if has_new_audio:
                try:
                    # VADで音声区間を検出（最新の音声チャンクをチェック）
                    new_audio = current_audio[self._last_processed_samples:]
                    is_speech = self._detect_speech(new_audio)
                    logger.debug(f"[VAD #{process_count}] is_speech={is_speech}, is_speaking={self._is_speaking}, new_audio_len={len(new_audio)}")

                    if self.config.two_pass:
                        # === Utterance ベースの Two-Pass 処理 ===

                        # 発話開始検出
                        if is_speech and not self._is_speaking:
                            self._is_speaking = True
                            self._utterance_start_samples = self._confirmed_samples
                            self._silence_start_time = None
                            logger.debug(f"[UTTERANCE #{process_count}] 🎤 Speech started at {self._utterance_start_samples/16000:.1f}s")

                        # 暫定部分の処理（確定済み以降の音声）- 発話中のみ
                        if self._is_speaking:
                            partial_audio = current_audio[self._confirmed_samples:]
                            if len(partial_audio) > 1600:  # 0.1秒以上の音声
                                logger.debug(f"[PARTIAL #{process_count}] Processing {len(partial_audio)/16000:.1f}s")
                                t0 = time.time()
                                # 確定テキストをコンテキストとして渡す（精度向上）
                                prompt = self._confirmed_text if self._confirmed_text else None
                                result = self._engine.transcribe_audio(partial_audio, beam_size=partial_beam, initial_prompt=prompt)
                                whisper_time = time.time() - t0

                                if result and result.text.strip():
                                    text = self._apply_corrections(result.text.strip(), process_count, "PARTIAL")
                                    self._partial_text = text
                                    logger.info(f"[PARTIAL #{process_count}] 🎤 '{text}' ({whisper_time:.2f}s)")
                                    if self.on_partial:
                                        self.on_partial(text)
                                else:
                                    if self.on_partial:
                                        self.on_partial("")

                    else:
                        # === Single-Pass処理 ===
                        t0 = time.time()
                        result = self._engine.transcribe_audio(current_audio, beam_size=partial_beam)
                        whisper_time = time.time() - t0

                        if result and result.text.strip():
                            text = self._apply_corrections(result.text.strip(), process_count, "PARTIAL")
                            self._partial_text = text
                            logger.info(f"[PARTIAL #{process_count}] 🎤 '{text}' ({whisper_time:.2f}s)")
                            if self.on_partial:
                                self.on_partial(text)

                    self._last_processed_samples = current_samples
                except Exception as e:
                    logger.error(f"Processing error: {e}")

        logger.debug(f"_process_loop ended, total processes: {process_count}")

    def _detect_speech(self, audio: np.ndarray) -> bool:
        """VADで音声区間を検出"""
        if len(audio) < 480:  # 30ms未満は無視
            return self._last_vad_speech

        # AudioPreprocessor の VADFilter を使用
        if self._preprocessor and self._preprocessor.vad and self._preprocessor.vad.enabled:
            try:
                is_speech = self._preprocessor.vad.is_speech(audio)
                self._last_vad_speech = is_speech
                return is_speech
            except Exception as e:
                logger.debug(f"VAD error: {e}")

        # フォールバック: 音量ベースの検出（VADが使えない場合）
        rms = np.sqrt(np.mean(audio ** 2))
        # 閾値を低めに設定（マイク感度に依存）
        # 0.005 = かなり静か、0.01 = 普通の発話、0.02 = はっきりした発話
        threshold = 0.003
        is_speech = rms > threshold
        if is_speech != self._last_vad_speech:
            logger.debug(f"[VAD-Fallback] RMS={rms:.4f}, threshold={threshold}, speech={is_speech}")
        self._last_vad_speech = is_speech
        return is_speech

    def _apply_corrections(self, text: str, process_count: int, prefix: str) -> str:
        """辞書と音声認識誤り訂正を適用"""
        # 辞書適用
        if self._dictionary:
            before_dict = text
            text = self._dictionary.apply(text)
            if text != before_dict:
                logger.debug(f"[{prefix} #{process_count}] 📖 Dictionary: '{before_dict}' → '{text}'")

        # 音声認識誤り訂正
        if self._phonetic_corrector:
            before_corr = text
            result = self._phonetic_corrector.correct(text)
            text = result.corrected_text
            if text != before_corr:
                logger.debug(f"[{prefix} #{process_count}] 🔊 Phonetic: '{before_corr}' → '{text}'")

        return text

    def _find_confirm_position(self, text: str) -> int:
        """確定可能な位置を見つける（句読点や助詞の後）"""
        # 日本語の句読点・区切り文字
        delimiters = ['。', '、', '？', '！', '．', '，', '?', '!', '.', ',', '　', ' ']
        # 日本語の助詞（文節の区切りになりやすい）
        particles = ['は', 'が', 'を', 'に', 'で', 'と', 'の', 'へ', 'から', 'まで', 'より', 'って', 'ね', 'よ', 'さ']

        best_pos = 0

        # 句読点を優先的に探す
        for i, char in enumerate(text):
            if char in delimiters:
                best_pos = i + 1

        # 句読点がなければ、助詞の後を探す（最低5文字以上）
        if best_pos == 0 and len(text) >= 5:
            for particle in particles:
                pos = text.rfind(particle)
                if pos > 3:  # 最低3文字は確保
                    candidate = pos + len(particle)
                    if candidate > best_pos and candidate < len(text) - 2:  # 末尾2文字は残す
                        best_pos = candidate

        return best_pos

    def stop(self) -> str:
        """録音停止して最終テキストを返す（未確定部分のみ処理）"""
        if not self._is_recording:
            return ""

        logger.info("=== Recording session STOP ===")
        self._is_recording = False

        # スレッド終了待ち
        logger.debug("Waiting for threads to finish...")
        if self._record_thread:
            self._record_thread.join(timeout=2.0)
        if self._process_thread:
            self._process_thread.join(timeout=1.0)
        logger.debug("Threads finished")

        # Two-Pass有効時は高精度なbeam searchで未確定部分のみ処理
        if self.config.two_pass:
            beam_size = self.config.final_beam_size
            logger.info(f"[Two-Pass] Final: beam_size={beam_size} (高精度)")
        else:
            beam_size = None
            logger.info("[Single-Pass] Final transcription")

        # 最終処理：未確定部分のみ処理（確定済み部分は再処理しない）
        final_text = self._confirmed_text  # 確定済みテキストはそのまま保持（再補正しない）
        remaining_text = ""

        if self._audio_buffer:
            full_audio = np.concatenate(self._audio_buffer)
            total_samples = len(full_audio)
            remaining_samples = total_samples - self._confirmed_samples

            logger.info(f"[FINAL] 🎙️ Total: {total_samples/16000:.1f}s, Confirmed: {self._confirmed_samples/16000:.1f}s, Remaining: {remaining_samples/16000:.1f}s")

            # 未確定部分がある場合のみ処理
            if remaining_samples > 1600:  # 0.1秒以上の未確定音声がある場合
                remaining_audio = full_audio[self._confirmed_samples:]
                try:
                    t0 = time.time()
                    # 確定テキストをコンテキストとして渡す（精度向上）
                    prompt = self._confirmed_text if self._confirmed_text else None
                    result = self._engine.transcribe_audio(remaining_audio, beam_size=beam_size, initial_prompt=prompt)
                    whisper_time = time.time() - t0
                    logger.info(f"[FINAL] ⏱️ Whisper処理時間: {whisper_time:.2f}s (beam_size={beam_size})")
                    if result and result.text.strip():
                        remaining_text = result.text.strip()
                        logger.info(f"[FINAL] 🎤 Remaining text: '{remaining_text}'")
                except Exception as e:
                    logger.error(f"[FINAL] ❌ Transcription error: {e}")
                    remaining_text = self._partial_text
            elif self._partial_text:
                # 未確定音声が短い場合は部分結果を使用
                remaining_text = self._partial_text
                logger.info(f"[FINAL] ⚠️ Using partial text: '{remaining_text}'")

        # 残りテキストにのみ辞書・音声補正を適用（確定テキストは処理済みなので再適用しない）
        if remaining_text:
            remaining_text = self._apply_corrections(remaining_text, 0, "FINAL-STOP")

        # 確定テキスト + 残りテキストを結合
        if remaining_text:
            final_text = final_text + remaining_text if final_text else remaining_text

        # 部分結果がある場合はそれを使用（フォールバック）
        if not final_text:
            final_text = self._partial_text
            logger.info(f"[FINAL] ⚠️ Fallback to partial text: '{final_text}'")

        logger.info(f"[FINAL] ✅ Output: '{final_text}'")

        if self.on_final:
            self.on_final(final_text)

        return final_text


class VoiceInputManager:
    """Push-to-Talk 音声入力マネージャー"""

    def __init__(self, config: Optional[VoiceInputConfig] = None):
        self.config = config or VoiceInputConfig()
        self._is_running = False
        self._current_session: Optional[VoiceInputSession] = None
        self._keyboard_listener = None
        self._on_output: Optional[Callable[[str], None]] = None

    def set_output_callback(self, callback: Callable[[str], None]):
        """出力コールバックを設定"""
        self._on_output = callback

    def _get_pynput_key(self):
        """pynputのキーオブジェクトを取得"""
        from pynput.keyboard import Key

        # HotkeyType から pynput Key属性名へのマッピング
        hotkey_to_attr = {
            HotkeyType.CTRL_RIGHT: "ctrl_r",
            HotkeyType.CTRL_LEFT: "ctrl_l",
            HotkeyType.ALT_RIGHT: "alt_r",
            HotkeyType.ALT_LEFT: "alt_l",
            HotkeyType.SHIFT_RIGHT: "shift_r",
            HotkeyType.SHIFT_LEFT: "shift_l",
            HotkeyType.F1: "f1",
            HotkeyType.F2: "f2",
            HotkeyType.F3: "f3",
            HotkeyType.F4: "f4",
            HotkeyType.F5: "f5",
            HotkeyType.F6: "f6",
            HotkeyType.F7: "f7",
            HotkeyType.F8: "f8",
            HotkeyType.F9: "f9",
            HotkeyType.F10: "f10",
            HotkeyType.F11: "f11",
            HotkeyType.F12: "f12",
            HotkeyType.CAPS_LOCK: "caps_lock",
            HotkeyType.SCROLL_LOCK: "scroll_lock",
            HotkeyType.PAUSE: "pause",
        }

        attr_name = hotkey_to_attr.get(self.config.hotkey)
        if attr_name:
            return getattr(Key, attr_name, None)
        return None

    def _on_key_press(self, key):
        """キー押下イベント"""
        target_key = self._get_pynput_key()
        if key == target_key and self._current_session is None:
            self._start_session()

    def _on_key_release(self, key):
        """キー離しイベント"""
        target_key = self._get_pynput_key()
        if key == target_key and self._current_session is not None:
            self._stop_session()

    def _start_session(self):
        """セッション開始"""
        self._current_session = VoiceInputSession(
            config=self.config,
            on_partial=self._on_partial_result,
            on_final=self._on_final_result,
        )
        self._current_session.start()

    def _stop_session(self):
        """セッション停止"""
        if self._current_session:
            final_text = self._current_session.stop()
            self._current_session = None

            if final_text:
                self._output_text(final_text)

    def _on_partial_result(self, text: str):
        """部分結果コールバック"""
        if self.config.show_realtime:
            # リアルタイム表示（ターミナルに表示）
            print(f"\r🎤 {text}", end="", flush=True)

    def _on_final_result(self, text: str):
        """最終結果コールバック"""
        if self.config.show_realtime:
            print()  # 改行

    def _output_text(self, text: str):
        """テキストを出力"""
        if not text:
            return

        success = False

        if self.config.output_mode in (OutputMode.CLIPBOARD, OutputMode.BOTH):
            ClipboardManager.copy(text)

        if self.config.output_mode in (OutputMode.TYPE, OutputMode.BOTH):
            # 少し待ってからタイプ（キー離しの処理時間を確保）
            time.sleep(0.1)
            success = KeyboardTyper.type_text(text)

        if self._on_output:
            self._on_output(text)

    def start(self):
        """マネージャー開始"""
        if self._is_running:
            return

        from pynput import keyboard

        self._is_running = True
        self._keyboard_listener = keyboard.Listener(
            on_press=self._on_key_press,
            on_release=self._on_key_release,
        )
        self._keyboard_listener.start()

    def stop(self):
        """マネージャー停止"""
        if self._current_session:
            self._current_session.stop()
            self._current_session = None

        if self._keyboard_listener:
            self._keyboard_listener.stop()
            self._keyboard_listener = None

        self._is_running = False

    def __enter__(self):
        self.start()
        return self

    def __exit__(self, exc_type, exc_val, exc_tb):
        self.stop()


def check_dependencies() -> dict[str, bool]:
    """依存関係のチェック"""
    deps = {}

    # pynput
    try:
        import pynput
        deps["pynput"] = True
    except ImportError:
        deps["pynput"] = False

    # クリップボード (Linux)
    if sys.platform == "linux":
        deps["xclip"] = subprocess.run(
            ["which", "xclip"], capture_output=True
        ).returncode == 0
        deps["xsel"] = subprocess.run(
            ["which", "xsel"], capture_output=True
        ).returncode == 0
        deps["wl-copy"] = subprocess.run(
            ["which", "wl-copy"], capture_output=True
        ).returncode == 0

    # キーボード入力シミュレーション (Linux)
    if sys.platform == "linux":
        deps["xdotool"] = subprocess.run(
            ["which", "xdotool"], capture_output=True
        ).returncode == 0
        deps["ydotool"] = subprocess.run(
            ["which", "ydotool"], capture_output=True
        ).returncode == 0
        deps["wtype"] = subprocess.run(
            ["which", "wtype"], capture_output=True
        ).returncode == 0

    return deps
