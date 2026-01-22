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

from .audio_capture import AudioCapture, AudioConfig, AudioSource, VADFilter
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
    ):
        self.config = config
        self.on_partial = on_partial
        self.on_final = on_final

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

        # 辞書読み込み
        if config.use_dictionary:
            self._dictionary = load_or_create_dictionary(config.dictionary_path)

        # 音声認識誤り訂正機能初期化
        if config.use_phonetic_correction:
            self._phonetic_corrector = PhoneticCorrector()

    def start(self):
        """録音開始"""
        if self._is_recording:
            return

        logger.info("=== Recording session START ===")
        logger.info(f"  Model: {self.config.model.value}")
        logger.info(f"  Language: {self.config.language}")
        logger.info(f"  Dictionary: {self.config.use_dictionary}")
        logger.info(f"  PhoneticCorrection: {self.config.use_phonetic_correction}")

        self._is_recording = True
        self._audio_buffer = []
        self._partial_text = ""
        self._last_processed_samples = 0

        # Whisperエンジン初期化（Streaming最適化パラメータ）
        logger.debug("Initializing WhisperEngine...")
        whisper_config = WhisperConfig(
            model=self.config.model,
            language=self.config.language,
            # Streaming最適化設定
            step_ms=500,  # 500ms毎に処理
            length_ms=3000,  # 3秒の窓（レイテンシ削減）
            keep_ms=200,  # コンテキスト保持
            beam_size=1,  # Greedy search（高速）
            max_tokens=32,
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
        vad = VADFilter() if self.config.use_vad else None
        chunk_count = 0

        try:
            with self._capture:
                while self._is_recording:
                    audio = self._capture.get_audio(timeout=0.1)
                    if audio is not None and len(audio) > 0:
                        # VADフィルタリング（完全にスキップせず、音声ありの部分のみ記録）
                        if vad and vad.enabled:
                            if vad.is_speech(audio):
                                self._audio_buffer.append(audio)
                                chunk_count += 1
                        else:
                            self._audio_buffer.append(audio)
                            chunk_count += 1

                        if chunk_count % 10 == 0:
                            total_samples = sum(len(a) for a in self._audio_buffer)
                            logger.debug(f"Audio: {chunk_count} chunks, {total_samples} samples ({total_samples/16000:.1f}s)")
        except Exception as e:
            logger.error(f"Recording error: {e}")

        logger.debug(f"_record_loop ended, total chunks: {chunk_count}")

    def _process_loop(self):
        """リアルタイム処理ループ - 定期的に部分結果を生成"""
        logger.debug("_process_loop started")
        process_count = 0

        while self._is_recording:
            time.sleep(0.5)  # 500ms ごとに処理

            if not self._audio_buffer:
                continue

            # 現在のバッファ全体を結合
            current_audio = np.concatenate(self._audio_buffer)
            current_samples = len(current_audio)

            # 新しい音声がある場合のみ処理
            if current_samples > self._last_processed_samples + 8000:  # 0.5秒以上の新規音声
                try:
                    process_count += 1
                    logger.debug(f"[Process #{process_count}] Input: {current_samples} samples ({current_samples/16000:.1f}s)")

                    # バッファ全体を文字起こし（1センテンスとして）
                    t0 = time.time()
                    result = self._engine.transcribe_audio(current_audio)
                    whisper_time = time.time() - t0
                    logger.debug(f"[Process #{process_count}] Whisper: {whisper_time:.2f}s")

                    if result and result.text.strip():
                        text = result.text.strip()
                        logger.info(f"[PARTIAL #{process_count}] 🎤 Whisper Raw: '{text}'")

                        # 辞書適用
                        if self._dictionary:
                            before_dict = text
                            text = self._dictionary.apply(text)
                            if text != before_dict:
                                logger.info(f"[PARTIAL #{process_count}] 📖 Dictionary: '{before_dict}' → '{text}'")

                        # 音声認識誤り訂正
                        if self._phonetic_corrector:
                            before_corr = text
                            result = self._phonetic_corrector.correct(text)
                            text = result.corrected_text
                            if text != before_corr:
                                logger.info(f"[PARTIAL #{process_count}] 🔊 Phonetic: '{before_corr}' → '{text}'")

                        self._partial_text = text
                        if self.on_partial:
                            self.on_partial(text)
                    self._last_processed_samples = current_samples
                except Exception as e:
                    logger.error(f"Processing error: {e}")

        logger.debug(f"_process_loop ended, total processes: {process_count}")

    def stop(self) -> str:
        """録音停止して最終テキストを返す"""
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

        # 最終処理：バッファ全体を一括で文字起こし
        final_text = ""
        if self._audio_buffer:
            full_audio = np.concatenate(self._audio_buffer)
            total_duration = len(full_audio) / 16000
            logger.info(f"[FINAL] 🎙️ Audio: {total_duration:.1f}s ({len(full_audio)} samples)")

            if len(full_audio) > 1600:  # 0.1秒以上の音声がある場合
                try:
                    t0 = time.time()
                    result = self._engine.transcribe_audio(full_audio)
                    logger.info(f"[FINAL] ⏱️ Whisper処理時間: {time.time()-t0:.2f}s")
                    if result and result.text.strip():
                        final_text = result.text.strip()
                        logger.info(f"[FINAL] 🎤 Whisper Raw: '{final_text}'")
                except Exception as e:
                    logger.error(f"[FINAL] ❌ Transcription error: {e}")
                    final_text = self._partial_text

        # 部分結果がある場合はそれを使用
        if not final_text:
            final_text = self._partial_text
            logger.info(f"[FINAL] ⚠️ Using partial text: '{final_text}'")

        # 辞書適用
        if self._dictionary and final_text:
            before_dict = final_text
            final_text = self._dictionary.apply(final_text)
            if final_text != before_dict:
                logger.info(f"[FINAL] 📖 Dictionary: '{before_dict}' → '{final_text}'")

        # 音声認識誤り訂正
        if self._phonetic_corrector and final_text:
            before_corr = final_text
            correction_result = self._phonetic_corrector.correct(final_text)
            final_text = correction_result.corrected_text
            if final_text != before_corr:
                logger.info(f"[FINAL] 🔊 Phonetic: '{before_corr}' → '{final_text}'")
                if correction_result.corrections:
                    for c in correction_result.corrections:
                        logger.info(f"[FINAL]    └─ {c.get('type', '?')}: {c.get('description', '')}")

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
