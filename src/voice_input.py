"""
Push-to-Talk 音声入力モジュール
キーを押している間録音し、離すと確定してペースト
"""

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

from .audio_capture import AudioCapture, AudioConfig, AudioSource, VADFilter
from .dictionary import Dictionary, load_or_create_dictionary
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
    output_mode: OutputMode = OutputMode.TYPE
    # 辞書設定
    dictionary_path: Optional[Path] = None
    use_dictionary: bool = True
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
                # AppleScript を使用
                escaped_text = text.replace('"', '\\"')
                script = f'''
                tell application "System Events"
                    keystroke "{escaped_text}"
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

        # 辞書読み込み
        if config.use_dictionary:
            self._dictionary = load_or_create_dictionary(config.dictionary_path)

    def start(self):
        """録音開始"""
        if self._is_recording:
            return

        self._is_recording = True
        self._audio_buffer = []
        self._partial_text = ""

        # Whisperエンジン初期化
        whisper_config = WhisperConfig(
            model=self.config.model,
            language=self.config.language,
            step_ms=500,
            length_ms=3000,
        )
        self._engine = WhisperEngine(whisper_config)
        self._engine.set_callback(self._on_transcription)

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

    def _record_loop(self):
        """録音ループ"""
        vad = VADFilter() if self.config.use_vad else None

        try:
            with self._capture:
                while self._is_recording:
                    audio = self._capture.get_audio(timeout=0.1)
                    if audio is not None and len(audio) > 0:
                        # VADフィルタリング
                        if vad and vad.enabled and not vad.is_speech(audio):
                            continue

                        self._audio_buffer.append(audio)
                        self._engine.add_audio(audio)

                        # リアルタイム処理
                        if self._engine.get_buffer_duration() >= 3.0:
                            self._engine.process_realtime()
        except Exception as e:
            print(f"Recording error: {e}")

    def _on_transcription(self, result: TranscriptionResult):
        """文字起こし結果コールバック"""
        text = result.text
        if self._dictionary:
            text = self._dictionary.apply(text)

        self._partial_text = text
        if self.on_partial:
            self.on_partial(text)

    def stop(self) -> str:
        """録音停止して最終テキストを返す"""
        if not self._is_recording:
            return ""

        self._is_recording = False

        # スレッド終了待ち
        if self._record_thread:
            self._record_thread.join(timeout=2.0)

        # 最終処理
        final_text = self._partial_text
        if self._engine:
            final_result = self._engine.finalize()
            if final_result:
                final_text = final_result.text

        # 辞書適用
        if self._dictionary and final_text:
            final_text = self._dictionary.apply(final_text)

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

        key_mapping = {
            HotkeyType.CTRL_RIGHT: Key.ctrl_r,
            HotkeyType.CTRL_LEFT: Key.ctrl_l,
            HotkeyType.ALT_RIGHT: Key.alt_r,
            HotkeyType.ALT_LEFT: Key.alt_l,
            HotkeyType.SHIFT_RIGHT: Key.shift_r,
            HotkeyType.SHIFT_LEFT: Key.shift_l,
            HotkeyType.F1: Key.f1,
            HotkeyType.F2: Key.f2,
            HotkeyType.F3: Key.f3,
            HotkeyType.F4: Key.f4,
            HotkeyType.F5: Key.f5,
            HotkeyType.F6: Key.f6,
            HotkeyType.F7: Key.f7,
            HotkeyType.F8: Key.f8,
            HotkeyType.F9: Key.f9,
            HotkeyType.F10: Key.f10,
            HotkeyType.F11: Key.f11,
            HotkeyType.F12: Key.f12,
            HotkeyType.CAPS_LOCK: Key.caps_lock,
            HotkeyType.SCROLL_LOCK: Key.scroll_lock,
            HotkeyType.PAUSE: Key.pause,
        }
        return key_mapping.get(self.config.hotkey)

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
