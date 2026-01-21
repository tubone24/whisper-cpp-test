"""
GUI表示モジュール
Push-to-Talk音声入力のリアルタイム表示用フローティングウィンドウ
"""

import threading
import tkinter as tk
from tkinter import font as tkfont
from typing import Callable, Optional


class VoiceInputGUI:
    """音声入力用のフローティングGUIウィンドウ（メインスレッドで実行）"""

    def __init__(
        self,
        on_close: Optional[Callable[[], None]] = None,
        hotkey_name: str = "Ctrl R",
    ):
        self.on_close = on_close
        self.hotkey_name = hotkey_name
        self._root: Optional[tk.Tk] = None
        self._label: Optional[tk.Label] = None
        self._status_label: Optional[tk.Label] = None
        self._is_recording = False
        self._current_text = ""

    def setup(self):
        """GUIをセットアップ（メインスレッドで呼び出す）"""
        self._root = tk.Tk()
        self._root.title("Whisper Voice Input")

        # ウィンドウ設定
        self._root.attributes("-topmost", True)  # 常に最前面
        self._root.overrideredirect(False)  # タイトルバーを表示
        self._root.resizable(True, True)

        # macOS用の設定
        try:
            self._root.attributes("-alpha", 0.95)  # 少し透明
        except tk.TclError:
            pass

        # ウィンドウサイズと位置
        window_width = 400
        window_height = 120
        screen_width = self._root.winfo_screenwidth()
        screen_height = self._root.winfo_screenheight()
        x = (screen_width - window_width) // 2
        y = screen_height - window_height - 100  # 画面下部に配置
        self._root.geometry(f"{window_width}x{window_height}+{x}+{y}")

        # 背景色
        self._root.configure(bg="#1e1e1e")

        # フォント設定
        try:
            main_font = tkfont.Font(family="Hiragino Sans", size=14)
            status_font = tkfont.Font(family="Hiragino Sans", size=11)
        except Exception:
            main_font = tkfont.Font(size=14)
            status_font = tkfont.Font(size=11)

        # ステータスラベル（上部）
        self._status_label = tk.Label(
            self._root,
            text=f"🎤 {self.hotkey_name} を押して録音",
            font=status_font,
            fg="#888888",
            bg="#1e1e1e",
            anchor="w",
            padx=10,
            pady=5,
        )
        self._status_label.pack(fill="x")

        # テキスト表示ラベル（メイン）
        self._label = tk.Label(
            self._root,
            text="",
            font=main_font,
            fg="#ffffff",
            bg="#1e1e1e",
            wraplength=380,
            justify="left",
            anchor="nw",
            padx=10,
            pady=5,
        )
        self._label.pack(fill="both", expand=True)

        # ウィンドウクローズイベント
        self._root.protocol("WM_DELETE_WINDOW", self._on_window_close)

    def _on_window_close(self):
        """ウィンドウが閉じられた時"""
        if self.on_close:
            self.on_close()
        if self._root:
            self._root.quit()

    def run(self):
        """GUIメインループを開始（メインスレッドで呼び出す）"""
        if self._root:
            self._root.mainloop()

    def set_recording(self, is_recording: bool):
        """録音状態を設定（スレッドセーフ）"""
        self._is_recording = is_recording
        if self._root:
            self._root.after(0, self._update_status)

    def set_text(self, text: str):
        """表示テキストを設定（スレッドセーフ）"""
        self._current_text = text
        if self._root:
            self._root.after(0, self._update_text)

    def set_final_text(self, text: str):
        """最終テキストを設定（クリップボードにコピー済み）（スレッドセーフ）"""
        self._current_text = text
        self._is_recording = False
        if self._root:
            self._root.after(0, self._show_final)

    def _update_status(self):
        """ステータス表示を更新"""
        if self._status_label:
            if self._is_recording:
                self._status_label.config(
                    text="🔴 録音中...",
                    fg="#ff6b6b",
                )
            else:
                self._status_label.config(
                    text=f"🎤 {self.hotkey_name} を押して録音",
                    fg="#888888",
                )

    def _update_text(self):
        """テキスト表示を更新"""
        if self._label:
            self._label.config(text=self._current_text)

    def _show_final(self):
        """最終結果を表示"""
        if self._status_label:
            self._status_label.config(
                text="✓ クリップボードにコピーしました",
                fg="#6bcf6b",
            )
        if self._label:
            self._label.config(text=self._current_text)
        # 2秒後にリセット
        if self._root:
            self._root.after(2000, self._reset_display)

    def _reset_display(self):
        """表示をリセット"""
        if self._status_label:
            self._status_label.config(
                text=f"🎤 {self.hotkey_name} を押して録音",
                fg="#888888",
            )
        if self._label:
            self._label.config(text="")
        self._current_text = ""

    def stop(self):
        """GUIを停止（スレッドセーフ）"""
        if self._root:
            self._root.after(0, self._root.quit)


class GUIVoiceInputManager:
    """GUI付きPush-to-Talk音声入力マネージャー"""

    def __init__(self, config):
        from .voice_input import HotkeyType

        self.config = config
        self._voice_manager = None
        self._gui: Optional[VoiceInputGUI] = None
        self._voice_thread: Optional[threading.Thread] = None
        self._stopping = False

        # ホットキー名を取得
        hotkey_names = {
            HotkeyType.CTRL_RIGHT: "Ctrl R",
            HotkeyType.CTRL_LEFT: "Ctrl L",
            HotkeyType.ALT_RIGHT: "Alt R",
            HotkeyType.ALT_LEFT: "Alt L",
            HotkeyType.SHIFT_RIGHT: "Shift R",
            HotkeyType.SHIFT_LEFT: "Shift L",
            HotkeyType.F1: "F1", HotkeyType.F2: "F2", HotkeyType.F3: "F3",
            HotkeyType.F4: "F4", HotkeyType.F5: "F5", HotkeyType.F6: "F6",
            HotkeyType.F7: "F7", HotkeyType.F8: "F8", HotkeyType.F9: "F9",
            HotkeyType.F10: "F10", HotkeyType.F11: "F11", HotkeyType.F12: "F12",
            HotkeyType.CAPS_LOCK: "Caps Lock",
            HotkeyType.SCROLL_LOCK: "Scroll Lock",
            HotkeyType.PAUSE: "Pause",
        }
        self._hotkey_name = hotkey_names.get(config.hotkey, "Key")

    def _start_voice_manager(self):
        """音声入力マネージャーを別スレッドで開始"""
        from .voice_input import VoiceInputManager

        self._voice_manager = VoiceInputManager(self.config)
        self._voice_manager.set_output_callback(self._on_output)

        # カスタムコールバックを設定
        original_start_session = self._voice_manager._start_session
        original_stop_session = self._voice_manager._stop_session

        def custom_start_session():
            if self._gui:
                self._gui.set_recording(True)
                self._gui.set_text("")
            original_start_session()
            # 部分結果コールバックを上書き
            if self._voice_manager._current_session:
                self._voice_manager._current_session.on_partial = self._on_partial

        def custom_stop_session():
            original_stop_session()

        self._voice_manager._start_session = custom_start_session
        self._voice_manager._stop_session = custom_stop_session

        self._voice_manager.start()

        # 終了を待つ
        while not self._stopping:
            import time
            time.sleep(0.1)

        self._voice_manager.stop()

    def _on_partial(self, text: str):
        """部分結果コールバック"""
        if self._gui:
            self._gui.set_text(text)

    def _on_output(self, text: str):
        """最終出力コールバック"""
        if self._gui:
            self._gui.set_final_text(text)

    def _on_gui_close(self):
        """GUIが閉じられた時"""
        self._stopping = True

    def run(self):
        """実行（メインスレッドで呼び出す）"""
        # GUIをセットアップ（メインスレッド）
        self._gui = VoiceInputGUI(
            on_close=self._on_gui_close,
            hotkey_name=self._hotkey_name,
        )
        self._gui.setup()

        # 音声入力マネージャーを別スレッドで開始
        self._voice_thread = threading.Thread(target=self._start_voice_manager, daemon=True)
        self._voice_thread.start()

        # GUIメインループ（メインスレッド）
        try:
            self._gui.run()
        except KeyboardInterrupt:
            pass
        finally:
            self._stopping = True
            if self._voice_thread:
                self._voice_thread.join(timeout=2.0)
