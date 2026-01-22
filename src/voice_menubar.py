"""
macOSメニューバー常駐モジュール
メニューバーにアイコンを表示し、hotkeyでポップオーバー風ウィンドウを表示
"""

import sys
import threading
import time
from typing import Callable, Optional

# macOS専用
if sys.platform != "darwin":
    raise ImportError("This module is only available on macOS")

try:
    from AppKit import (
        NSApplication,
        NSApplicationActivationPolicyAccessory,
        NSBackingStoreBuffered,
        NSColor,
        NSFont,
        NSMenu,
        NSMenuItem,
        NSScreen,
        NSStatusBar,
        NSTextField,
        NSTextView,
        NSVariableStatusItemLength,
        NSView,
        NSWindow,
        NSWindowCollectionBehaviorCanJoinAllSpaces,
        NSWindowCollectionBehaviorStationary,
        NSBezierPath,
    )
    from Foundation import NSMakeRect, NSRunLoop, NSDate, NSObject
    from objc import super as objc_super
    from dispatch import dispatch_async, dispatch_get_main_queue
except ImportError:
    raise ImportError(
        "PyObjC is required. Install with: uv pip install 'whisper-realtime[macos]'"
    )


# ウィンドウスタイルマスク（ボーダーレス）
NSWindowStyleMaskBorderless = 0


def run_on_main_thread(func):
    """メインスレッドで関数を実行"""
    dispatch_async(dispatch_get_main_queue(), func)


class PopoverContentView(NSView):
    """ポップオーバー風のカスタムビュー（角丸の背景を描画）"""

    def initWithFrame_(self, frame):
        self = objc_super(PopoverContentView, self).initWithFrame_(frame)
        if self is None:
            return None
        return self

    def drawRect_(self, rect):
        """角丸の背景を描画"""
        # 背景色（濃いグレー、少し透明）
        bg_color = NSColor.colorWithCalibratedRed_green_blue_alpha_(
            0.12, 0.12, 0.12, 0.95
        )
        bg_color.set()

        # 角丸の四角形
        path = NSBezierPath.bezierPathWithRoundedRect_xRadius_yRadius_(
            rect, 10.0, 10.0
        )
        path.fill()


class MenuBarApp:
    """macOSメニューバー常駐アプリ"""

    def __init__(
        self,
        hotkey_name: str = "Ctrl R",
        on_quit: Optional[Callable[[], None]] = None,
    ):
        self.hotkey_name = hotkey_name
        self.on_quit = on_quit

        self._status_item = None
        self._popover_window: Optional[NSWindow] = None
        self._status_label: Optional[NSTextField] = None
        self._text_label: Optional[NSTextField] = None
        self._is_visible = False

    def setup(self):
        """メニューバーアイコンをセットアップ"""
        # ステータスバーにアイテムを追加
        status_bar = NSStatusBar.systemStatusBar()
        self._status_item = status_bar.statusItemWithLength_(
            NSVariableStatusItemLength
        )

        # アイコン（🎤 絵文字をテキストとして表示）
        button = self._status_item.button()
        button.setTitle_("🎤")

        # メニューを設定
        menu = NSMenu.alloc().init()

        quit_item = NSMenuItem.alloc().initWithTitle_action_keyEquivalent_(
            "終了", "terminate:", ""
        )
        menu.addItem_(quit_item)

        self._status_item.setMenu_(menu)

        # ポップオーバーウィンドウを作成
        self._create_popover_window()

    def _create_popover_window(self):
        """ポップオーバー風ウィンドウを作成"""
        # ウィンドウサイズ
        width = 350
        height = 80

        # ステータスアイテムの位置を取得し、そのディスプレイを基準にする
        button = self._status_item.button()
        button_window = button.window()

        if button_window:
            # ステータスバーアイコンのあるディスプレイを取得
            button_frame = button_window.frame()
            screen = button_window.screen() or NSScreen.mainScreen()
            screen_frame = screen.frame()

            # アイコンの中央下にポップオーバーを配置
            x = button_frame.origin.x - width / 2 + button_frame.size.width / 2
            # メニューバーの下に配置（画面上端からの相対位置）
            menu_bar_height = screen_frame.size.height + screen_frame.origin.y - button_frame.origin.y
            y = button_frame.origin.y - height - 5
        else:
            # フォールバック: メイン画面の右上
            screen = NSScreen.mainScreen()
            screen_frame = screen.frame()
            x = screen_frame.size.width - width - 50
            y = screen_frame.size.height - 30 - height - 5

        # 画面端をはみ出さないように調整
        screen_min_x = screen_frame.origin.x
        screen_max_x = screen_frame.origin.x + screen_frame.size.width
        if x + width > screen_max_x:
            x = screen_max_x - width - 10
        if x < screen_min_x + 10:
            x = screen_min_x + 10

        # ボーダーレスウィンドウを作成
        self._popover_window = NSWindow.alloc().initWithContentRect_styleMask_backing_defer_(
            NSMakeRect(x, y, width, height),
            NSWindowStyleMaskBorderless,
            NSBackingStoreBuffered,
            False,
        )

        # ウィンドウ設定
        self._popover_window.setLevel_(1000)  # 最前面
        self._popover_window.setOpaque_(False)
        self._popover_window.setBackgroundColor_(NSColor.clearColor())
        self._popover_window.setHasShadow_(True)
        self._popover_window.setCollectionBehavior_(
            NSWindowCollectionBehaviorCanJoinAllSpaces
            | NSWindowCollectionBehaviorStationary
        )

        # カスタム背景ビュー
        content_view = PopoverContentView.alloc().initWithFrame_(
            NSMakeRect(0, 0, width, height)
        )
        self._popover_window.setContentView_(content_view)

        # ステータスラベル（上部）
        self._status_label = NSTextField.alloc().initWithFrame_(
            NSMakeRect(15, height - 30, width - 30, 20)
        )
        self._status_label.setStringValue_(f"🎤 {self.hotkey_name} を押して録音")
        self._status_label.setTextColor_(
            NSColor.colorWithCalibratedRed_green_blue_alpha_(0.53, 0.53, 0.53, 1.0)
        )
        self._status_label.setBackgroundColor_(NSColor.clearColor())
        self._status_label.setBezeled_(False)
        self._status_label.setEditable_(False)
        self._status_label.setSelectable_(False)
        self._status_label.setFont_(NSFont.systemFontOfSize_(12))
        content_view.addSubview_(self._status_label)

        # テキストラベル（メイン）
        self._text_label = NSTextField.alloc().initWithFrame_(
            NSMakeRect(15, 10, width - 30, height - 45)
        )
        self._text_label.setStringValue_("")
        self._text_label.setTextColor_(NSColor.whiteColor())
        self._text_label.setBackgroundColor_(NSColor.clearColor())
        self._text_label.setBezeled_(False)
        self._text_label.setEditable_(False)
        self._text_label.setSelectable_(False)
        self._text_label.setFont_(NSFont.systemFontOfSize_(14))
        content_view.addSubview_(self._text_label)

    def show_popover(self):
        """ポップオーバーを表示（メインスレッドで実行）"""
        def _show():
            if self._popover_window and not self._is_visible:
                self._popover_window.setAlphaValue_(1.0)
                self._popover_window.orderFront_(None)
                self._is_visible = True

        run_on_main_thread(_show)

    def hide_popover(self):
        """ポップオーバーを非表示（メインスレッドで実行）"""
        def _hide():
            if self._popover_window and self._is_visible:
                self._popover_window.orderOut_(None)
                self._is_visible = False

        run_on_main_thread(_hide)

    def set_recording(self, is_recording: bool):
        """録音状態を設定（メインスレッドで実行）"""
        def _update():
            if self._status_label:
                if is_recording:
                    self._status_label.setStringValue_("🔴 録音中...")
                    self._status_label.setTextColor_(
                        NSColor.colorWithCalibratedRed_green_blue_alpha_(1.0, 0.42, 0.42, 1.0)
                    )
                else:
                    self._status_label.setStringValue_(f"🎤 {self.hotkey_name} を押して録音")
                    self._status_label.setTextColor_(
                        NSColor.colorWithCalibratedRed_green_blue_alpha_(0.53, 0.53, 0.53, 1.0)
                    )

        run_on_main_thread(_update)
        if is_recording:
            self.show_popover()

    def set_text(self, text: str):
        """テキストを設定（メインスレッドで実行）"""
        def _update():
            if self._text_label:
                self._text_label.setStringValue_(text)

        run_on_main_thread(_update)

    def set_final_text(self, text: str):
        """最終テキストを設定し、しばらくして非表示（メインスレッドで実行）"""
        def _update():
            if self._status_label:
                self._status_label.setStringValue_("✓ クリップボードにコピーしました")
                self._status_label.setTextColor_(
                    NSColor.colorWithCalibratedRed_green_blue_alpha_(0.42, 0.81, 0.42, 1.0)
                )
            if self._text_label:
                self._text_label.setStringValue_(text)

        run_on_main_thread(_update)

        # 2秒後に非表示
        def hide_later():
            time.sleep(2.0)
            self.hide_popover()

        threading.Thread(target=hide_later, daemon=True).start()

    def update_status_icon(self, is_recording: bool):
        """ステータスバーアイコンを更新（メインスレッドで実行）"""
        def _update():
            if self._status_item:
                button = self._status_item.button()
                button.setTitle_("🔴" if is_recording else "🎤")

        run_on_main_thread(_update)


class MenuBarVoiceInputManager:
    """メニューバー常駐音声入力マネージャー"""

    def __init__(self, config):
        from .voice_input import HotkeyType

        self.config = config
        self._voice_manager = None
        self._menubar_app: Optional[MenuBarApp] = None
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
            if self._menubar_app:
                self._menubar_app.set_recording(True)
                self._menubar_app.set_text("")
                self._menubar_app.update_status_icon(True)
            original_start_session()
            # 部分結果コールバックを上書き
            if self._voice_manager._current_session:
                self._voice_manager._current_session.on_partial = self._on_partial

        def custom_stop_session():
            if self._menubar_app:
                self._menubar_app.update_status_icon(False)
            original_stop_session()

        self._voice_manager._start_session = custom_start_session
        self._voice_manager._stop_session = custom_stop_session

        self._voice_manager.start()

        # 終了を待つ
        while not self._stopping:
            time.sleep(0.1)

        self._voice_manager.stop()

    def _on_partial(self, text: str):
        """部分結果コールバック"""
        if self._menubar_app:
            self._menubar_app.set_text(text)

    def _on_output(self, text: str):
        """最終出力コールバック"""
        if self._menubar_app:
            self._menubar_app.set_final_text(text)

    def _on_quit(self):
        """終了時コールバック"""
        self._stopping = True

    def run(self):
        """実行"""
        # NSApplicationをアクセサリーアプリとして設定（Dockに表示しない）
        app = NSApplication.sharedApplication()
        app.setActivationPolicy_(NSApplicationActivationPolicyAccessory)

        # メニューバーアプリをセットアップ
        self._menubar_app = MenuBarApp(
            hotkey_name=self._hotkey_name,
            on_quit=self._on_quit,
        )
        self._menubar_app.setup()

        # 音声入力マネージャーを別スレッドで開始
        self._voice_thread = threading.Thread(
            target=self._start_voice_manager, daemon=True
        )
        self._voice_thread.start()

        # メインループ
        print(f"🎤 メニューバーで待機中... {self._hotkey_name} を押して録音")
        try:
            while not self._stopping:
                # RunLoopを少しだけ回す
                NSRunLoop.currentRunLoop().runUntilDate_(
                    NSDate.dateWithTimeIntervalSinceNow_(0.1)
                )
        except KeyboardInterrupt:
            pass
        finally:
            self._stopping = True
            if self._voice_thread:
                self._voice_thread.join(timeout=2.0)
