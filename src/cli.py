#!/usr/bin/env python3
"""
whisper-realtime CLI
リアルタイム音声文字起こしのコマンドラインインターフェース
"""

import signal
import sys
import threading
import time
from dataclasses import dataclass
from pathlib import Path
from typing import Optional

import click
import numpy as np
from rich.console import Console, Group
from rich.live import Live
from rich.panel import Panel
from rich.table import Table
from rich.text import Text
from rich.layout import Layout
from rich.style import Style

from .audio_capture import AudioCapture, AudioConfig, AudioSource, VADFilter
from .diarization import DiarizationManager
from .dictionary import (
    Dictionary,
    create_example_dictionary,
    get_default_dictionary_path,
    load_or_create_dictionary,
)
from .whisper_engine import (
    MODEL_PROFILES,
    StreamingWhisperEngine,
    WhisperConfig,
    WhisperEngine,
    WhisperModel,
)

console = Console()


@dataclass
class ConversationEntry:
    """会話エントリー"""
    speaker: str
    text: str
    timestamp: float


# 話者ごとの色
SPEAKER_COLORS = [
    "cyan",
    "green",
    "yellow",
    "magenta",
    "blue",
    "red",
]


class RealtimeDisplay:
    """リアルタイム表示マネージャー（スタック表示対応）"""

    def __init__(self, show_speaker: bool = False, max_history: int = 50):
        self.show_speaker = show_speaker
        self.max_history = max_history
        self.partial_text = ""
        self.current_speaker = ""
        self.conversation_history: list[ConversationEntry] = []
        self._speaker_colors: dict[str, str] = {}
        self._color_index = 0
        self._lock = threading.Lock()

    def _get_speaker_color(self, speaker: str) -> str:
        """話者の色を取得（なければ割り当て）"""
        if speaker not in self._speaker_colors:
            self._speaker_colors[speaker] = SPEAKER_COLORS[self._color_index % len(SPEAKER_COLORS)]
            self._color_index += 1
        return self._speaker_colors[speaker]

    def update(self, text: str, is_partial: bool = False, speaker: str = ""):
        """テキストを更新"""
        with self._lock:
            if speaker:
                self.current_speaker = speaker

            if is_partial:
                self.partial_text = text
            else:
                # 確定テキストを会話履歴に追加
                final_text = self.partial_text if self.partial_text else text
                if final_text:
                    entry = ConversationEntry(
                        speaker=self.current_speaker or "話者",
                        text=final_text,
                        timestamp=time.time(),
                    )
                    self.conversation_history.append(entry)

                    # 履歴の上限を超えたら古いものを削除
                    if len(self.conversation_history) > self.max_history:
                        self.conversation_history = self.conversation_history[-self.max_history:]

                self.partial_text = ""

                # 新しいテキストがあれば追加
                if text and text != final_text:
                    entry = ConversationEntry(
                        speaker=self.current_speaker or "話者",
                        text=text,
                        timestamp=time.time(),
                    )
                    self.conversation_history.append(entry)

    def _render_history(self) -> Panel:
        """会話履歴パネルを生成"""
        content = Text()

        if not self.conversation_history:
            content.append("(会話履歴なし)", style="dim")
        else:
            # 最新の会話を表示
            display_entries = self.conversation_history[-30:]  # 最新30件

            for entry in display_entries:
                if self.show_speaker:
                    color = self._get_speaker_color(entry.speaker)
                    content.append(f"[{entry.speaker}] ", style=f"bold {color}")
                content.append(f"{entry.text}\n")

        return Panel(
            content,
            title="[bold blue]会話履歴[/bold blue]",
            border_style="blue",
            padding=(0, 1),
        )

    def _render_live(self) -> Panel:
        """リアルタイム文字起こしパネルを生成"""
        content = Text()

        if self.partial_text:
            if self.show_speaker and self.current_speaker:
                color = self._get_speaker_color(self.current_speaker)
                content.append(f"[{self.current_speaker}] ", style=f"bold {color}")
            content.append(self.partial_text, style="italic")
        else:
            content.append("🎤 音声を待機中...", style="dim")

        return Panel(
            content,
            title="[bold green]リアルタイム[/bold green]",
            subtitle="[dim]Ctrl+C で終了[/dim]",
            border_style="green",
            padding=(0, 1),
        )

    def render(self) -> Group:
        """表示用コンテンツを生成（履歴 + リアルタイム）"""
        with self._lock:
            return Group(
                self._render_history(),
                self._render_live(),
            )

    def get_full_text(self) -> str:
        """全テキストを取得"""
        with self._lock:
            lines = []
            for entry in self.conversation_history:
                if self.show_speaker:
                    lines.append(f"[{entry.speaker}] {entry.text}")
                else:
                    lines.append(entry.text)
            if self.partial_text:
                lines.append(self.partial_text)
            return "\n".join(lines)


def list_audio_devices():
    """オーディオデバイス一覧を表示"""
    devices = AudioCapture.list_devices()

    table = Table(title="利用可能なオーディオデバイス")
    table.add_column("ID", style="cyan")
    table.add_column("名前", style="green")
    table.add_column("チャンネル", style="yellow")
    table.add_column("デフォルト", style="magenta")

    for device in devices:
        table.add_row(
            str(device["id"]),
            device["name"],
            str(device["channels"]),
            "★" if device["is_default"] else "",
        )

    console.print(table)

    # システム音声キャプチャのステータス
    console.print("\n[bold]システム音声キャプチャ:[/bold]")

    # ScreenCaptureKit確認
    try:
        from .system_audio_capture import is_screencapturekit_available, get_screencapturekit_error
        if is_screencapturekit_available():
            console.print("  [green]✓ ScreenCaptureKit[/green] - BlackHole不要でシステム音声をキャプチャ可能")
        else:
            error = get_screencapturekit_error() or "不明なエラー"
            console.print(f"  [yellow]○ ScreenCaptureKit[/yellow] - 利用不可: {error}")
    except ImportError:
        console.print("  [dim]○ ScreenCaptureKit[/dim] - モジュール未インストール")

    # BlackHole検出
    blackhole_id = AudioCapture.find_blackhole_device()
    if blackhole_id is not None:
        console.print(f"  [green]✓ BlackHole[/green] - デバイスID: {blackhole_id}")
    else:
        console.print("  [dim]○ BlackHole[/dim] - 未インストール")

    console.print("\n[dim]ScreenCaptureKitを有効にするには:[/dim]")
    console.print("  uv pip install 'whisper-realtime[macos]'")
    console.print("  システム設定 > プライバシー > 画面収録 で許可")


def list_models(models_path: Path):
    """利用可能なモデル一覧を表示"""
    table = Table(title="利用可能なWhisperモデル")
    table.add_column("モデル", style="cyan")
    table.add_column("サイズ", style="yellow")
    table.add_column("ステータス", style="green")
    table.add_column("推奨用途", style="magenta")

    model_info = {
        "tiny": ("~75MB", "最速、リアルタイム向け"),
        "tiny.en": ("~75MB", "英語のみ、最速"),
        "base": ("~142MB", "バランス型"),
        "base.en": ("~142MB", "英語のみ、バランス"),
        "small": ("~466MB", "高精度"),
        "small.en": ("~466MB", "英語のみ、高精度"),
        "medium": ("~1.5GB", "より高精度"),
        "medium.en": ("~1.5GB", "英語のみ"),
        "large-v1": ("~2.9GB", "最高精度"),
        "large-v2": ("~2.9GB", "最高精度v2"),
        "large-v3": ("~2.9GB", "最高精度v3"),
        "large-v3-turbo": ("~1.5GB", "高精度+高速"),
    }

    for model in WhisperModel:
        model_file = models_path / f"ggml-{model.value}.bin"
        exists = model_file.exists()
        size, usage = model_info.get(model.value, ("?", "?"))

        table.add_row(
            model.value,
            size,
            "[green]インストール済[/green]" if exists else "[dim]未インストール[/dim]",
            usage,
        )

    console.print(table)
    console.print("\nモデルのダウンロード:")
    console.print("  ./setup.sh --model <model-name>")


@click.group()
def cli():
    """whisper-realtime: リアルタイム音声文字起こし"""
    pass


@cli.command()
def devices():
    """オーディオデバイス一覧を表示"""
    list_audio_devices()


@cli.command()
@click.option("--device", "-d", type=int, help="マイクデバイスID")
@click.option("--duration", type=int, default=5, help="テスト時間（秒）")
def test_mic(device: Optional[int], duration: int):
    """マイク入力をテスト（音声レベルを表示）"""
    from rich.progress import Progress, SpinnerColumn, TextColumn, BarColumn

    audio_config = AudioConfig(sample_rate=16000, chunk_duration=0.1)

    console.print(f"[bold]マイクテスト[/bold] ({duration}秒間)")
    console.print("話しかけてください...\n")

    try:
        audio_capture = AudioCapture(
            config=audio_config,
            source=AudioSource.MICROPHONE,
            device_id=device,
        )
    except Exception as e:
        console.print(f"[red]エラー: マイクを開けません: {e}[/red]")
        console.print("\n[yellow]ヒント:[/yellow]")
        console.print("  1. システム環境設定 > プライバシーとセキュリティ > マイク")
        console.print("  2. ターミナルアプリにマイクアクセスを許可")
        return

    start_time = time.time()
    max_level = 0

    with audio_capture:
        while time.time() - start_time < duration:
            audio = audio_capture.get_audio(timeout=0.2)
            if audio is not None and len(audio) > 0:
                # RMSレベルを計算
                rms = np.sqrt(np.mean(audio ** 2))
                level = min(int(rms * 500), 50)  # 0-50のバー
                max_level = max(max_level, level)

                # レベルメーター表示
                bar = "█" * level + "░" * (50 - level)
                console.print(f"\r[green]{bar}[/green] {rms:.4f}", end="")

    console.print("\n")

    if max_level > 5:
        console.print(f"[green]✓ マイクは正常に動作しています[/green] (最大レベル: {max_level})")
    elif max_level > 0:
        console.print(f"[yellow]△ 音声レベルが低いです[/yellow] (最大レベル: {max_level})")
        console.print("  マイクに近づいて話してみてください")
    else:
        console.print("[red]✗ 音声が検出されませんでした[/red]")
        console.print("\n[yellow]確認事項:[/yellow]")
        console.print("  1. マイクがミュートになっていないか確認")
        console.print("  2. システム環境設定でマイク入力を確認")
        console.print("  3. uv run whisper-realtime devices で正しいデバイスを確認")


@cli.command()
@click.option("--path", type=click.Path(exists=True), help="モデルディレクトリ")
def models(path: Optional[str]):
    """利用可能なモデル一覧を表示"""
    models_path = Path(path) if path else Path(__file__).parent.parent / "models"
    list_models(models_path)


@cli.command()
@click.option(
    "--source", "-s",
    type=click.Choice(["mic", "system", "both"]),
    default="mic",
    help="音声入力ソース",
)
@click.option(
    "--model", "-m",
    type=click.Choice([m.value for m in WhisperModel]),
    default="base",
    help="使用するWhisperモデル",
)
@click.option(
    "--profile", "-p",
    type=click.Choice(list(MODEL_PROFILES.keys())),
    help="モデルプロファイル (realtime/balanced/quality/best)",
)
@click.option(
    "--language", "-l",
    default="ja",
    help="言語コード (ja, en, auto)",
)
@click.option(
    "--device", "-d",
    type=int,
    help="マイクデバイスID",
)
@click.option(
    "--system-device",
    type=int,
    help="システム音声デバイスID (BlackHole)",
)
@click.option(
    "--speaker/--no-speaker",
    default=False,
    help="話者分離を有効化",
)
@click.option(
    "--translate/--no-translate",
    default=False,
    help="英語に翻訳",
)
@click.option(
    "--output", "-o",
    type=click.Path(),
    help="出力ファイルパス",
)
@click.option(
    "--step",
    type=int,
    default=500,
    help="処理ステップ (ms)",
)
@click.option(
    "--length",
    type=int,
    default=3000,
    help="処理窓の長さ (ms) - 短いほど反応が早い",
)
@click.option(
    "--vad/--no-vad",
    default=True,
    help="音声区間検出を使用",
)
@click.option(
    "--debug/--no-debug",
    default=False,
    help="デバッグモード（処理状況を表示）",
)
def start(
    source: str,
    model: str,
    profile: Optional[str],
    language: str,
    device: Optional[int],
    system_device: Optional[int],
    speaker: bool,
    translate: bool,
    output: Optional[str],
    step: int,
    length: int,
    vad: bool,
    debug: bool,
):
    """リアルタイム文字起こしを開始"""
    import os

    # デバッグモードの場合、環境変数を設定
    if debug:
        os.environ["WHISPER_DEBUG"] = "1"

    # モデル設定
    if profile:
        whisper_model = MODEL_PROFILES[profile]
    else:
        whisper_model = WhisperModel(model)

    # 設定表示
    console.print(Panel.fit(
        f"[bold]設定[/bold]\n"
        f"モデル: {whisper_model.value}\n"
        f"言語: {language}\n"
        f"ソース: {source}\n"
        f"話者分離: {'有効' if speaker else '無効'}\n"
        f"処理ステップ: {step}ms\n"
        f"処理窓: {length}ms",
        title="whisper-realtime",
        border_style="blue",
    ))

    # 音声ソース設定
    audio_source = {
        "mic": AudioSource.MICROPHONE,
        "system": AudioSource.SYSTEM,
        "both": AudioSource.BOTH,
    }[source]

    # Whisper設定（Streaming最適化パラメータ）
    whisper_config = WhisperConfig(
        model=whisper_model,
        language=language,
        translate=translate,
        step_ms=step,
        length_ms=length,
        # Streaming高速化オプション
        beam_size=1,  # Greedy search for speed
        max_tokens=32,
        use_flash_attn=True,
        no_timestamps=True,
    )

    # 話者分離マネージャー
    diarizer = DiarizationManager(use_pyannote=False) if speaker else None

    # 表示マネージャー
    display = RealtimeDisplay(show_speaker=speaker)

    # VADフィルター
    vad_filter = VADFilter() if vad else None

    # 音声キャプチャ設定
    audio_config = AudioConfig(
        sample_rate=16000,
        chunk_duration=step / 1000,
    )

    # 終了フラグ
    running = True

    def signal_handler(sig, frame):
        nonlocal running
        running = False
        console.print("\n[yellow]終了中...[/yellow]")

    signal.signal(signal.SIGINT, signal_handler)

    # Whisperエンジン初期化
    try:
        engine = WhisperEngine(whisper_config)
    except FileNotFoundError as e:
        console.print(f"[red]エラー: {e}[/red]")
        console.print("\n[yellow]setup.sh を実行してセットアップしてください[/yellow]")
        sys.exit(1)

    # 結果コールバック
    def on_result(result):
        text = result.text
        is_partial = not result.is_final

        if speaker and diarizer:
            current_speaker = diarizer._current_speaker
        else:
            current_speaker = ""

        display.update(text, is_partial=is_partial, speaker=current_speaker)

    engine.set_callback(on_result)

    # 音声キャプチャ開始
    try:
        audio_capture = AudioCapture(
            config=audio_config,
            source=audio_source,
            device_id=device,
            system_device_id=system_device,
        )
    except Exception as e:
        console.print(f"[red]音声キャプチャエラー: {e}[/red]")
        sys.exit(1)

    console.print("\n[green]録音開始... (Ctrl+C で終了)[/green]")
    if debug:
        console.print("[dim]デバッグモード: 処理状況を表示します[/dim]")
    console.print()

    audio_chunks_received = 0
    last_debug_time = time.time()

    try:
        with audio_capture:
            with Live(display.render(), console=console, refresh_per_second=4) as live:
                while running:
                    # 音声データを取得
                    audio = audio_capture.get_audio(timeout=0.1)

                    if audio is not None and len(audio) > 0:
                        audio_chunks_received += 1

                        # デバッグ: 音声レベル表示
                        if debug and time.time() - last_debug_time > 1.0:
                            rms = np.sqrt(np.mean(audio ** 2))
                            buffer_dur = engine.get_buffer_duration()
                            console.print(
                                f"[dim]音声レベル: {rms:.4f} | "
                                f"バッファ: {buffer_dur:.1f}s / {length/1000:.1f}s | "
                                f"チャンク: {audio_chunks_received}[/dim]"
                            )
                            last_debug_time = time.time()

                        # VADフィルタリング
                        if vad_filter and vad_filter.enabled:
                            if not vad_filter.is_speech(audio):
                                if debug:
                                    pass  # 無音スキップ
                                continue

                        # 話者分離
                        if speaker and diarizer:
                            diarizer.process_audio(audio)

                        # Whisperエンジンにデータ追加
                        engine.add_audio(audio)

                        # 十分なデータが溜まったら処理
                        buffer_duration = engine.get_buffer_duration()
                        if buffer_duration >= length / 1000:
                            if debug:
                                console.print(f"[dim]→ 文字起こし実行中... ({buffer_duration:.1f}s)[/dim]")
                            engine.process_realtime()

                    # 表示更新
                    live.update(display.render())

                # 最終処理
                final = engine.finalize()
                if final:
                    display.update(final.text, is_partial=False)
                    live.update(display.render())

    except Exception as e:
        console.print(f"[red]エラー: {e}[/red]")
        import traceback
        traceback.print_exc()

    # 結果出力
    full_text = display.get_full_text()

    if output:
        output_path = Path(output)
        output_path.write_text(full_text, encoding="utf-8")
        console.print(f"\n[green]出力保存: {output_path}[/green]")

    console.print("\n[bold]文字起こし結果:[/bold]")
    console.print(Panel(full_text or "(なし)", border_style="green"))


@cli.command()
@click.option(
    "--model", "-m",
    type=click.Choice([m.value for m in WhisperModel]),
    default="base",
    help="使用するWhisperモデル",
)
@click.option(
    "--language", "-l",
    default="ja",
    help="言語コード",
)
@click.option(
    "--device", "-d",
    type=int,
    help="オーディオデバイスID",
)
def stream(model: str, language: str, device: Optional[int]):
    """
    whisper.cpp stream を使用したストリーミング文字起こし
    (whisper.cpp の stream バイナリが必要)
    """

    whisper_model = WhisperModel(model)
    config = WhisperConfig(model=whisper_model, language=language)

    display = RealtimeDisplay()

    def on_text(text: str, is_partial: bool):
        display.update(text, is_partial=is_partial)

    console.print("[green]ストリーミング開始... (Ctrl+C で終了)[/green]\n")

    try:
        engine = StreamingWhisperEngine(config)
        engine.set_callback(on_text)

        with Live(display.render(), console=console, refresh_per_second=4) as live:
            engine.start(capture_id=device)

            while engine.is_running:
                time.sleep(0.1)
                live.update(display.render())

    except FileNotFoundError as e:
        console.print(f"[red]エラー: {e}[/red]")
        console.print("\n[yellow]whisper.cpp の stream バイナリをビルドしてください[/yellow]")
    except KeyboardInterrupt:
        pass

    console.print("\n[bold]結果:[/bold]")
    console.print(display.get_full_text())


@cli.command()
@click.option(
    "--hotkey", "-k",
    type=click.Choice([
        "ctrl_r", "ctrl_l", "alt_r", "alt_l",
        "shift_r", "shift_l",
        "f1", "f2", "f3", "f4", "f5", "f6",
        "f7", "f8", "f9", "f10", "f11", "f12",
        "caps_lock", "scroll_lock", "pause",
    ]),
    default="ctrl_r",
    help="録音開始/停止のホットキー",
)
@click.option(
    "--model", "-m",
    type=click.Choice([m.value for m in WhisperModel]),
    default="base",
    help="使用するWhisperモデル",
)
@click.option(
    "--language", "-l",
    default="ja",
    help="言語コード",
)
@click.option(
    "--device", "-d",
    type=int,
    help="マイクデバイスID",
)
@click.option(
    "--dictionary/--no-dictionary",
    default=True,
    help="辞書機能を使用",
)
@click.option(
    "--dictionary-path",
    type=click.Path(),
    help="辞書ファイルパス",
)
@click.option(
    "--gui/--no-gui",
    default=False,
    help="GUIウィンドウを表示",
)
@click.option(
    "--menubar/--no-menubar",
    default=False,
    help="メニューバー常駐モード (macOS専用)",
)
@click.option(
    "--phonetic/--no-phonetic",
    default=True,
    help="音声認識誤り訂正を使用（同音異義語・カタカナ正規化など）",
)
def voice(
    hotkey: str,
    model: str,
    language: str,
    device: Optional[int],
    dictionary: bool,
    dictionary_path: Optional[str],
    gui: bool,
    menubar: bool,
    phonetic: bool,
):
    """
    Push-to-Talk 音声入力モード (Aqua Voice風)

    指定したホットキーを押している間、音声を録音し、
    離すと文字起こし結果をクリップボードにコピーします。

    使用例:
        whisper-realtime voice                    # 右Ctrlで録音
        whisper-realtime voice -k f9              # F9で録音
        whisper-realtime voice -m large-v3-turbo  # 高精度モデルを使用
        whisper-realtime voice --gui              # GUIウィンドウを表示
        whisper-realtime voice --menubar          # メニューバー常駐モード (macOS)
    """
    from .voice_input import (
        HotkeyType,
        OutputMode,
        VoiceInputConfig,
        VoiceInputManager,
        check_dependencies,
    )

    # 依存関係チェック
    deps = check_dependencies()

    if not deps.get("pynput", False):
        console.print("[red]エラー: pynput がインストールされていません[/red]")
        console.print("インストール: uv pip install pynput")
        sys.exit(1)

    # Linux環境でのツールチェック
    if sys.platform == "linux":
        has_clipboard = deps.get("xclip") or deps.get("xsel") or deps.get("wl-copy")

        if not has_clipboard:
            console.print("[yellow]警告: クリップボードツールがありません[/yellow]")
            console.print("  インストール: sudo apt install xclip")

    # ホットキー変換
    hotkey_map = {
        "ctrl_r": HotkeyType.CTRL_RIGHT,
        "ctrl_l": HotkeyType.CTRL_LEFT,
        "alt_r": HotkeyType.ALT_RIGHT,
        "alt_l": HotkeyType.ALT_LEFT,
        "shift_r": HotkeyType.SHIFT_RIGHT,
        "shift_l": HotkeyType.SHIFT_LEFT,
        "f1": HotkeyType.F1, "f2": HotkeyType.F2, "f3": HotkeyType.F3,
        "f4": HotkeyType.F4, "f5": HotkeyType.F5, "f6": HotkeyType.F6,
        "f7": HotkeyType.F7, "f8": HotkeyType.F8, "f9": HotkeyType.F9,
        "f10": HotkeyType.F10, "f11": HotkeyType.F11, "f12": HotkeyType.F12,
        "caps_lock": HotkeyType.CAPS_LOCK,
        "scroll_lock": HotkeyType.SCROLL_LOCK,
        "pause": HotkeyType.PAUSE,
    }

    # 設定
    config = VoiceInputConfig(
        hotkey=hotkey_map[hotkey],
        output_mode=OutputMode.CLIPBOARD,
        model=WhisperModel(model),
        language=language,
        device_id=device,
        use_dictionary=dictionary,
        dictionary_path=Path(dictionary_path) if dictionary_path else None,
        use_phonetic_correction=phonetic,
    )

    hotkey_display = hotkey.replace("_", " ").title()

    # メニューバーモード (macOS専用)
    if menubar:
        if sys.platform != "darwin":
            console.print("[red]エラー: メニューバーモードはmacOS専用です[/red]")
            sys.exit(1)

        try:
            from .voice_menubar import MenuBarVoiceInputManager
        except ImportError as e:
            console.print(f"[red]エラー: {e}[/red]")
            console.print("インストール: uv pip install 'whisper-realtime[macos]'")
            sys.exit(1)

        console.print(f"[green]メニューバーモードで起動中... (ホットキー: {hotkey_display})[/green]")

        try:
            menubar_manager = MenuBarVoiceInputManager(config)
            menubar_manager.run()
        except KeyboardInterrupt:
            pass
        finally:
            console.print("\n[yellow]終了しました[/yellow]")
        return

    # GUIモード
    if gui:
        from .voice_gui import GUIVoiceInputManager

        console.print(f"[green]GUIモードで起動中... (ホットキー: {hotkey_display})[/green]")

        try:
            gui_manager = GUIVoiceInputManager(config)
            gui_manager.run()  # メインスレッドでGUI実行
        except KeyboardInterrupt:
            pass
        finally:
            console.print("\n[yellow]終了しました[/yellow]")
        return

    # CLIモード
    console.print(Panel.fit(
        f"[bold]Push-to-Talk 音声入力[/bold]\n\n"
        f"ホットキー: [cyan]{hotkey_display}[/cyan]\n"
        f"モデル: {model}\n"
        f"言語: {language}\n"
        f"辞書: {'有効' if dictionary else '無効'}\n\n"
        f"[dim]ホットキーを押している間、音声を録音します\n"
        f"離すと文字起こし結果をクリップボードにコピーします\n"
        f"Ctrl+C で終了[/dim]",
        title="whisper-realtime voice",
        border_style="green",
    ))

    # 辞書ファイルパス表示
    if dictionary:
        dict_path = Path(dictionary_path) if dictionary_path else get_default_dictionary_path()
        console.print(f"[dim]辞書ファイル: {dict_path}[/dim]\n")

    try:
        manager = VoiceInputManager(config)

        def on_output(text: str):
            console.print(f"[green]✓[/green] {text}")

        manager.set_output_callback(on_output)

        with manager:
            console.print("[green]準備完了！ホットキーを押して録音を開始してください[/green]")
            # メインループ
            while True:
                time.sleep(0.1)

    except KeyboardInterrupt:
        console.print("\n[yellow]終了しました[/yellow]")


@cli.group()
def dictionary():
    """辞書機能の管理"""
    pass


@dictionary.command(name="show")
@click.option(
    "--path",
    type=click.Path(),
    help="辞書ファイルパス",
)
def dictionary_show(path: Optional[str]):
    """現在の辞書を表示"""
    dict_path = Path(path) if path else get_default_dictionary_path()

    if not dict_path.exists():
        console.print(f"[yellow]辞書ファイルが見つかりません: {dict_path}[/yellow]")
        console.print("'whisper-realtime dictionary init' で作成できます")
        return

    dictionary = Dictionary.from_json(dict_path)
    data = dictionary.to_dict()

    console.print(f"[bold]辞書ファイル:[/bold] {dict_path}\n")

    # 単純置換ルール
    if data.get("replacements"):
        table = Table(title="単純置換ルール")
        table.add_column("置換元", style="cyan")
        table.add_column("置換先", style="green")
        table.add_column("正規表現", style="yellow")

        for rule in data["replacements"]:
            table.add_row(
                rule["pattern"],
                rule["replacement"],
                "○" if rule.get("is_regex") else "",
            )
        console.print(table)
        console.print()

    # 文脈ルール
    if data.get("context_rules"):
        table = Table(title="文脈に応じた置換ルール")
        table.add_column("置換元", style="cyan")
        table.add_column("置換先", style="green")
        table.add_column("文脈キーワード", style="magenta")
        table.add_column("除外キーワード", style="red")

        for rule in data["context_rules"]:
            table.add_row(
                rule["pattern"],
                rule["replacement"],
                ", ".join(rule.get("context_keywords", [])),
                ", ".join(rule.get("negative_keywords", [])),
            )
        console.print(table)


@dictionary.command(name="init")
@click.option(
    "--path",
    type=click.Path(),
    help="辞書ファイルパス",
)
@click.option(
    "--force/--no-force", "-f",
    default=False,
    help="既存ファイルを上書き",
)
def dictionary_init(path: Optional[str], force: bool):
    """サンプル辞書を作成"""
    dict_path = Path(path) if path else get_default_dictionary_path()

    if dict_path.exists() and not force:
        console.print(f"[yellow]辞書ファイルが既に存在します: {dict_path}[/yellow]")
        console.print("上書きするには -f オプションを使用してください")
        return

    # 親ディレクトリ作成
    dict_path.parent.mkdir(parents=True, exist_ok=True)

    # サンプル辞書作成
    example_data = create_example_dictionary()
    dictionary = Dictionary.from_dict(example_data)
    dictionary.save_json(dict_path)

    console.print(f"[green]辞書ファイルを作成しました: {dict_path}[/green]")
    console.print("\n[dim]このファイルを編集して、カスタム置換ルールを追加できます[/dim]")


@dictionary.command(name="add")
@click.argument("pattern")
@click.argument("replacement")
@click.option(
    "--context", "-c",
    multiple=True,
    help="文脈キーワード（複数指定可）",
)
@click.option(
    "--path",
    type=click.Path(),
    help="辞書ファイルパス",
)
def dictionary_add(pattern: str, replacement: str, context: tuple, path: Optional[str]):
    """辞書にルールを追加

    例:
        whisper-realtime dictionary add 家具 KAG -c 会社 -c 開発
    """
    dict_path = Path(path) if path else get_default_dictionary_path()

    # 辞書読み込み
    if dict_path.exists():
        dictionary = Dictionary.from_json(dict_path)
    else:
        dict_path.parent.mkdir(parents=True, exist_ok=True)
        dictionary = Dictionary()

    data = dictionary.to_dict()

    # ルール追加
    if context:
        # 文脈ルール
        data["context_rules"].append({
            "pattern": pattern,
            "replacement": replacement,
            "context_keywords": list(context),
            "negative_keywords": [],
            "window_size": 50,
        })
        console.print(f"[green]文脈ルールを追加:[/green] {pattern} → {replacement}")
        console.print(f"  文脈キーワード: {', '.join(context)}")
    else:
        # 単純置換ルール
        data["replacements"].append({
            "pattern": pattern,
            "replacement": replacement,
            "is_regex": False,
        })
        console.print(f"[green]単純置換ルールを追加:[/green] {pattern} → {replacement}")

    # 保存
    dictionary = Dictionary.from_dict(data)
    dictionary.save_json(dict_path)
    console.print(f"[dim]保存: {dict_path}[/dim]")


@dictionary.command(name="test")
@click.argument("text")
@click.option(
    "--path",
    type=click.Path(),
    help="辞書ファイルパス",
)
def dictionary_test(text: str, path: Optional[str]):
    """辞書による変換をテスト

    例:
        whisper-realtime dictionary test "家具という会社について"
    """
    dict_path = Path(path) if path else get_default_dictionary_path()

    if not dict_path.exists():
        console.print(f"[yellow]辞書ファイルが見つかりません: {dict_path}[/yellow]")
        return

    dictionary = Dictionary.from_json(dict_path)
    result = dictionary.apply(text)

    console.print(f"[dim]入力:[/dim] {text}")
    console.print(f"[green]出力:[/green] {result}")

    if text == result:
        console.print("[dim]（変換なし）[/dim]")


@cli.command(name="voice-single")
@click.option(
    "--model", "-m",
    type=click.Choice([m.value for m in WhisperModel]),
    default="base",
    help="使用するWhisperモデル",
)
@click.option(
    "--language", "-l",
    default="ja",
    help="言語コード",
)
@click.option(
    "--device", "-d",
    type=int,
    help="マイクデバイスID",
)
@click.option(
    "--dictionary/--no-dictionary",
    default=True,
    help="辞書機能を使用",
)
@click.option(
    "--dictionary-path",
    type=click.Path(),
    help="辞書ファイルパス",
)
@click.option(
    "--phonetic/--no-phonetic",
    default=True,
    help="音声認識誤り訂正を使用（同音異義語・カタカナ正規化など）",
)
def voice_single(
    model: str,
    language: str,
    device: Optional[int],
    dictionary: bool,
    dictionary_path: Optional[str],
    phonetic: bool,
):
    """
    シングルショット音声入力モード（外部アプリ連携用）

    起動時に録音を開始し、SIGINT (Ctrl+C) で録音を終了して結果を出力。
    SwiftのMenuBarアプリなど外部から呼び出すためのコマンド。

    出力形式:
        stdout: PARTIAL:<部分結果> / FINAL:<最終結果>
        stderr: デバッグログ（タイムスタンプ付き）
    """
    import logging

    from .voice_input import VoiceInputConfig, VoiceInputSession, ClipboardManager, HotkeyType
    from .dictionary import load_or_create_dictionary

    # voice_input.pyでlogging設定済みなのでloggerを取得
    logger = logging.getLogger('voice_single')
    logger.info("=== voice-single started ===")
    logger.info(f"  Model: {model}")
    logger.info(f"  Language: {language}")
    logger.info(f"  Dictionary: {dictionary}")
    logger.info(f"  PhoneticCorrection: {phonetic}")

    whisper_model = WhisperModel(model)

    # 設定
    config = VoiceInputConfig(
        hotkey=HotkeyType.CTRL_RIGHT,  # 使用しないがデフォルト設定
        model=whisper_model,
        language=language,
        device_id=device,
        use_dictionary=dictionary,
        dictionary_path=Path(dictionary_path) if dictionary_path else None,
        use_phonetic_correction=phonetic,
    )

    # 終了フラグ
    running = True
    session = None

    def on_partial(text: str):
        """部分結果コールバック"""
        logger.debug(f"PARTIAL output: '{text[:50]}...' ({len(text)} chars)")
        print(f"PARTIAL:{text}", flush=True)

    def on_final(text: str):
        """最終結果コールバック"""
        pass  # stop時に処理

    def signal_handler(sig, frame):
        nonlocal running
        logger.info(f"Signal received: {sig}")
        running = False

    signal.signal(signal.SIGINT, signal_handler)
    signal.signal(signal.SIGTERM, signal_handler)

    # セッション作成と録音開始
    logger.info("Creating VoiceInputSession...")
    session = VoiceInputSession(
        config=config,
        on_partial=on_partial,
        on_final=on_final,
    )

    logger.info("Starting recording...")
    session.start()

    # SIGINTまで待機
    try:
        while running:
            time.sleep(0.1)
    except KeyboardInterrupt:
        logger.info("KeyboardInterrupt received")

    # 録音停止と結果取得
    logger.info("Stopping recording...")
    final_text = session.stop()

    if final_text:
        # クリップボードにコピー
        logger.info(f"Copying to clipboard: '{final_text}'")
        ClipboardManager.copy(final_text)
        print(f"FINAL:{final_text}", flush=True)
    else:
        logger.warning("No final text")
        print("FINAL:", flush=True)

    logger.info("=== voice-single ended ===")


def main():
    """エントリーポイント"""
    cli()


if __name__ == "__main__":
    main()
