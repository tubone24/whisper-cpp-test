#!/usr/bin/env python3
"""
whisper-realtime CLI
Real-time audio transcription command-line interface
"""

import json
import signal
import sys
import threading
import time
from dataclasses import dataclass
from enum import Enum
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


class OutputFormat(str, Enum):
    """Output format"""
    RICH = "rich"
    JSON = "json"
    PLAIN = "plain"

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
    """Conversation entry"""
    speaker: str
    text: str
    timestamp: float


# Speaker colors
SPEAKER_COLORS = [
    "cyan",
    "green",
    "yellow",
    "magenta",
    "blue",
    "red",
]


class RealtimeDisplay:
    """Real-time display manager with stack display support"""

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
        """Get speaker color (assign if not exists)"""
        if speaker not in self._speaker_colors:
            self._speaker_colors[speaker] = SPEAKER_COLORS[self._color_index % len(SPEAKER_COLORS)]
            self._color_index += 1
        return self._speaker_colors[speaker]

    def update(self, text: str, is_partial: bool = False, speaker: str = ""):
        """Update text"""
        with self._lock:
            if speaker:
                self.current_speaker = speaker

            if is_partial:
                self.partial_text = text
            else:
                # Add confirmed text to conversation history
                final_text = self.partial_text if self.partial_text else text
                if final_text:
                    entry = ConversationEntry(
                        speaker=self.current_speaker or "Speaker",
                        text=final_text,
                        timestamp=time.time(),
                    )
                    self.conversation_history.append(entry)

                    # Remove old entries if exceeds max
                    if len(self.conversation_history) > self.max_history:
                        self.conversation_history = self.conversation_history[-self.max_history:]

                self.partial_text = ""

                # Add new text if different
                if text and text != final_text:
                    entry = ConversationEntry(
                        speaker=self.current_speaker or "Speaker",
                        text=text,
                        timestamp=time.time(),
                    )
                    self.conversation_history.append(entry)

    def _render_history(self) -> Panel:
        """Generate conversation history panel"""
        content = Text()

        if not self.conversation_history:
            content.append("(No conversation history)", style="dim")
        else:
            # Show latest conversations
            display_entries = self.conversation_history[-30:]  # Latest 30

            for entry in display_entries:
                if self.show_speaker:
                    color = self._get_speaker_color(entry.speaker)
                    content.append(f"[{entry.speaker}] ", style=f"bold {color}")
                content.append(f"{entry.text}\n")

        return Panel(
            content,
            title="[bold blue]History[/bold blue]",
            border_style="blue",
            padding=(0, 1),
        )

    def _render_live(self) -> Panel:
        """Generate real-time transcription panel"""
        content = Text()

        if self.partial_text:
            if self.show_speaker and self.current_speaker:
                color = self._get_speaker_color(self.current_speaker)
                content.append(f"[{self.current_speaker}] ", style=f"bold {color}")
            content.append(self.partial_text, style="italic")
        else:
            content.append("Waiting for audio...", style="dim")

        return Panel(
            content,
            title="[bold green]Real-time[/bold green]",
            subtitle="[dim]Ctrl+C to stop[/dim]",
            border_style="green",
            padding=(0, 1),
        )

    def render(self) -> Group:
        """Generate display content (history + real-time)"""
        with self._lock:
            return Group(
                self._render_history(),
                self._render_live(),
            )

    def get_full_text(self) -> str:
        """Get full text"""
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


class JSONOutput:
    """JSON Lines output manager (for Raycast Extension)"""

    def __init__(self, show_speaker: bool = False):
        self.show_speaker = show_speaker
        self.conversation_history: list[ConversationEntry] = []
        self._lock = threading.Lock()
        self._start_time = time.time()

    def _emit(self, data: dict):
        """Emit JSON line"""
        print(json.dumps(data, ensure_ascii=False), flush=True)

    def update(self, text: str, is_partial: bool = False, speaker: str = ""):
        """Update text and emit JSON"""
        with self._lock:
            timestamp = time.time() - self._start_time

            if is_partial:
                self._emit({
                    "type": "partial",
                    "speaker": speaker or "",
                    "text": text,
                    "timestamp": round(timestamp, 2),
                })
            else:
                # Final text
                entry = ConversationEntry(
                    speaker=speaker or "Speaker",
                    text=text,
                    timestamp=time.time(),
                )
                self.conversation_history.append(entry)

                self._emit({
                    "type": "final",
                    "speaker": speaker or "",
                    "text": text,
                    "timestamp": round(timestamp, 2),
                })

    def emit_status(self, status: str, message: str = ""):
        """Emit status message"""
        self._emit({
            "type": "status",
            "status": status,
            "message": message,
            "timestamp": round(time.time() - self._start_time, 2),
        })

    def emit_error(self, error: str):
        """Emit error message"""
        self._emit({
            "type": "error",
            "error": error,
            "timestamp": round(time.time() - self._start_time, 2),
        })

    def emit_level(self, level: float):
        """Emit audio level"""
        self._emit({
            "type": "level",
            "level": round(level, 3),
            "timestamp": round(time.time() - self._start_time, 2),
        })

    def get_full_text(self) -> str:
        """Get full text"""
        with self._lock:
            lines = []
            for entry in self.conversation_history:
                if self.show_speaker:
                    lines.append(f"[{entry.speaker}] {entry.text}")
                else:
                    lines.append(entry.text)
            return "\n".join(lines)


def list_audio_devices():
    """List available audio devices"""
    devices = AudioCapture.list_devices()

    table = Table(title="Available Audio Devices")
    table.add_column("ID", style="cyan")
    table.add_column("Name", style="green")
    table.add_column("Channels", style="yellow")
    table.add_column("Default", style="magenta")

    for device in devices:
        table.add_row(
            str(device["id"]),
            device["name"],
            str(device["channels"]),
            "★" if device["is_default"] else "",
        )

    console.print(table)

    # System audio capture status
    console.print("\n[bold]System Audio Capture:[/bold]")

    # ScreenCaptureKit check
    try:
        from .system_audio_capture import is_screencapturekit_available, get_screencapturekit_error
        if is_screencapturekit_available():
            console.print("  [green]✓ ScreenCaptureKit[/green] - Can capture system audio without BlackHole")
        else:
            error = get_screencapturekit_error() or "Unknown error"
            console.print(f"  [yellow]○ ScreenCaptureKit[/yellow] - Not available: {error}")
    except ImportError:
        console.print("  [dim]○ ScreenCaptureKit[/dim] - Module not installed")

    # BlackHole detection
    blackhole_id = AudioCapture.find_blackhole_device()
    if blackhole_id is not None:
        console.print(f"  [green]✓ BlackHole[/green] - Device ID: {blackhole_id}")
    else:
        console.print("  [dim]○ BlackHole[/dim] - Not installed")

    console.print("\n[dim]To enable ScreenCaptureKit:[/dim]")
    console.print("  uv pip install 'whisper-realtime[macos]'")
    console.print("  System Preferences > Privacy > Screen Recording")


def list_models(models_path: Path):
    """List available models"""
    table = Table(title="Available Whisper Models")
    table.add_column("Model", style="cyan")
    table.add_column("Size", style="yellow")
    table.add_column("Status", style="green")
    table.add_column("Recommended Use", style="magenta")

    model_info = {
        "tiny": ("~75MB", "Fastest, real-time"),
        "tiny.en": ("~75MB", "English only, fastest"),
        "base": ("~142MB", "Balanced"),
        "base.en": ("~142MB", "English only, balanced"),
        "small": ("~466MB", "High accuracy"),
        "small.en": ("~466MB", "English only, high accuracy"),
        "medium": ("~1.5GB", "Higher accuracy"),
        "medium.en": ("~1.5GB", "English only"),
        "large-v1": ("~2.9GB", "Best accuracy"),
        "large-v2": ("~2.9GB", "Best accuracy v2"),
        "large-v3": ("~2.9GB", "Best accuracy v3"),
        "large-v3-turbo": ("~1.5GB", "High accuracy + fast"),
    }

    for model in WhisperModel:
        model_file = models_path / f"ggml-{model.value}.bin"
        exists = model_file.exists()
        size, usage = model_info.get(model.value, ("?", "?"))

        table.add_row(
            model.value,
            size,
            "[green]Installed[/green]" if exists else "[dim]Not installed[/dim]",
            usage,
        )

    console.print(table)
    console.print("\nDownload models:")
    console.print("  ./setup.sh --model <model-name>")


@click.group()
def cli():
    """whisper-realtime: Real-time audio transcription"""
    pass


@cli.command()
def devices():
    """List available audio devices"""
    list_audio_devices()


@cli.command()
@click.option("--device", "-d", type=int, help="Microphone device ID")
@click.option("--duration", type=int, default=5, help="Test duration (seconds)")
def test_mic(device: Optional[int], duration: int):
    """Test microphone input (show audio level)"""
    from rich.progress import Progress, SpinnerColumn, TextColumn, BarColumn

    audio_config = AudioConfig(sample_rate=16000, chunk_duration=0.1)

    console.print(f"[bold]Microphone Test[/bold] ({duration} seconds)")
    console.print("Please speak...\n")

    try:
        audio_capture = AudioCapture(
            config=audio_config,
            source=AudioSource.MICROPHONE,
            device_id=device,
        )
    except Exception as e:
        console.print(f"[red]Error: Cannot open microphone: {e}[/red]")
        console.print("\n[yellow]Hint:[/yellow]")
        console.print("  1. System Preferences > Privacy & Security > Microphone")
        console.print("  2. Allow terminal app to access microphone")
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
        console.print(f"[green]✓ Microphone is working correctly[/green] (max level: {max_level})")
    elif max_level > 0:
        console.print(f"[yellow]△ Audio level is low[/yellow] (max level: {max_level})")
        console.print("  Try speaking closer to the microphone")
    else:
        console.print("[red]✗ No audio detected[/red]")
        console.print("\n[yellow]Check:[/yellow]")
        console.print("  1. Make sure microphone is not muted")
        console.print("  2. Check microphone input in system settings")
        console.print("  3. Run 'uv run whisper-realtime devices' to check device")


@cli.command()
@click.option("--path", type=click.Path(exists=True), help="Model directory")
def models(path: Optional[str]):
    """List available models"""
    models_path = Path(path) if path else Path(__file__).parent.parent / "models"
    list_models(models_path)


@cli.command()
@click.option(
    "--source", "-s",
    type=click.Choice(["mic", "system", "both"]),
    default="mic",
    help="Audio input source",
)
@click.option(
    "--model", "-m",
    type=click.Choice([m.value for m in WhisperModel]),
    default="base",
    help="Whisper model to use",
)
@click.option(
    "--profile", "-p",
    type=click.Choice(list(MODEL_PROFILES.keys())),
    help="Model profile (realtime/balanced/quality/best)",
)
@click.option(
    "--language", "-l",
    default="ja",
    help="Language code (ja, en, auto)",
)
@click.option(
    "--device", "-d",
    type=int,
    help="Microphone device ID",
)
@click.option(
    "--system-device",
    type=int,
    help="System audio device ID (BlackHole)",
)
@click.option(
    "--speaker/--no-speaker",
    default=False,
    help="Enable speaker diarization",
)
@click.option(
    "--translate/--no-translate",
    default=False,
    help="Translate to English",
)
@click.option(
    "--output", "-o",
    type=click.Path(),
    help="Output file path",
)
@click.option(
    "--step",
    type=int,
    default=500,
    help="Processing step (ms)",
)
@click.option(
    "--length",
    type=int,
    default=3000,
    help="Processing window length (ms) - shorter means faster response",
)
@click.option(
    "--vad/--no-vad",
    default=True,
    help="Use voice activity detection",
)
@click.option(
    "--debug/--no-debug",
    default=False,
    help="Debug mode (show processing status)",
)
@click.option(
    "--output-format", "-f",
    type=click.Choice(["rich", "json", "plain"]),
    default="rich",
    help="Output format (rich: terminal UI, json: JSON Lines, plain: plain text)",
)
@click.option(
    "--record", "-r",
    type=click.Path(),
    help="Record audio and save to WAV file (specify path)",
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
    output_format: str,
    record: Optional[str],
):
    """Start real-time transcription"""
    import os

    # 出力形式の判定
    use_json = output_format == "json"
    use_plain = output_format == "plain"

    # デバッグモードの場合、環境変数を設定
    if debug:
        os.environ["WHISPER_DEBUG"] = "1"

    # モデル設定
    if profile:
        whisper_model = MODEL_PROFILES[profile]
    else:
        whisper_model = WhisperModel(model)

    # 録音用バッファ
    recorded_audio: list = [] if record else None

    # 出力マネージャー（形式に応じて切り替え）
    if use_json:
        display = JSONOutput(show_speaker=speaker)
        display.emit_status("starting", f"model={whisper_model.value}, language={language}, record={record or 'disabled'}")
    else:
        display = RealtimeDisplay(show_speaker=speaker)
        # Show settings (rich/plain format only)
        if not use_plain:
            settings_lines = [
                f"[bold]Settings[/bold]",
                f"Model: {whisper_model.value}",
                f"Language: {language}",
                f"Source: {source}",
                f"Speaker diarization: {'Enabled' if speaker else 'Disabled'}",
                f"Processing step: {step}ms",
                f"Processing window: {length}ms",
            ]
            if record:
                settings_lines.append(f"[green]Recording: {record}[/green]")
            console.print(Panel.fit(
                "\n".join(settings_lines),
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
        if use_json:
            display.emit_status("stopping")
        else:
            console.print("\n[yellow]Stopping...[/yellow]")

    signal.signal(signal.SIGINT, signal_handler)

    # Initialize Whisper engine
    try:
        engine = WhisperEngine(whisper_config)
    except FileNotFoundError as e:
        if use_json:
            display.emit_error(str(e))
        else:
            console.print(f"[red]Error: {e}[/red]")
            console.print("\n[yellow]Please run setup.sh to complete setup[/yellow]")
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

    # Start audio capture
    try:
        audio_capture = AudioCapture(
            config=audio_config,
            source=audio_source,
            device_id=device,
            system_device_id=system_device,
        )
    except Exception as e:
        if use_json:
            display.emit_error(f"Audio capture error: {e}")
        else:
            console.print(f"[red]Audio capture error: {e}[/red]")
        sys.exit(1)

    if use_json:
        display.emit_status("recording")
    else:
        console.print("\n[green]Recording started... (Ctrl+C to stop)[/green]")
        if debug:
            console.print("[dim]Debug mode: showing processing status[/dim]")
        console.print()

    audio_chunks_received = 0
    last_debug_time = time.time()

    try:
        with audio_capture:
            # JSON形式の場合はLiveを使わない
            if use_json:
                while running:
                    audio = audio_capture.get_audio(timeout=0.1)

                    if audio is not None and len(audio) > 0:
                        audio_chunks_received += 1

                        # 音声レベルを計算して出力
                        rms = np.sqrt(np.mean(audio ** 2))
                        level = min(float(rms * 10), 1.0)
                        display.emit_level(level)

                        # 録音用に音声を蓄積（VAD前の生データ）
                        if recorded_audio is not None:
                            recorded_audio.append(audio.copy())

                        # VADフィルタリング
                        if vad_filter and vad_filter.enabled:
                            if not vad_filter.is_speech(audio):
                                continue

                        # 話者分離
                        if speaker and diarizer:
                            diarizer.process_audio(audio)

                        # Whisperエンジンにデータ追加
                        engine.add_audio(audio)

                        # 十分なデータが溜まったら処理
                        buffer_duration = engine.get_buffer_duration()
                        if buffer_duration >= length / 1000:
                            engine.process_realtime()

                # 最終処理
                final = engine.finalize()
                if final:
                    display.update(final.text, is_partial=False)

            else:
                # Rich形式の場合はLive表示
                with Live(display.render(), console=console, refresh_per_second=4) as live:
                    while running:
                        audio = audio_capture.get_audio(timeout=0.1)

                        if audio is not None and len(audio) > 0:
                            audio_chunks_received += 1

                            # 録音用に音声を蓄積（VAD前の生データ）
                            if recorded_audio is not None:
                                recorded_audio.append(audio.copy())

                            # Debug: show audio level
                            if debug and time.time() - last_debug_time > 1.0:
                                rms = np.sqrt(np.mean(audio ** 2))
                                buffer_dur = engine.get_buffer_duration()
                                console.print(
                                    f"[dim]Audio level: {rms:.4f} | "
                                    f"Buffer: {buffer_dur:.1f}s / {length/1000:.1f}s | "
                                    f"Chunks: {audio_chunks_received}[/dim]"
                                )
                                last_debug_time = time.time()

                            # VADフィルタリング
                            if vad_filter and vad_filter.enabled:
                                if not vad_filter.is_speech(audio):
                                    continue

                            # 話者分離
                            if speaker and diarizer:
                                diarizer.process_audio(audio)

                            # Whisperエンジンにデータ追加
                            engine.add_audio(audio)

                            # Process when enough data is buffered
                            buffer_duration = engine.get_buffer_duration()
                            if buffer_duration >= length / 1000:
                                if debug:
                                    console.print(f"[dim]→ Transcribing... ({buffer_duration:.1f}s)[/dim]")
                                engine.process_realtime()

                        # 表示更新
                        live.update(display.render())

                    # 最終処理
                    final = engine.finalize()
                    if final:
                        display.update(final.text, is_partial=False)
                        live.update(display.render())

    except Exception as e:
        if use_json:
            display.emit_error(str(e))
        else:
            console.print(f"[red]Error: {e}[/red]")
            import traceback
            traceback.print_exc()

    # 結果出力
    full_text = display.get_full_text()

    if use_json:
        display.emit_status("completed", full_text)
    else:
        if output:
            output_path = Path(output)
            output_path.write_text(full_text, encoding="utf-8")
            console.print(f"\n[green]Output saved: {output_path}[/green]")

        console.print("\n[bold]Transcription result:[/bold]")
        console.print(Panel(full_text or "(none)", border_style="green"))

    # ファイル出力（json形式でも-oオプション指定時は出力）
    if output and use_json:
        output_path = Path(output)
        output_path.write_text(full_text, encoding="utf-8")

    # 録音した音声をWAVファイルに保存
    if record and recorded_audio:
        import wave

        record_path = Path(record)
        # 拡張子が.wavでなければ追加
        if record_path.suffix.lower() != ".wav":
            record_path = record_path.with_suffix(".wav")

        # 録音データを結合
        all_audio = np.concatenate(recorded_audio)
        # float32からint16に変換
        audio_int16 = (all_audio * 32767).astype(np.int16)

        with wave.open(str(record_path), "wb") as wf:
            wf.setnchannels(1)
            wf.setsampwidth(2)  # 16-bit
            wf.setframerate(16000)
            wf.writeframes(audio_int16.tobytes())

        duration_sec = len(all_audio) / 16000
        if use_json:
            display.emit_status("recorded", f"saved={record_path}, duration={duration_sec:.1f}s")
        else:
            console.print(f"\n[green]Recording saved: {record_path} ({duration_sec:.1f}s)[/green]")


@cli.command()
@click.option(
    "--model", "-m",
    type=click.Choice([m.value for m in WhisperModel]),
    default="base",
    help="Whisper model to use",
)
@click.option(
    "--language", "-l",
    default="ja",
    help="Language code",
)
@click.option(
    "--device", "-d",
    type=int,
    help="Audio device ID",
)
def stream(model: str, language: str, device: Optional[int]):
    """
    Streaming transcription using whisper.cpp stream
    (requires whisper.cpp stream binary)
    """

    whisper_model = WhisperModel(model)
    config = WhisperConfig(model=whisper_model, language=language)

    display = RealtimeDisplay()

    def on_text(text: str, is_partial: bool):
        display.update(text, is_partial=is_partial)

    console.print("[green]Streaming started... (Ctrl+C to stop)[/green]\n")

    try:
        engine = StreamingWhisperEngine(config)
        engine.set_callback(on_text)

        with Live(display.render(), console=console, refresh_per_second=4) as live:
            engine.start(capture_id=device)

            while engine.is_running:
                time.sleep(0.1)
                live.update(display.render())

    except FileNotFoundError as e:
        console.print(f"[red]Error: {e}[/red]")
        console.print("\n[yellow]Please build the whisper.cpp stream binary[/yellow]")
    except KeyboardInterrupt:
        pass

    console.print("\n[bold]Result:[/bold]")
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
    help="Hotkey to start/stop recording",
)
@click.option(
    "--model", "-m",
    type=click.Choice([m.value for m in WhisperModel]),
    default="base",
    help="Whisper model to use",
)
@click.option(
    "--language", "-l",
    default="ja",
    help="Language code",
)
@click.option(
    "--device", "-d",
    type=int,
    help="Microphone device ID",
)
@click.option(
    "--dictionary/--no-dictionary",
    default=True,
    help="Use dictionary feature",
)
@click.option(
    "--dictionary-path",
    type=click.Path(),
    help="Dictionary file path",
)
@click.option(
    "--gui/--no-gui",
    default=False,
    help="Show GUI window",
)
@click.option(
    "--menubar/--no-menubar",
    default=False,
    help="Menu bar mode (macOS only)",
)
@click.option(
    "--phonetic/--no-phonetic",
    default=True,
    help="Use ASR error correction (homophone disambiguation, katakana normalization, etc.)",
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
    Push-to-Talk voice input mode (Aqua Voice style)

    Records audio while holding the specified hotkey,
    and copies transcription to clipboard when released.

    Examples:
        whisper-realtime voice                    # Record with Right Ctrl
        whisper-realtime voice -k f9              # Record with F9
        whisper-realtime voice -m large-v3-turbo  # Use high-accuracy model
        whisper-realtime voice --gui              # Show GUI window
        whisper-realtime voice --menubar          # Menu bar mode (macOS)
    """
    from .voice_input import (
        HotkeyType,
        OutputMode,
        VoiceInputConfig,
        VoiceInputManager,
        check_dependencies,
    )

    # Check dependencies
    deps = check_dependencies()

    if not deps.get("pynput", False):
        console.print("[red]Error: pynput is not installed[/red]")
        console.print("Install: uv pip install pynput")
        sys.exit(1)

    # Check tools for Linux environment
    if sys.platform == "linux":
        has_clipboard = deps.get("xclip") or deps.get("xsel") or deps.get("wl-copy")

        if not has_clipboard:
            console.print("[yellow]Warning: No clipboard tool available[/yellow]")
            console.print("  Install: sudo apt install xclip")

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

    # Menu bar mode (macOS only)
    if menubar:
        if sys.platform != "darwin":
            console.print("[red]Error: Menu bar mode is only available on macOS[/red]")
            sys.exit(1)

        try:
            from .voice_menubar import MenuBarVoiceInputManager
        except ImportError as e:
            console.print(f"[red]Error: {e}[/red]")
            console.print("Install: uv pip install 'whisper-realtime[macos]'")
            sys.exit(1)

        console.print(f"[green]Starting menu bar mode... (Hotkey: {hotkey_display})[/green]")

        try:
            menubar_manager = MenuBarVoiceInputManager(config)
            menubar_manager.run()
        except KeyboardInterrupt:
            pass
        finally:
            console.print("\n[yellow]Exited[/yellow]")
        return

    # GUI mode
    if gui:
        from .voice_gui import GUIVoiceInputManager

        console.print(f"[green]Starting GUI mode... (Hotkey: {hotkey_display})[/green]")

        try:
            gui_manager = GUIVoiceInputManager(config)
            gui_manager.run()  # Run GUI in main thread
        except KeyboardInterrupt:
            pass
        finally:
            console.print("\n[yellow]Exited[/yellow]")
        return

    # CLI mode
    console.print(Panel.fit(
        f"[bold]Push-to-Talk Voice Input[/bold]\n\n"
        f"Hotkey: [cyan]{hotkey_display}[/cyan]\n"
        f"Model: {model}\n"
        f"Language: {language}\n"
        f"Dictionary: {'Enabled' if dictionary else 'Disabled'}\n\n"
        f"[dim]Hold the hotkey to record audio\n"
        f"Release to copy transcription to clipboard\n"
        f"Ctrl+C to exit[/dim]",
        title="whisper-realtime voice",
        border_style="green",
    ))

    # Show dictionary file path
    if dictionary:
        dict_path = Path(dictionary_path) if dictionary_path else get_default_dictionary_path()
        console.print(f"[dim]Dictionary file: {dict_path}[/dim]\n")

    try:
        manager = VoiceInputManager(config)

        def on_output(text: str):
            console.print(f"[green]✓[/green] {text}")

        manager.set_output_callback(on_output)

        with manager:
            console.print("[green]Ready! Press hotkey to start recording[/green]")
            # Main loop
            while True:
                time.sleep(0.1)

    except KeyboardInterrupt:
        console.print("\n[yellow]Exited[/yellow]")


@cli.group()
def dictionary():
    """Manage dictionary feature"""
    pass


@dictionary.command(name="show")
@click.option(
    "--path",
    type=click.Path(),
    help="Dictionary file path",
)
def dictionary_show(path: Optional[str]):
    """Show current dictionary"""
    dict_path = Path(path) if path else get_default_dictionary_path()

    if not dict_path.exists():
        console.print(f"[yellow]Dictionary file not found: {dict_path}[/yellow]")
        console.print("Create with 'whisper-realtime dictionary init'")
        return

    dictionary = Dictionary.from_json(dict_path)
    data = dictionary.to_dict()

    console.print(f"[bold]Dictionary file:[/bold] {dict_path}\n")

    # Simple replacement rules
    if data.get("replacements"):
        table = Table(title="Simple Replacement Rules")
        table.add_column("Pattern", style="cyan")
        table.add_column("Replacement", style="green")
        table.add_column("Regex", style="yellow")

        for rule in data["replacements"]:
            table.add_row(
                rule["pattern"],
                rule["replacement"],
                "Yes" if rule.get("is_regex") else "",
            )
        console.print(table)
        console.print()

    # Context rules
    if data.get("context_rules"):
        table = Table(title="Context-aware Replacement Rules")
        table.add_column("Pattern", style="cyan")
        table.add_column("Replacement", style="green")
        table.add_column("Context Keywords", style="magenta")
        table.add_column("Negative Keywords", style="red")

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
    help="Dictionary file path",
)
@click.option(
    "--force/--no-force", "-f",
    default=False,
    help="Overwrite existing file",
)
def dictionary_init(path: Optional[str], force: bool):
    """Create sample dictionary"""
    dict_path = Path(path) if path else get_default_dictionary_path()

    if dict_path.exists() and not force:
        console.print(f"[yellow]Dictionary file already exists: {dict_path}[/yellow]")
        console.print("Use -f option to overwrite")
        return

    # Create parent directory
    dict_path.parent.mkdir(parents=True, exist_ok=True)

    # Create sample dictionary
    example_data = create_example_dictionary()
    dictionary = Dictionary.from_dict(example_data)
    dictionary.save_json(dict_path)

    console.print(f"[green]Dictionary file created: {dict_path}[/green]")
    console.print("\n[dim]Edit this file to add custom replacement rules[/dim]")


@dictionary.command(name="add")
@click.argument("pattern")
@click.argument("replacement")
@click.option(
    "--context", "-c",
    multiple=True,
    help="Context keywords (can specify multiple)",
)
@click.option(
    "--path",
    type=click.Path(),
    help="Dictionary file path",
)
def dictionary_add(pattern: str, replacement: str, context: tuple, path: Optional[str]):
    """Add rule to dictionary

    Example:
        whisper-realtime dictionary add "kagu" "KAG" -c company -c development
    """
    dict_path = Path(path) if path else get_default_dictionary_path()

    # Load dictionary
    if dict_path.exists():
        dictionary = Dictionary.from_json(dict_path)
    else:
        dict_path.parent.mkdir(parents=True, exist_ok=True)
        dictionary = Dictionary()

    data = dictionary.to_dict()

    # Add rule
    if context:
        # Context rule
        data["context_rules"].append({
            "pattern": pattern,
            "replacement": replacement,
            "context_keywords": list(context),
            "negative_keywords": [],
            "window_size": 50,
        })
        console.print(f"[green]Context rule added:[/green] {pattern} → {replacement}")
        console.print(f"  Context keywords: {', '.join(context)}")
    else:
        # Simple replacement rule
        data["replacements"].append({
            "pattern": pattern,
            "replacement": replacement,
            "is_regex": False,
        })
        console.print(f"[green]Simple rule added:[/green] {pattern} → {replacement}")

    # Save
    dictionary = Dictionary.from_dict(data)
    dictionary.save_json(dict_path)
    console.print(f"[dim]Saved: {dict_path}[/dim]")


@dictionary.command(name="test")
@click.argument("text")
@click.option(
    "--path",
    type=click.Path(),
    help="Dictionary file path",
)
def dictionary_test(text: str, path: Optional[str]):
    """Test dictionary transformation

    Example:
        whisper-realtime dictionary test "kagu company info"
    """
    dict_path = Path(path) if path else get_default_dictionary_path()

    if not dict_path.exists():
        console.print(f"[yellow]Dictionary file not found: {dict_path}[/yellow]")
        return

    dictionary = Dictionary.from_json(dict_path)
    result = dictionary.apply(text)

    console.print(f"[dim]Input:[/dim] {text}")
    console.print(f"[green]Output:[/green] {result}")

    if text == result:
        console.print("[dim](no changes)[/dim]")


@cli.command(name="voice-single")
@click.option(
    "--model", "-m",
    type=click.Choice([m.value for m in WhisperModel]),
    default="base",
    help="Whisper model to use",
)
@click.option(
    "--language", "-l",
    default="ja",
    help="Language code",
)
@click.option(
    "--device", "-d",
    type=int,
    help="Microphone device ID",
)
@click.option(
    "--dictionary/--no-dictionary",
    default=True,
    help="Use dictionary feature",
)
@click.option(
    "--dictionary-path",
    type=click.Path(),
    help="Dictionary file path",
)
@click.option(
    "--phonetic/--no-phonetic",
    default=True,
    help="Use ASR error correction (homophone disambiguation, katakana normalization, etc.)",
)
@click.option(
    "--step", "-s",
    type=int,
    default=500,
    help="Processing step interval in milliseconds (default: 500)",
)
@click.option(
    "--length", "-L",
    type=int,
    default=10000,
    help="Processing window length in milliseconds (default: 10000 for longer texts)",
)
@click.option(
    "--keep", "-k",
    type=int,
    default=500,
    help="Context keep duration in milliseconds (default: 500)",
)
@click.option(
    "--max-tokens", "-t",
    type=int,
    default=128,
    help="Maximum tokens per inference (default: 128 for longer texts)",
)
def voice_single(
    model: str,
    language: str,
    device: Optional[int],
    dictionary: bool,
    dictionary_path: Optional[str],
    phonetic: bool,
    step: int,
    length: int,
    keep: int,
    max_tokens: int,
):
    """
    Single-shot voice input mode (for external app integration)

    Starts recording on launch, stops and outputs result on SIGINT (Ctrl+C).
    Command designed to be called from external apps like Swift MenuBar app.

    Output format:
        stdout: PARTIAL:<partial result> / FINAL:<final result>
        stderr: Debug logs (with timestamps)
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
    logger.info(f"  Step: {step}ms, Length: {length}ms, Keep: {keep}ms, MaxTokens: {max_tokens}")

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
        step_ms=step,
        length_ms=length,
        keep_ms=keep,
        max_tokens=max_tokens,
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

    def on_level(level: float):
        """音声レベルコールバック"""
        print(f"LEVEL:{level:.3f}", flush=True)

    def on_spectrum(spectrum: list[float]):
        """スペクトラムコールバック"""
        # カンマ区切りで出力（例: SPECTRUM:0.1,0.2,0.3,...）
        spectrum_str = ",".join(f"{v:.3f}" for v in spectrum)
        print(f"SPECTRUM:{spectrum_str}", flush=True)

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
        on_level=on_level,
        on_spectrum=on_spectrum,
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
