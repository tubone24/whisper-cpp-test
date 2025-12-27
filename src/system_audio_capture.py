"""
システム音声キャプチャモジュール (macOS)
ScreenCaptureKit を使用してBlackHoleなしでシステム音声をキャプチャ

macOS 13.0+ が必要

Note: macOS Sonoma/Sequoia では SCStreamOutput への弱参照問題があるため、
SCStreamOutput と SCStreamDelegate を同一クラスで実装する必要がある。
参考: https://github.com/ronaldoussoren/pyobjc/issues/647
"""

import queue
import sys
import threading
import time
from typing import Optional

import numpy as np

# ScreenCaptureKit用のフラグ
SCREENCAPTUREKIT_AVAILABLE = False
SCK_ERROR_MESSAGE = None

# macOS以外では無効
if sys.platform != "darwin":
    SCK_ERROR_MESSAGE = "ScreenCaptureKit is only available on macOS"
else:
    try:
        import objc
        from Foundation import NSObject

        # pyobjc-framework-ScreenCaptureKit から直接インポート
        try:
            import ScreenCaptureKit
            from ScreenCaptureKit import (
                SCContentFilter,
                SCShareableContent,
                SCStream,
                SCStreamConfiguration,
            )
            SCREENCAPTUREKIT_AVAILABLE = True
        except ImportError as e:
            SCK_ERROR_MESSAGE = (
                f"pyobjc-framework-ScreenCaptureKit が必要です: {e}\n"
                "インストール: uv pip install 'whisper-realtime[macos]'"
            )
    except ImportError as e:
        SCK_ERROR_MESSAGE = f"PyObjC import error: {e}"


class ScreenCaptureKitAudioCapture:
    """
    ScreenCaptureKit を使用したシステム音声キャプチャ

    macOS 13.0 (Ventura) 以降で利用可能
    BlackHole不要でシステム音声をキャプチャできる

    注意: 画面収録の権限が必要です。
    システム環境設定 > プライバシーとセキュリティ > 画面収録
    """

    def __init__(self, sample_rate: int = 16000, chunk_duration: float = 0.5):
        if not SCREENCAPTUREKIT_AVAILABLE:
            raise RuntimeError(
                f"ScreenCaptureKit が利用できません。\n{SCK_ERROR_MESSAGE}\n\n"
                "必要条件:\n"
                "  - macOS 13.0 (Ventura) 以降\n"
                "  - uv pip install 'whisper-realtime[macos]'"
            )

        self.sample_rate = sample_rate
        self.chunk_duration = chunk_duration
        self.audio_queue: queue.Queue = queue.Queue()
        self.is_running = False
        self._stream = None
        self._stream_output = None  # Strong reference to prevent GC
        self._error_message = None
        self._lock = threading.Lock()

    def start(self):
        """キャプチャを開始"""
        if self.is_running:
            return

        import objc
        from Foundation import NSObject

        parent = self

        # SCStreamOutput と SCStreamDelegate を同一クラスで実装
        # macOS Sonoma/Sequoia の弱参照問題を回避するため
        SCStreamOutput = objc.protocolNamed('SCStreamOutput')
        SCStreamDelegate = objc.protocolNamed('SCStreamDelegate')

        if SCStreamOutput is None:
            raise RuntimeError(
                "SCStreamOutput protocol not found. "
                "Ensure pyobjc-framework-ScreenCaptureKit is properly installed."
            )

        protocols = [SCStreamOutput]
        if SCStreamDelegate is not None:
            protocols.append(SCStreamDelegate)

        class StreamHandler(NSObject, protocols=protocols):
            """
            音声出力ハンドラーとデリゲートを統合
            弱参照問題を回避するため、両方のプロトコルを実装
            """

            def init(self):
                self = objc.super(StreamHandler, self).init()
                if self is None:
                    return None
                self._parent = parent
                return self

            def stream_didOutputSampleBuffer_ofType_(self, stream, sampleBuffer, outputType):
                """音声サンプルを受信 (SCStreamOutput)"""
                # outputType: 0 = screen, 1 = audio
                if outputType != 1:
                    return

                try:
                    # CoreMedia からデータを取得
                    import CoreMedia

                    block_buffer = CoreMedia.CMSampleBufferGetDataBuffer(sampleBuffer)
                    if block_buffer is None:
                        return

                    data_length = CoreMedia.CMBlockBufferGetDataLength(block_buffer)
                    if data_length == 0:
                        return

                    # データをコピー
                    data_pointer, length_out = CoreMedia.CMBlockBufferGetDataPointer(
                        block_buffer, 0, None, None
                    )

                    if data_pointer is not None and length_out > 0:
                        # float32として解釈
                        import ctypes
                        buffer = (ctypes.c_char * length_out).from_address(data_pointer)
                        audio_data = np.frombuffer(bytes(buffer), dtype=np.float32).copy()

                        if len(audio_data) > 0:
                            self._parent.audio_queue.put(audio_data)

                except Exception as e:
                    # CoreMedia関連のエラーは一度だけ表示
                    if "CoreMedia" not in str(e):
                        print(f"Audio processing error: {e}")

            def stream_didStopWithError_(self, stream, error):
                """ストリームが停止 (SCStreamDelegate)"""
                if error:
                    self._parent._error_message = str(error)
                    print(f"Stream stopped with error: {error}")
                self._parent.is_running = False

            def outputVideoEffectDidStartForStream_(self, stream):
                """ビデオエフェクト開始 (無視)"""
                pass

            def outputVideoEffectDidStopForStream_(self, stream):
                """ビデオエフェクト停止 (無視)"""
                pass

        try:
            # 共有可能なコンテンツを取得
            content_ready = threading.Event()
            shareable_content = [None]
            content_error = [None]

            def content_handler(content, error):
                if error:
                    content_error[0] = str(error)
                else:
                    shareable_content[0] = content
                content_ready.set()

            # excludingDesktopWindows: NO, onScreenWindowsOnly: YES
            SCShareableContent.getShareableContentExcludingDesktopWindows_onScreenWindowsOnly_completionHandler_(
                False, True, content_handler
            )

            # 権限確認のため待機
            if not content_ready.wait(timeout=10.0):
                raise RuntimeError(
                    "画面収録の権限確認がタイムアウトしました。\n"
                    "システム環境設定 > プライバシーとセキュリティ > 画面収録 で許可してください。"
                )

            if content_error[0]:
                raise RuntimeError(f"権限エラー: {content_error[0]}")

            content = shareable_content[0]
            if content is None:
                raise RuntimeError(
                    "画面収録の権限がありません。\n"
                    "システム環境設定 > プライバシーとセキュリティ > 画面収録 で許可してください。"
                )

            # ディスプレイを取得
            displays = content.displays()
            if not displays or len(displays) == 0:
                raise RuntimeError("ディスプレイが見つかりません")

            display = displays[0]
            print(f"[ScreenCaptureKit] ディスプレイ検出: {display.displayID()}")

            # ストリーム設定
            config = SCStreamConfiguration.alloc().init()

            # 最小のビデオ設定（音声のみキャプチャしたいが、ビデオも必要）
            config.setWidth_(2)
            config.setHeight_(2)
            config.setShowsCursor_(False)

            # 音声設定
            config.setCapturesAudio_(True)
            config.setExcludesCurrentProcessAudio_(True)
            config.setSampleRate_(float(self.sample_rate))
            config.setChannelCount_(1)

            # フィルター（デスクトップ全体をキャプチャ）
            content_filter = SCContentFilter.alloc().initWithDisplay_excludingWindows_(
                display, []
            )

            if content_filter is None:
                raise RuntimeError("コンテンツフィルターの作成に失敗しました")

            # StreamHandlerを作成（強参照を維持）
            self._stream_output = StreamHandler.alloc().init()
            if self._stream_output is None:
                raise RuntimeError("StreamHandlerの作成に失敗しました")

            # ストリームを作成
            self._stream = SCStream.alloc().initWithFilter_configuration_delegate_(
                content_filter, config, self._stream_output
            )

            if self._stream is None:
                raise RuntimeError(
                    "SCStreamの作成に失敗しました。\n"
                    "画面収録の権限を確認してください: システム設定 > プライバシー > 画面収録"
                )

            print("[ScreenCaptureKit] ストリーム作成完了")

            # ディスパッチキューを作成
            dispatch_queue = None
            try:
                from dispatch import dispatch_queue_create, DISPATCH_QUEUE_SERIAL
                dispatch_queue = dispatch_queue_create(b"audio_capture_queue", DISPATCH_QUEUE_SERIAL)
            except ImportError:
                try:
                    from libdispatch import dispatch_queue_create, DISPATCH_QUEUE_SERIAL
                    dispatch_queue = dispatch_queue_create(b"audio_capture_queue", DISPATCH_QUEUE_SERIAL)
                except ImportError:
                    # ディスパッチキューなしで続行（メインスレッドで処理）
                    print("[ScreenCaptureKit] 警告: libdispatchが利用できません")
                    pass

            # 音声出力を追加
            success, error = self._stream.addStreamOutput_type_sampleHandlerQueue_error_(
                self._stream_output,
                1,  # SCStreamOutputTypeAudio
                dispatch_queue,
                None
            )

            if not success:
                error_msg = str(error) if error else "不明なエラー"
                raise RuntimeError(f"音声出力の追加に失敗しました: {error_msg}")

            print("[ScreenCaptureKit] 音声出力追加完了")

            # キャプチャ開始
            start_ready = threading.Event()
            start_error = [None]

            def start_handler(error):
                if error:
                    start_error[0] = str(error)
                else:
                    parent.is_running = True
                start_ready.set()

            self._stream.startCaptureWithCompletionHandler_(start_handler)

            if not start_ready.wait(timeout=10.0):
                raise RuntimeError("キャプチャ開始がタイムアウトしました")

            if start_error[0]:
                raise RuntimeError(f"キャプチャ開始エラー: {start_error[0]}")

            print("[ScreenCaptureKit] システム音声キャプチャを開始しました")

        except Exception as e:
            self.is_running = False
            self._stream = None
            self._stream_output = None
            raise RuntimeError(f"ScreenCaptureKit の開始に失敗: {e}")

    def stop(self):
        """キャプチャを停止"""
        with self._lock:
            if self._stream:
                stop_ready = threading.Event()

                def stop_handler(error):
                    if error:
                        print(f"Stop error: {error}")
                    stop_ready.set()

                self._stream.stopCaptureWithCompletionHandler_(stop_handler)
                stop_ready.wait(timeout=5.0)
                self._stream = None

            self._stream_output = None
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

    def __exit__(self, *args):
        self.stop()


class SimpleSystemAudioCapture:
    """
    シンプルなシステム音声キャプチャ
    sounddeviceを使用してBlackHoleまたは他の仮想オーディオデバイスからキャプチャ
    """

    def __init__(
        self,
        sample_rate: int = 16000,
        chunk_duration: float = 0.5,
        device_id: Optional[int] = None,
    ):
        import sounddevice as sd

        self.sample_rate = sample_rate
        self.chunk_duration = chunk_duration
        self.device_id = device_id or self._find_virtual_device()
        self.audio_queue: queue.Queue = queue.Queue()
        self.is_running = False
        self._stream = None

    def _find_virtual_device(self) -> Optional[int]:
        """仮想オーディオデバイスを検索"""
        import sounddevice as sd

        devices = sd.query_devices()
        virtual_names = ["blackhole", "loopback", "soundflower", "virtual"]

        for i, device in enumerate(devices):
            name_lower = device["name"].lower()
            if device["max_input_channels"] > 0:
                for vname in virtual_names:
                    if vname in name_lower:
                        return i
        return None

    def _audio_callback(self, indata, frames, time_info, status):
        """音声コールバック"""
        if status:
            print(f"Audio status: {status}")
        audio = indata[:, 0].copy() if indata.shape[1] > 1 else indata.flatten().copy()
        self.audio_queue.put(audio)

    def start(self):
        """キャプチャを開始"""
        import sounddevice as sd

        if self.is_running:
            return

        if self.device_id is None:
            raise RuntimeError(
                "仮想オーディオデバイスが見つかりません。\n"
                "BlackHoleをインストールしてください: brew install blackhole-2ch"
            )

        chunk_size = int(self.sample_rate * self.chunk_duration)
        self._stream = sd.InputStream(
            device=self.device_id,
            samplerate=self.sample_rate,
            channels=1,
            dtype="float32",
            blocksize=chunk_size,
            callback=self._audio_callback,
        )
        self._stream.start()
        self.is_running = True

    def stop(self):
        """キャプチャを停止"""
        if self._stream:
            self._stream.stop()
            self._stream.close()
            self._stream = None
        self.is_running = False

    def get_audio(self, timeout: float = 1.0) -> Optional[np.ndarray]:
        """音声データを取得"""
        try:
            return self.audio_queue.get(timeout=timeout)
        except queue.Empty:
            return None

    def __enter__(self):
        self.start()
        return self

    def __exit__(self, *args):
        self.stop()


def get_system_audio_capture(
    sample_rate: int = 16000,
    chunk_duration: float = 0.5,
    prefer_screencapturekit: bool = True,
    device_id: Optional[int] = None,
):
    """
    システム音声キャプチャを取得

    Args:
        sample_rate: サンプルレート
        chunk_duration: チャンク長（秒）
        prefer_screencapturekit: ScreenCaptureKitを優先するか
        device_id: 仮想デバイスID（BlackHole等）

    Returns:
        適切なキャプチャオブジェクト
    """
    if prefer_screencapturekit and SCREENCAPTUREKIT_AVAILABLE:
        try:
            return ScreenCaptureKitAudioCapture(sample_rate, chunk_duration)
        except Exception as e:
            print(f"ScreenCaptureKit unavailable: {e}")
            print("Falling back to virtual audio device...")

    return SimpleSystemAudioCapture(sample_rate, chunk_duration, device_id)


def is_screencapturekit_available() -> bool:
    """ScreenCaptureKitが利用可能か確認"""
    return SCREENCAPTUREKIT_AVAILABLE


def get_screencapturekit_error() -> Optional[str]:
    """ScreenCaptureKitのエラーメッセージを取得"""
    return SCK_ERROR_MESSAGE
