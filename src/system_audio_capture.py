"""
システム音声キャプチャモジュール (macOS)
ScreenCaptureKit を使用してBlackHoleなしでシステム音声をキャプチャ

macOS 13.0+ が必要
"""

import queue
import threading
import time
from typing import Callable, Optional

import numpy as np

# ScreenCaptureKit用のフラグ
SCREENCAPTUREKIT_AVAILABLE = False

try:
    import objc
    from Foundation import NSObject, NSRunLoop, NSDate
    from AppKit import NSApplication

    # ScreenCaptureKitのロード
    try:
        objc.loadBundle(
            'ScreenCaptureKit',
            bundle_path='/System/Library/Frameworks/ScreenCaptureKit.framework',
            module_globals=globals()
        )
        SCREENCAPTUREKIT_AVAILABLE = True
    except Exception:
        pass
except ImportError:
    pass


class ScreenCaptureKitAudioCapture:
    """
    ScreenCaptureKit を使用したシステム音声キャプチャ

    macOS 13.0 (Ventura) 以降で利用可能
    BlackHole不要でシステム音声をキャプチャできる
    """

    def __init__(self, sample_rate: int = 16000, chunk_duration: float = 0.5):
        if not SCREENCAPTUREKIT_AVAILABLE:
            raise RuntimeError(
                "ScreenCaptureKit が利用できません。\n"
                "必要条件:\n"
                "  - macOS 13.0 (Ventura) 以降\n"
                "  - pip install pyobjc-framework-ScreenCaptureKit pyobjc-framework-Cocoa"
            )

        self.sample_rate = sample_rate
        self.chunk_duration = chunk_duration
        self.audio_queue: queue.Queue = queue.Queue()
        self.is_running = False
        self._stream = None
        self._delegate = None

    def _create_audio_handler(self):
        """オーディオハンドラーを作成"""
        parent = self

        class AudioDelegate(NSObject):
            """SCStreamのデリゲート"""

            def stream_didOutputSampleBuffer_ofType_(self, stream, sampleBuffer, outputType):
                """音声サンプルを受信"""
                try:
                    # CMSampleBufferから音声データを抽出
                    import CoreMedia
                    import AudioToolbox

                    # オーディオバッファリストを取得
                    block_buffer = CoreMedia.CMSampleBufferGetDataBuffer(sampleBuffer)
                    if block_buffer is None:
                        return

                    # データを取得
                    length, data_pointer = CoreMedia.CMBlockBufferGetDataPointer(
                        block_buffer, 0, None, None
                    )

                    if data_pointer and length > 0:
                        # float32として解釈
                        audio_data = np.frombuffer(
                            data_pointer[:length],
                            dtype=np.float32
                        ).copy()

                        # リサンプリングが必要な場合
                        if len(audio_data) > 0:
                            parent.audio_queue.put(audio_data)

                except Exception as e:
                    print(f"Audio processing error: {e}")

            def stream_didStopWithError_(self, stream, error):
                """ストリーム停止時"""
                if error:
                    print(f"Stream stopped with error: {error}")
                parent.is_running = False

        return AudioDelegate.alloc().init()

    async def _request_permission(self):
        """画面録画権限をリクエスト"""
        # SCShareableContentで権限確認
        try:
            content = await SCShareableContent.getShareableContentWithCompletionHandler_(None)
            return content is not None
        except Exception:
            return False

    def start(self):
        """キャプチャを開始"""
        if self.is_running:
            return

        try:
            # 共有可能なコンテンツを取得（画面キャプチャ権限が必要）
            def completion_handler(content, error):
                if error:
                    print(f"Error getting shareable content: {error}")
                    return

                if content is None:
                    print("No shareable content available")
                    return

                # ディスプレイを取得
                displays = content.displays()
                if not displays or len(displays) == 0:
                    print("No displays found")
                    return

                display = displays[0]

                # ストリーム設定
                config = SCStreamConfiguration.alloc().init()
                config.setWidth_(1)  # 最小サイズ（音声のみ）
                config.setHeight_(1)
                config.setCapturesAudio_(True)
                config.setExcludesCurrentProcessAudio_(True)  # 自分のアプリの音声は除外
                config.setSampleRate_(self.sample_rate)
                config.setChannelCount_(1)

                # フィルター（ディスプレイ全体をキャプチャ）
                filter = SCContentFilter.alloc().initWithDisplay_excludingWindows_(
                    display, []
                )

                # ストリームを作成
                self._stream = SCStream.alloc().initWithFilter_configuration_delegate_(
                    filter, config, self._delegate
                )

                # 音声出力を追加
                self._delegate = self._create_audio_handler()

                # ストリームを開始
                def start_handler(error):
                    if error:
                        print(f"Failed to start stream: {error}")
                    else:
                        self.is_running = True

                self._stream.startCaptureWithCompletionHandler_(start_handler)

            SCShareableContent.getShareableContentWithCompletionHandler_(completion_handler)

            # RunLoopを少し回して完了を待つ
            run_loop = NSRunLoop.currentRunLoop()
            timeout = NSDate.dateWithTimeIntervalSinceNow_(2.0)
            while not self.is_running and NSDate.date().compare_(timeout) == -1:
                run_loop.runMode_beforeDate_("NSDefaultRunLoopMode", NSDate.dateWithTimeIntervalSinceNow_(0.1))

        except Exception as e:
            raise RuntimeError(f"ScreenCaptureKit の開始に失敗: {e}")

    def stop(self):
        """キャプチャを停止"""
        if self._stream:
            self._stream.stopCaptureWithCompletionHandler_(lambda error: None)
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

    return SimpleSystemAudioCapture(sample_rate, chunk_duration, device_id)


def is_screencapturekit_available() -> bool:
    """ScreenCaptureKitが利用可能か確認"""
    return SCREENCAPTUREKIT_AVAILABLE
