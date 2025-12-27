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
SCK_CLASSES = {}

try:
    import objc
    from Foundation import NSObject, NSRunLoop, NSDate
    from AppKit import NSApplication

    # ScreenCaptureKitのロード
    try:
        SCK_BUNDLE = objc.loadBundle(
            'ScreenCaptureKit',
            bundle_path='/System/Library/Frameworks/ScreenCaptureKit.framework',
            module_globals=SCK_CLASSES
        )
        SCREENCAPTUREKIT_AVAILABLE = True
    except Exception as e:
        print(f"ScreenCaptureKit load error: {e}")
except ImportError as e:
    print(f"PyObjC import error: {e}")


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
        self._output = None
        self._error_message = None

    def start(self):
        """キャプチャを開始"""
        if self.is_running:
            return

        import objc
        from Foundation import NSObject, NSRunLoop, NSDate

        # libdispatchのインポート（オプション）
        try:
            from dispatch import dispatch_queue_create, DISPATCH_QUEUE_SERIAL
            has_dispatch = True
        except ImportError:
            has_dispatch = False
            # dispatch_get_main_queueを使用するフォールバック
            try:
                from libdispatch import dispatch_get_main_queue
            except ImportError:
                dispatch_get_main_queue = None

        parent = self

        # SCStreamOutputプロトコルを実装したクラス
        SCStreamOutput = objc.protocolNamed('SCStreamOutput')

        class AudioOutputHandler(NSObject, protocols=[SCStreamOutput]):
            """音声出力ハンドラー"""

            def stream_didOutputSampleBuffer_ofType_(self, stream, sampleBuffer, outputType):
                """音声サンプルを受信"""
                # outputType: 0 = screen, 1 = audio
                if outputType != 1:
                    return

                try:
                    import CoreMedia

                    # CMSampleBufferからデータを取得
                    block_buffer = CoreMedia.CMSampleBufferGetDataBuffer(sampleBuffer)
                    if block_buffer is None:
                        return

                    # データ長を取得
                    data_length = CoreMedia.CMBlockBufferGetDataLength(block_buffer)
                    if data_length == 0:
                        return

                    # データをコピー
                    data = CoreMedia.CMBlockBufferCopyDataBytes(
                        block_buffer, 0, data_length, None
                    )

                    if data and len(data) > 0:
                        # float32として解釈 (ScreenCaptureKitはFloat32を出力)
                        audio_data = np.frombuffer(data, dtype=np.float32).copy()
                        if len(audio_data) > 0:
                            parent.audio_queue.put(audio_data)

                except Exception as e:
                    if "CoreMedia" not in str(e):
                        print(f"Audio processing error: {e}")

        class StreamDelegate(NSObject):
            """SCStreamデリゲート"""

            def stream_didStopWithError_(self, stream, error):
                if error:
                    parent._error_message = str(error)
                    print(f"Stream stopped with error: {error}")
                parent.is_running = False

        try:
            # クラスを取得
            SCShareableContent = SCK_CLASSES.get('SCShareableContent')
            SCStreamConfiguration = SCK_CLASSES.get('SCStreamConfiguration')
            SCContentFilter = SCK_CLASSES.get('SCContentFilter')
            SCStream = SCK_CLASSES.get('SCStream')

            if not all([SCShareableContent, SCStreamConfiguration, SCContentFilter, SCStream]):
                raise RuntimeError("ScreenCaptureKit classes not found")

            # 共有可能なコンテンツを取得
            content_ready = threading.Event()
            shareable_content = [None]
            content_error = [None]

            def content_handler(content, error):
                if error:
                    content_error[0] = error
                else:
                    shareable_content[0] = content
                content_ready.set()

            SCShareableContent.getShareableContentWithCompletionHandler_(content_handler)

            # 権限確認のため待機
            if not content_ready.wait(timeout=5.0):
                raise RuntimeError(
                    "画面収録の権限がありません。\n"
                    "システム環境設定 > プライバシーとセキュリティ > 画面収録 で許可してください。"
                )

            if content_error[0]:
                raise RuntimeError(f"権限エラー: {content_error[0]}")

            content = shareable_content[0]
            if content is None:
                raise RuntimeError("画面収録の権限がありません")

            # ディスプレイを取得
            displays = content.displays()
            if not displays or len(displays) == 0:
                raise RuntimeError("ディスプレイが見つかりません")

            display = displays[0]

            # ストリーム設定
            config = SCStreamConfiguration.alloc().init()
            config.setWidth_(1)  # 最小サイズ（音声のみ）
            config.setHeight_(1)
            config.setCapturesAudio_(True)
            config.setExcludesCurrentProcessAudio_(True)
            config.setSampleRate_(float(self.sample_rate))
            config.setChannelCount_(1)

            # フィルター
            filter = SCContentFilter.alloc().initWithDisplay_excludingWindows_(display, [])

            # デリゲートを先に作成
            delegate = StreamDelegate.alloc().init()

            # ストリームを作成
            self._stream = SCStream.alloc().initWithFilter_configuration_delegate_(
                filter, config, delegate
            )

            # 音声出力ハンドラーを追加
            self._output = AudioOutputHandler.alloc().init()

            # ディスパッチキューを作成
            if has_dispatch:
                audio_queue = dispatch_queue_create(b"audio_queue", DISPATCH_QUEUE_SERIAL)
            else:
                # libdispatchが利用できない場合はNoneを使用
                # (macOSのデフォルトキューが使われる)
                audio_queue = None

            # 出力を追加
            error_ptr = objc.nil
            success = self._stream.addStreamOutput_type_sampleHandlerQueue_error_(
                self._output,
                1,  # SCStreamOutputTypeAudio
                audio_queue,
                error_ptr
            )

            if not success:
                raise RuntimeError("音声出力の追加に失敗しました")

            # キャプチャ開始
            start_ready = threading.Event()
            start_error = [None]

            def start_handler(error):
                if error:
                    start_error[0] = error
                else:
                    parent.is_running = True
                start_ready.set()

            self._stream.startCaptureWithCompletionHandler_(start_handler)

            if not start_ready.wait(timeout=5.0):
                raise RuntimeError("キャプチャ開始がタイムアウトしました")

            if start_error[0]:
                raise RuntimeError(f"キャプチャ開始エラー: {start_error[0]}")

            print("[ScreenCaptureKit] システム音声キャプチャを開始しました")

        except Exception as e:
            self.is_running = False
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
