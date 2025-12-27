"""
話者分離（Speaker Diarization）モジュール
オプション機能 - pyannote.audio が必要
"""

import threading
import time
from collections import defaultdict
from dataclasses import dataclass
from typing import Optional

import numpy as np


@dataclass
class SpeakerSegment:
    """話者セグメント"""
    speaker_id: str
    start_time: float
    end_time: float
    text: str = ""
    embedding: Optional[np.ndarray] = None


class SimpleSpeakerTracker:
    """
    シンプルな話者追跡
    音声特徴（ピッチ、エネルギー）に基づく話者変更検出
    """

    def __init__(
        self,
        min_silence_duration: float = 1.0,
        energy_threshold: float = 0.01,
        max_speakers: int = 4,
    ):
        self.min_silence_duration = min_silence_duration
        self.energy_threshold = energy_threshold
        self.max_speakers = max_speakers

        self.current_speaker = 0
        self._last_speech_time = 0
        self._silence_start = 0
        self._in_silence = False

        # 話者プロファイル（平均ピッチとエネルギー）
        self._speaker_profiles: list[dict] = []
        self._current_features: list[dict] = []  # 現在の発話の特徴
        self._feature_window = 10  # 特徴を平均化するウィンドウサイズ

    def _extract_features(self, audio: np.ndarray, sample_rate: int = 16000) -> dict:
        """音声から特徴を抽出"""
        # RMSエネルギー
        energy = np.sqrt(np.mean(audio ** 2))

        # 簡易ピッチ推定（ゼロ交差率）
        zero_crossings = np.sum(np.abs(np.diff(np.sign(audio)))) / 2
        zcr = zero_crossings / len(audio) * sample_rate

        return {"energy": energy, "zcr": zcr}

    def _match_speaker(self, features: dict) -> int:
        """特徴から最も近い話者を見つける"""
        if not self._speaker_profiles:
            return 0

        # 各話者との距離を計算
        min_distance = float("inf")
        best_speaker = 0

        for i, profile in enumerate(self._speaker_profiles):
            # 特徴の正規化距離
            energy_diff = abs(features["energy"] - profile["energy"]) / max(profile["energy"], 0.001)
            zcr_diff = abs(features["zcr"] - profile["zcr"]) / max(profile["zcr"], 1)
            distance = energy_diff + zcr_diff

            if distance < min_distance:
                min_distance = distance
                best_speaker = i

        # 閾値を超えたら新しい話者の可能性
        if min_distance > 1.5 and len(self._speaker_profiles) < self.max_speakers:
            return len(self._speaker_profiles)  # 新しい話者

        return best_speaker

    def _update_profile(self, speaker_id: int, features: dict):
        """話者プロファイルを更新"""
        while len(self._speaker_profiles) <= speaker_id:
            self._speaker_profiles.append({"energy": 0, "zcr": 0, "count": 0})

        profile = self._speaker_profiles[speaker_id]
        count = profile["count"]

        # 移動平均で更新
        alpha = 0.3 if count < 5 else 0.1
        profile["energy"] = profile["energy"] * (1 - alpha) + features["energy"] * alpha
        profile["zcr"] = profile["zcr"] * (1 - alpha) + features["zcr"] * alpha
        profile["count"] = count + 1

    def process_audio(self, audio: np.ndarray, sample_rate: int = 16000) -> int:
        """
        音声を処理して話者IDを返す

        Args:
            audio: 音声データ
            sample_rate: サンプルレート

        Returns:
            話者ID（0から始まる整数）
        """
        features = self._extract_features(audio, sample_rate)
        current_time = time.time()

        if features["energy"] < self.energy_threshold:
            # 無音検出
            if not self._in_silence:
                self._silence_start = current_time
                self._in_silence = True

                # 無音になったら、現在の特徴をプロファイルに反映
                if self._current_features:
                    avg_features = {
                        "energy": np.mean([f["energy"] for f in self._current_features]),
                        "zcr": np.mean([f["zcr"] for f in self._current_features]),
                    }
                    self._update_profile(self.current_speaker, avg_features)
                    self._current_features = []
        else:
            # 発話検出
            if self._in_silence:
                silence_duration = current_time - self._silence_start

                if silence_duration > self.min_silence_duration:
                    # 長い無音の後 = 話者変更の可能性を検討
                    matched_speaker = self._match_speaker(features)
                    self.current_speaker = matched_speaker
                    self._current_features = []

                self._in_silence = False

            # 現在の特徴を蓄積
            self._current_features.append(features)
            if len(self._current_features) > self._feature_window:
                self._current_features.pop(0)

            self._last_speech_time = current_time

        return self.current_speaker

    def get_speaker_label(self, speaker_id: int) -> str:
        """話者ラベルを取得"""
        return f"話者{speaker_id + 1}"


class PyAnnoteSpeakerDiarizer:
    """
    pyannote.audio を使用した高精度話者分離

    使用するには追加インストールが必要:
        uv pip install pyannote-audio torch

    また、Hugging Face のトークンが必要:
        https://huggingface.co/pyannote/speaker-diarization-3.1
    """

    def __init__(self, hf_token: Optional[str] = None, device: str = "mps"):
        """
        Args:
            hf_token: Hugging Face アクセストークン
            device: 処理デバイス (mps, cuda, cpu)
        """
        self.hf_token = hf_token
        self.device = device
        self._pipeline = None
        self._embedder = None
        self._speaker_embeddings: dict[str, np.ndarray] = {}
        self._lock = threading.Lock()

    def _load_pipeline(self):
        """パイプラインを遅延ロード"""
        if self._pipeline is not None:
            return

        try:
            from pyannote.audio import Pipeline

            self._pipeline = Pipeline.from_pretrained(
                "pyannote/speaker-diarization-3.1",
                use_auth_token=self.hf_token,
            )
            self._pipeline.to(self.device)
        except ImportError:
            raise ImportError(
                "pyannote.audio がインストールされていません。\n"
                "インストール: uv pip install pyannote-audio torch"
            )
        except Exception as e:
            raise RuntimeError(f"パイプラインの読み込みに失敗: {e}")

    def _load_embedder(self):
        """話者埋め込みモデルを遅延ロード"""
        if self._embedder is not None:
            return

        try:
            from pyannote.audio import Model
            from pyannote.audio.pipelines.speaker_verification import (
                PretrainedSpeakerEmbedding,
            )

            self._embedder = PretrainedSpeakerEmbedding(
                "speechbrain/spkrec-ecapa-voxceleb",
                device=self.device,
            )
        except ImportError:
            pass  # 埋め込みは任意

    def diarize_file(self, audio_path: str) -> list[SpeakerSegment]:
        """
        音声ファイルを話者分離

        Args:
            audio_path: 音声ファイルパス

        Returns:
            話者セグメントのリスト
        """
        self._load_pipeline()

        diarization = self._pipeline(audio_path)

        segments = []
        for turn, _, speaker in diarization.itertracks(yield_label=True):
            segments.append(SpeakerSegment(
                speaker_id=speaker,
                start_time=turn.start,
                end_time=turn.end,
            ))

        return segments

    def get_speaker_embedding(self, audio: np.ndarray, sample_rate: int = 16000) -> Optional[np.ndarray]:
        """音声から話者埋め込みを取得"""
        self._load_embedder()
        if self._embedder is None:
            return None

        try:
            import torch

            # 正規化
            audio = audio.astype(np.float32)
            if audio.max() > 1.0:
                audio = audio / 32768.0

            # テンソルに変換
            waveform = torch.from_numpy(audio).unsqueeze(0)

            embedding = self._embedder(waveform)
            return embedding.cpu().numpy()
        except Exception:
            return None

    def identify_speaker(
        self,
        audio: np.ndarray,
        sample_rate: int = 16000,
        threshold: float = 0.7,
    ) -> str:
        """
        音声から話者を識別

        Args:
            audio: 音声データ
            sample_rate: サンプルレート
            threshold: 類似度閾値

        Returns:
            話者ID
        """
        embedding = self.get_speaker_embedding(audio, sample_rate)
        if embedding is None:
            return "unknown"

        with self._lock:
            # 既存の話者との類似度を計算
            best_match = None
            best_score = 0

            for speaker_id, ref_embedding in self._speaker_embeddings.items():
                # コサイン類似度
                score = np.dot(embedding.flatten(), ref_embedding.flatten()) / (
                    np.linalg.norm(embedding) * np.linalg.norm(ref_embedding)
                )
                if score > best_score:
                    best_score = score
                    best_match = speaker_id

            if best_score >= threshold and best_match:
                return best_match

            # 新しい話者として登録
            new_id = f"speaker_{len(self._speaker_embeddings) + 1}"
            self._speaker_embeddings[new_id] = embedding.flatten()
            return new_id


class DiarizationManager:
    """話者分離の統合マネージャー"""

    def __init__(self, use_pyannote: bool = False, hf_token: Optional[str] = None):
        """
        Args:
            use_pyannote: pyannote.audio を使用するか
            hf_token: Hugging Face トークン（pyannote使用時）
        """
        self.use_pyannote = use_pyannote

        if use_pyannote:
            try:
                self._diarizer = PyAnnoteSpeakerDiarizer(hf_token=hf_token)
            except ImportError:
                print("Warning: pyannote.audio が利用できないため、簡易話者追跡を使用します")
                self.use_pyannote = False
                self._diarizer = SimpleSpeakerTracker()
        else:
            self._diarizer = SimpleSpeakerTracker()

        self._current_speaker = "Speaker 1"
        self._segments: list[SpeakerSegment] = []

    def process_audio(self, audio: np.ndarray, sample_rate: int = 16000) -> str:
        """
        音声を処理して話者を識別

        Args:
            audio: 音声データ
            sample_rate: サンプルレート

        Returns:
            話者ラベル
        """
        if self.use_pyannote and isinstance(self._diarizer, PyAnnoteSpeakerDiarizer):
            speaker_id = self._diarizer.identify_speaker(audio, sample_rate)
            self._current_speaker = speaker_id
        elif isinstance(self._diarizer, SimpleSpeakerTracker):
            speaker_id = self._diarizer.process_audio(audio, sample_rate)
            self._current_speaker = self._diarizer.get_speaker_label(speaker_id)

        return self._current_speaker

    def add_segment(self, text: str, start_time: float, end_time: float):
        """セグメントを追加"""
        self._segments.append(SpeakerSegment(
            speaker_id=self._current_speaker,
            start_time=start_time,
            end_time=end_time,
            text=text,
        ))

    def get_segments(self) -> list[SpeakerSegment]:
        """全セグメントを取得"""
        return self._segments

    def get_formatted_transcript(self) -> str:
        """フォーマット済みトランスクリプトを取得"""
        if not self._segments:
            return ""

        lines = []
        current_speaker = None

        for segment in self._segments:
            if segment.speaker_id != current_speaker:
                current_speaker = segment.speaker_id
                lines.append(f"\n[{current_speaker}]")
            lines.append(segment.text)

        return " ".join(lines)
