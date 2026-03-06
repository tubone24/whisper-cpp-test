"""
辞書機能モジュール
同音異義語の置換を文脈に応じて行う
"""

import json
import re
from dataclasses import dataclass, field
from pathlib import Path
from typing import Optional


@dataclass
class ReplacementRule:
    """単純置換ルール"""
    pattern: str  # 置換元（正規表現可）
    replacement: str  # 置換先
    is_regex: bool = False  # 正規表現として扱うか


@dataclass
class ContextRule:
    """文脈に応じた置換ルール"""
    pattern: str  # 置換元
    replacement: str  # 置換先
    context_keywords: list[str] = field(default_factory=list)  # 文脈キーワード
    negative_keywords: list[str] = field(default_factory=list)  # 除外キーワード
    window_size: int = 50  # 文脈を見る文字数（前後）


@dataclass
class DictionaryConfig:
    """辞書設定"""
    # 単純置換ルール
    replacements: list[ReplacementRule] = field(default_factory=list)
    # 文脈に応じた置換ルール
    context_rules: list[ContextRule] = field(default_factory=list)
    # ハルシネーションフィルター（削除するパターン）
    hallucination_filters: list[str] = field(default_factory=list)


class Dictionary:
    """辞書による文字置換を行うクラス"""

    def __init__(self, config: Optional[DictionaryConfig] = None):
        self.config = config or DictionaryConfig()
        self._compiled_patterns: dict[str, re.Pattern] = {}

    def _compile_pattern(self, pattern: str, is_regex: bool = False) -> re.Pattern:
        """パターンをコンパイル（キャッシュ付き）"""
        cache_key = f"{pattern}:{is_regex}"
        if cache_key not in self._compiled_patterns:
            if is_regex:
                self._compiled_patterns[cache_key] = re.compile(pattern)
            else:
                # リテラル文字列としてエスケープ
                self._compiled_patterns[cache_key] = re.compile(re.escape(pattern))
        return self._compiled_patterns[cache_key]

    def _has_context(self, text: str, pos: int, rule: ContextRule) -> bool:
        """指定位置の周辺に文脈キーワードがあるかチェック"""
        start = max(0, pos - rule.window_size)
        end = min(len(text), pos + len(rule.pattern) + rule.window_size)
        context = text[start:end]

        # 除外キーワードがあれば置換しない
        for neg_keyword in rule.negative_keywords:
            if neg_keyword in context:
                return False

        # 文脈キーワードがあれば置換する
        if not rule.context_keywords:
            return True  # キーワードが空なら常に置換

        for keyword in rule.context_keywords:
            if keyword in context:
                return True
        return False

    def apply(self, text: str) -> str:
        """テキストに辞書を適用"""
        if not text:
            return text

        result = text

        # 0. ハルシネーションフィルターを適用（最初に処理）
        for pattern in self.config.hallucination_filters:
            compiled = self._compile_pattern(pattern, is_regex=True)
            result = compiled.sub("", result)

        # 1. 文脈に応じた置換を適用（先に処理）
        for rule in self.config.context_rules:
            pattern = self._compile_pattern(rule.pattern)
            # 置換位置を後ろから処理（前から処理すると位置がずれる）
            matches = list(pattern.finditer(result))
            for match in reversed(matches):
                if self._has_context(result, match.start(), rule):
                    result = result[:match.start()] + rule.replacement + result[match.end():]

        # 2. 単純置換を適用
        for rule in self.config.replacements:
            pattern = self._compile_pattern(rule.pattern, rule.is_regex)
            result = pattern.sub(rule.replacement, result)

        return result.strip()

    @classmethod
    def from_json(cls, json_path: Path) -> "Dictionary":
        """JSONファイルから辞書を読み込み"""
        with open(json_path, "r", encoding="utf-8") as f:
            data = json.load(f)
        return cls.from_dict(data)

    @classmethod
    def from_dict(cls, data: dict) -> "Dictionary":
        """辞書データから作成"""
        config = DictionaryConfig()

        # 単純置換ルール
        for item in data.get("replacements", []):
            if isinstance(item, dict):
                rule = ReplacementRule(
                    pattern=item["pattern"],
                    replacement=item["replacement"],
                    is_regex=item.get("is_regex", False),
                )
            else:
                # 旧形式: {"元": "先"} のような形式
                continue
            config.replacements.append(rule)

        # シンプルなキー:値形式もサポート
        simple_replacements = data.get("simple", {})
        for pattern, replacement in simple_replacements.items():
            config.replacements.append(ReplacementRule(
                pattern=pattern,
                replacement=replacement,
            ))

        # 文脈に応じた置換ルール
        for item in data.get("context_rules", []):
            rule = ContextRule(
                pattern=item["pattern"],
                replacement=item["replacement"],
                context_keywords=item.get("context_keywords", []),
                negative_keywords=item.get("negative_keywords", []),
                window_size=item.get("window_size", 50),
            )
            config.context_rules.append(rule)

        # ハルシネーションフィルター
        config.hallucination_filters = data.get("hallucination_filters", [])

        return cls(config)

    def to_dict(self) -> dict:
        """辞書データに変換"""
        data = {
            "replacements": [],
            "context_rules": [],
            "hallucination_filters": self.config.hallucination_filters,
        }

        for rule in self.config.replacements:
            data["replacements"].append({
                "pattern": rule.pattern,
                "replacement": rule.replacement,
                "is_regex": rule.is_regex,
            })

        for rule in self.config.context_rules:
            data["context_rules"].append({
                "pattern": rule.pattern,
                "replacement": rule.replacement,
                "context_keywords": rule.context_keywords,
                "negative_keywords": rule.negative_keywords,
                "window_size": rule.window_size,
            })

        return data

    def save_json(self, json_path: Path):
        """JSONファイルに保存"""
        with open(json_path, "w", encoding="utf-8") as f:
            json.dump(self.to_dict(), f, ensure_ascii=False, indent=2)


def create_example_dictionary() -> dict:
    """サンプル辞書を作成"""
    return {
        "simple": {
            # 単純な置換（常に置換される）
            "ウィスパー": "Whisper",
            "クロード": "Claude",
        },
        "replacements": [
            # より詳細な単純置換ルール
            {
                "pattern": "エーアイ",
                "replacement": "AI",
                "is_regex": False,
            },
        ],
        "context_rules": [
            # 文脈に応じた置換
            {
                "pattern": "家具",
                "replacement": "KAG",
                "context_keywords": ["会社", "開発", "プロジェクト", "チーム", "株式会社", "サービス"],
                "negative_keywords": ["インテリア", "家具屋", "ソファ"],
                "window_size": 100,
            },
            {
                "pattern": "科学",
                "replacement": "化学",
                "context_keywords": ["実験", "物質", "反応", "分子", "原子"],
                "window_size": 50,
            },
        ],
        # ハルシネーションフィルター（Whisperが無音時に出力する定型文を削除）
        "hallucination_filters": [
            r"ご視聴ありがとうございま(す|した)。?",
            r"チャンネル登録.*お願いします。?",
            r"(ご|)チャンネル登録.*してね。?",
            r"字幕[:：].*",
            r"Thanks? for watching\.?",
            r"Please subscribe.*",
            r"See you (next time|in the next|later).*",
            r"Bye[\s\-]?bye\.?",
            r"お疲れ様でした。?",
            r"ではまた。?",
            r"^[\s　。、\.]+$",  # 空白や句読点のみ
        ],
    }


# デフォルト辞書パス
def get_default_dictionary_path() -> Path:
    """デフォルトの辞書ファイルパスを取得"""
    # ユーザー設定ディレクトリ
    config_dir = Path.home() / ".config" / "whisper-realtime"
    config_dir.mkdir(parents=True, exist_ok=True)
    return config_dir / "dictionary.json"


def load_or_create_dictionary(path: Optional[Path] = None) -> Dictionary:
    """辞書を読み込み、なければデフォルトを作成"""
    if path is None:
        path = get_default_dictionary_path()

    if path.exists():
        return Dictionary.from_json(path)
    else:
        # デフォルト辞書を作成して保存
        example_data = create_example_dictionary()
        dictionary = Dictionary.from_dict(example_data)
        dictionary.save_json(path)
        return dictionary
