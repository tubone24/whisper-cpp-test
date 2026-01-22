"""
文章校正モジュール
軽量なPython実装（部分結果用）とtextlint統合（最終結果用）を提供
"""

import json
import re
import subprocess
import tempfile
from dataclasses import dataclass, field
from pathlib import Path
from typing import Optional


@dataclass
class ProofreaderConfig:
    """校正設定"""
    # 軽量校正（部分結果用、高速）
    enable_light_proofreading: bool = True
    # textlint校正（最終結果用、重いため非推奨）
    enable_textlint: bool = False  # デフォルトOFF（重い処理）
    textlint_config_path: Optional[Path] = None
    # 校正ルール設定
    check_double_negative: bool = True  # 二重否定
    check_double_particle: bool = True  # 二重助詞
    check_sentence_length: bool = True  # 一文の長さ
    max_sentence_length: int = 100  # 最大文字数
    check_weak_expressions: bool = True  # 弱い表現
    check_redundant_expressions: bool = True  # 冗長表現
    check_max_ten: bool = True  # 読点の連続（「、」の数）
    max_ten_count: int = 3  # 一文あたりの最大読点数


@dataclass
class ProofreadResult:
    """校正結果"""
    original_text: str
    corrected_text: str
    corrections: list[dict] = field(default_factory=list)
    is_textlint_applied: bool = False


class LightProofreader:
    """
    軽量校正クラス（Python実装）
    ストリーミング中の部分結果に適用する高速な校正
    """

    def __init__(self, config: Optional[ProofreaderConfig] = None):
        self.config = config or ProofreaderConfig()
        self._patterns = self._compile_patterns()

    def _compile_patterns(self) -> dict[str, list[tuple[re.Pattern, str, str]]]:
        """校正パターンをコンパイル"""
        patterns = {
            "redundant": [],  # 冗長表現
            "weak": [],  # 弱い表現
            "double_particle": [],  # 二重助詞
            "typo": [],  # よくある誤変換
        }

        # 冗長表現
        redundant_rules = [
            (r"することができ(る|ます)", "でき\\1", "「することができる」→「できる」"),
            (r"することが可能", "可能", "「することが可能」→「可能」"),
            (r"行うことができ", "でき", "「行うことができ」→「でき」"),
            (r"という(こと|もの)が", "が", "「ということが」→「が」"),
            (r"において(は)?", "で\\1", "「において」→「で」"),
            (r"に関して(は)?", "について\\1", "「に関して」→「について」"),
            (r"についての", "の", "「についての」→「の」"),
            (r"という風に", "ように", "「という風に」→「ように」"),
            (r"のほう(が|を|に)", "\\1", "「のほうが」→「が」"),
            (r"~たりする$", "~る", "文末の「〜たりする」を簡潔に"),
        ]
        for pattern, replacement, desc in redundant_rules:
            patterns["redundant"].append(
                (re.compile(pattern), replacement, desc)
            )

        # 弱い表現（注意喚起のみ、自動修正しない）
        weak_rules = [
            (r"かもしれ(ない|ません)", None, "断定を避ける表現"),
            (r"と思(う|います|われ)", None, "主観的な表現"),
            (r"ような気がし(ます|た)", None, "曖昧な表現"),
        ]
        for pattern, replacement, desc in weak_rules:
            patterns["weak"].append(
                (re.compile(pattern), replacement, desc)
            )

        # 二重助詞
        double_particle_rules = [
            (r"([がをにはでと])(\1)", "\\1", "助詞の重複"),
            (r"の(の)", "の", "「の」の重複"),
        ]
        for pattern, replacement, desc in double_particle_rules:
            patterns["double_particle"].append(
                (re.compile(pattern), replacement, desc)
            )

        # よくある誤変換（音声認識で起こりやすいもの）
        typo_rules = [
            (r"意外と", "以外と", "「意外」と「以外」の混同注意"),  # 文脈依存なので注意
            (r"行なう", "行う", "「行なう」→「行う」（常用漢字表）"),
            (r"おこなう", "行う", "「おこなう」→「行う」"),
            (r"いただく事", "いただくこと", "「事」→「こと」（補助動詞）"),
            (r"出来る", "できる", "「出来る」→「できる」"),
            (r"無い", "ない", "「無い」→「ない」"),
            (r"有る", "ある", "「有る」→「ある」"),
            (r"良い(?!い)", "よい", "「良い」→「よい」（形容詞）"),
            (r"(?<![ァ-ン])様な", "ような", "「様な」→「ような」"),
            (r"等([。、]|$)", "など\\1", "「等」→「など」"),
            (r"及び", "および", "「及び」→「および」"),
            (r"又は", "または", "「又は」→「または」"),
            (r"但し", "ただし", "「但し」→「ただし」"),
            (r"従って", "したがって", "「従って」→「したがって」"),
            (r"更に", "さらに", "「更に」→「さらに」"),
            (r"殆ど", "ほとんど", "「殆ど」→「ほとんど」"),
            # 追加：textlint相当の表記統一ルール
            (r"全て", "すべて", "「全て」→「すべて」"),
            (r"予め", "あらかじめ", "「予め」→「あらかじめ」"),
            (r"既に", "すでに", "「既に」→「すでに」"),
            (r"概ね", "おおむね", "「概ね」→「おおむね」"),
            (r"直ぐ", "すぐ", "「直ぐ」→「すぐ」"),
            (r"暫く", "しばらく", "「暫く」→「しばらく」"),
            (r"是非", "ぜひ", "「是非」→「ぜひ」"),
            (r"所謂", "いわゆる", "「所謂」→「いわゆる」"),
            (r"尚", "なお", "「尚」→「なお」"),
            (r"即ち", "すなわち", "「即ち」→「すなわち」"),
            (r"何故", "なぜ", "「何故」→「なぜ」"),
            (r"様々", "さまざま", "「様々」→「さまざま」"),
            (r"所(?=[をにでがは])", "ところ", "「所」→「ところ」"),
            (r"時(?=[をにでがは])", "とき", "「時」→「とき」"),
            (r"事(?=[をにでがは])", "こと", "「事」→「こと」"),
            (r"物(?=[をにでがは])", "もの", "「物」→「もの」"),
            (r"為(?=[をにでがは])", "ため", "「為」→「ため」"),
            # カタカナ語の表記統一
            (r"メイン", "メイン", ""),  # そのまま
            (r"([ァ-ン])ー([ァ-ン])", "\\1ー\\2", "長音符の統一"),
        ]
        for pattern, replacement, desc in typo_rules:
            if replacement and desc:  # 空の説明はスキップ
                patterns["typo"].append(
                    (re.compile(pattern), replacement, desc)
                )

        return patterns

    def proofread(self, text: str) -> ProofreadResult:
        """テキストを校正"""
        if not text:
            return ProofreadResult(original_text=text, corrected_text=text)

        corrections = []
        result = text

        # 冗長表現の修正
        if self.config.check_redundant_expressions:
            for pattern, replacement, desc in self._patterns["redundant"]:
                if replacement:
                    new_result = pattern.sub(replacement, result)
                    if new_result != result:
                        corrections.append({
                            "type": "redundant",
                            "description": desc,
                            "original": result,
                            "corrected": new_result,
                        })
                        result = new_result

        # 二重助詞の修正
        if self.config.check_double_particle:
            for pattern, replacement, desc in self._patterns["double_particle"]:
                if replacement:
                    new_result = pattern.sub(replacement, result)
                    if new_result != result:
                        corrections.append({
                            "type": "double_particle",
                            "description": desc,
                        })
                        result = new_result

        # よくある誤変換の修正
        for pattern, replacement, desc in self._patterns["typo"]:
            if replacement:
                new_result = pattern.sub(replacement, result)
                if new_result != result:
                    corrections.append({
                        "type": "typo",
                        "description": desc,
                    })
                    result = new_result

        return ProofreadResult(
            original_text=text,
            corrected_text=result,
            corrections=corrections,
            is_textlint_applied=False,
        )


class TextlintProofreader:
    """
    textlint統合クラス
    最終結果に対してNode.js製textlintを呼び出して本格的な校正を行う
    """

    def __init__(self, config: Optional[ProofreaderConfig] = None):
        self.config = config or ProofreaderConfig()
        self._textlint_available: Optional[bool] = None
        self._config_path = self._get_or_create_config()

    def _get_or_create_config(self) -> Optional[Path]:
        """textlint設定ファイルを取得または作成"""
        if self.config.textlint_config_path and self.config.textlint_config_path.exists():
            return self.config.textlint_config_path

        # デフォルト設定ディレクトリ
        config_dir = Path.home() / ".config" / "whisper-realtime"
        config_dir.mkdir(parents=True, exist_ok=True)
        config_path = config_dir / ".textlintrc.json"

        if not config_path.exists():
            # デフォルト設定を作成
            default_config = {
                "rules": {
                    "preset-ja-technical-writing": {
                        "sentence-length": {
                            "max": 100
                        },
                        "max-ten": {
                            "max": 3
                        },
                        "no-doubled-joshi": {
                            "strict": False
                        }
                    }
                }
            }
            with open(config_path, "w", encoding="utf-8") as f:
                json.dump(default_config, f, ensure_ascii=False, indent=2)

        return config_path

    def is_available(self) -> bool:
        """textlintが利用可能かチェック"""
        if self._textlint_available is not None:
            return self._textlint_available

        try:
            result = subprocess.run(
                ["npx", "textlint", "--version"],
                capture_output=True,
                timeout=10,
            )
            self._textlint_available = result.returncode == 0
        except (subprocess.TimeoutExpired, FileNotFoundError):
            self._textlint_available = False

        return self._textlint_available

    def proofread(self, text: str) -> ProofreadResult:
        """textlintでテキストを校正"""
        if not text:
            return ProofreadResult(original_text=text, corrected_text=text)

        if not self.is_available():
            return ProofreadResult(
                original_text=text,
                corrected_text=text,
                corrections=[{"type": "error", "description": "textlint not available"}],
                is_textlint_applied=False,
            )

        try:
            # 一時ファイルに書き出し
            with tempfile.NamedTemporaryFile(
                mode="w",
                suffix=".txt",
                encoding="utf-8",
                delete=False,
            ) as f:
                f.write(text)
                temp_path = f.name

            # textlint --fix を実行
            cmd = ["npx", "textlint", "--fix", temp_path]
            if self._config_path and self._config_path.exists():
                cmd.extend(["--config", str(self._config_path)])

            result = subprocess.run(
                cmd,
                capture_output=True,
                text=True,
                timeout=30,
            )

            # 修正後のテキストを読み込み
            with open(temp_path, "r", encoding="utf-8") as f:
                corrected_text = f.read()

            # 一時ファイルを削除
            Path(temp_path).unlink()

            corrections = []
            if corrected_text != text:
                corrections.append({
                    "type": "textlint",
                    "description": "textlintによる自動修正",
                })

            return ProofreadResult(
                original_text=text,
                corrected_text=corrected_text,
                corrections=corrections,
                is_textlint_applied=True,
            )

        except subprocess.TimeoutExpired:
            return ProofreadResult(
                original_text=text,
                corrected_text=text,
                corrections=[{"type": "error", "description": "textlint timeout"}],
                is_textlint_applied=False,
            )
        except Exception as e:
            return ProofreadResult(
                original_text=text,
                corrected_text=text,
                corrections=[{"type": "error", "description": str(e)}],
                is_textlint_applied=False,
            )


class Proofreader:
    """
    統合校正クラス
    部分結果には軽量校正、最終結果にはtextlintを適用
    """

    def __init__(self, config: Optional[ProofreaderConfig] = None):
        self.config = config or ProofreaderConfig()
        self._light = LightProofreader(config) if config.enable_light_proofreading else None
        self._textlint = TextlintProofreader(config) if config.enable_textlint else None

    def proofread_partial(self, text: str) -> str:
        """
        部分結果を校正（軽量・高速）
        ストリーミング中のリアルタイム表示用
        """
        if not text or not self._light:
            return text
        result = self._light.proofread(text)
        return result.corrected_text

    def proofread_final(self, text: str) -> ProofreadResult:
        """
        最終結果を校正（textlint使用・精度重視）
        録音終了時の最終出力用
        """
        if not text:
            return ProofreadResult(original_text=text, corrected_text=text)

        # まず軽量校正を適用
        result = text
        corrections = []

        if self._light:
            light_result = self._light.proofread(result)
            result = light_result.corrected_text
            corrections.extend(light_result.corrections)

        # textlintで追加校正
        if self._textlint and self._textlint.is_available():
            textlint_result = self._textlint.proofread(result)
            result = textlint_result.corrected_text
            corrections.extend(textlint_result.corrections)

            return ProofreadResult(
                original_text=text,
                corrected_text=result,
                corrections=corrections,
                is_textlint_applied=True,
            )

        return ProofreadResult(
            original_text=text,
            corrected_text=result,
            corrections=corrections,
            is_textlint_applied=False,
        )

    def is_textlint_available(self) -> bool:
        """textlintが利用可能か確認"""
        if self._textlint:
            return self._textlint.is_available()
        return False


def setup_textlint() -> bool:
    """
    textlintと日本語ルールセットをインストール
    Returns: インストール成功したらTrue
    """
    try:
        # Node.jsが利用可能か確認
        result = subprocess.run(
            ["node", "--version"],
            capture_output=True,
            timeout=10,
        )
        if result.returncode != 0:
            print("Error: Node.js is not installed")
            return False

        # textlintと日本語プリセットをインストール
        packages = [
            "textlint",
            "textlint-rule-preset-ja-technical-writing",
            "textlint-rule-preset-japanese",
        ]

        print("Installing textlint and Japanese rules...")
        result = subprocess.run(
            ["npm", "install", "-g"] + packages,
            capture_output=True,
            text=True,
            timeout=120,
        )

        if result.returncode == 0:
            print("textlint installed successfully!")
            return True
        else:
            print(f"Error installing textlint: {result.stderr}")
            return False

    except subprocess.TimeoutExpired:
        print("Error: Installation timed out")
        return False
    except FileNotFoundError:
        print("Error: npm not found. Please install Node.js first.")
        return False
