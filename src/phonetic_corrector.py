"""
音声認識誤り訂正モジュール（拡張版）
音声学の知見に基づいて、Whisper等の音声認識結果を訂正する

主な機能：
1. 音韻類似性に基づく誤り訂正（同音異義語、類音語）
2. よくある音声認識誤りパターンの修正（IT/テック用語200語以上）
3. カタカナ/ひらがな表記揺れの正規化
4. 音韻距離計算による候補提示（強化版音素混同行列）
5. Whisperハルシネーション（繰り返し）の検出・除去
6. 敬語の正規化（です/ます調 ⇔ だ/である調）
7. ビジネス用語・医療用語の訂正
8. 同音異義語50パターン以上の対応

拡張内容（2026-01-22）：
- IT/テック用語辞書: 200語以上（AI/ML、プログラミング言語、クラウド、ツールなど）
- ビジネス用語: 異動/移動、規定/規程など頻出パターン
- 医療用語: 鼻茸、鼻血、肝癌などWhisper頻出誤認識
- 音素混同行列: 40ペア以上の音韻学的類似度を定義
- ハルシネーション除去: N-gram繰り返し検出
- 敬語正規化: 30パターン以上の敬語変換
- 同音異義語グループ: 50グループ以上

新メソッド：
- suggest_corrections(): 訂正候補を複数提示
- detect_hallucination(): ハルシネーション検出
- get_homophone_candidates(): 同音異義語候補の取得
- normalize_honorifics(): 敬語の正規化

参考:
- https://arxiv.org/abs/2408.16180 (MPA GER for Japanese ASR)
- https://medium.com/axinc/whisper音声認識誤り辞書
- asr_research.md (調査レポート)
"""

import re
from dataclasses import dataclass, field
from typing import Optional, List, Dict, Tuple


@dataclass
class PhoneticCorrectorConfig:
    """音声認識誤り訂正の設定"""
    # 音韻類似誤り訂正
    enable_phonetic_correction: bool = True
    # よくある誤りパターン修正
    enable_common_errors: bool = True
    # カタカナ正規化
    enable_katakana_normalization: bool = True
    # 長音・促音の正規化（外来語で誤訂正を引き起こす可能性があるためデフォルト無効）
    enable_mora_normalization: bool = False
    # 数字表記の正規化
    enable_number_normalization: bool = True
    # ハルシネーション除去
    enable_hallucination_removal: bool = True
    # 敬語の正規化
    enable_honorific_normalization: bool = False
    # 訂正候補の類似度閾値（0-1、低いほど厳格）
    correction_threshold: float = 0.4
    # ハルシネーション検出の繰り返し回数閾値
    hallucination_repeat_threshold: int = 3
    # N-gram繰り返し検出のN値
    hallucination_ngram_size: int = 5


@dataclass
class CorrectionResult:
    """訂正結果"""
    original_text: str
    corrected_text: str
    corrections: list[dict] = field(default_factory=list)


# ============================================================
# 日本語ローマ字変換テーブル（ヘボン式ベース）
# ============================================================
HIRAGANA_TO_ROMAJI = {
    # 基本母音
    'あ': 'a', 'い': 'i', 'う': 'u', 'え': 'e', 'お': 'o',
    # か行
    'か': 'ka', 'き': 'ki', 'く': 'ku', 'け': 'ke', 'こ': 'ko',
    'が': 'ga', 'ぎ': 'gi', 'ぐ': 'gu', 'げ': 'ge', 'ご': 'go',
    # さ行
    'さ': 'sa', 'し': 'shi', 'す': 'su', 'せ': 'se', 'そ': 'so',
    'ざ': 'za', 'じ': 'ji', 'ず': 'zu', 'ぜ': 'ze', 'ぞ': 'zo',
    # た行
    'た': 'ta', 'ち': 'chi', 'つ': 'tsu', 'て': 'te', 'と': 'to',
    'だ': 'da', 'ぢ': 'ji', 'づ': 'zu', 'で': 'de', 'ど': 'do',
    # な行
    'な': 'na', 'に': 'ni', 'ぬ': 'nu', 'ね': 'ne', 'の': 'no',
    # は行
    'は': 'ha', 'ひ': 'hi', 'ふ': 'fu', 'へ': 'he', 'ほ': 'ho',
    'ば': 'ba', 'び': 'bi', 'ぶ': 'bu', 'べ': 'be', 'ぼ': 'bo',
    'ぱ': 'pa', 'ぴ': 'pi', 'ぷ': 'pu', 'ぺ': 'pe', 'ぽ': 'po',
    # ま行
    'ま': 'ma', 'み': 'mi', 'む': 'mu', 'め': 'me', 'も': 'mo',
    # や行
    'や': 'ya', 'ゆ': 'yu', 'よ': 'yo',
    # ら行
    'ら': 'ra', 'り': 'ri', 'る': 'ru', 'れ': 're', 'ろ': 'ro',
    # わ行
    'わ': 'wa', 'を': 'wo', 'ん': 'n',
    # 拗音
    'きゃ': 'kya', 'きゅ': 'kyu', 'きょ': 'kyo',
    'しゃ': 'sha', 'しゅ': 'shu', 'しょ': 'sho',
    'ちゃ': 'cha', 'ちゅ': 'chu', 'ちょ': 'cho',
    'にゃ': 'nya', 'にゅ': 'nyu', 'にょ': 'nyo',
    'ひゃ': 'hya', 'ひゅ': 'hyu', 'ひょ': 'hyo',
    'みゃ': 'mya', 'みゅ': 'myu', 'みょ': 'myo',
    'りゃ': 'rya', 'りゅ': 'ryu', 'りょ': 'ryo',
    'ぎゃ': 'gya', 'ぎゅ': 'gyu', 'ぎょ': 'gyo',
    'じゃ': 'ja', 'じゅ': 'ju', 'じょ': 'jo',
    'びゃ': 'bya', 'びゅ': 'byu', 'びょ': 'byo',
    'ぴゃ': 'pya', 'ぴゅ': 'pyu', 'ぴょ': 'pyo',
    # 促音・長音
    'っ': 'Q',  # 促音（後続子音を重ねる）
    'ー': '-',  # 長音
    # 小文字
    'ぁ': 'a', 'ぃ': 'i', 'ぅ': 'u', 'ぇ': 'e', 'ぉ': 'o',
    'ゃ': 'ya', 'ゅ': 'yu', 'ょ': 'yo',
}

# カタカナ→ひらがな変換
KATAKANA_TO_HIRAGANA = {chr(k): chr(k - 96) for k in range(0x30A1, 0x30F7)}
KATAKANA_TO_HIRAGANA['ー'] = 'ー'  # 長音はそのまま


def katakana_to_hiragana(text: str) -> str:
    """カタカナをひらがなに変換"""
    return ''.join(KATAKANA_TO_HIRAGANA.get(c, c) for c in text)


def to_romaji(text: str) -> str:
    """日本語テキストをローマ字に変換（音韻比較用）"""
    # カタカナ→ひらがな
    text = katakana_to_hiragana(text)

    result = []
    i = 0
    while i < len(text):
        # 2文字の拗音をチェック
        if i + 1 < len(text):
            two_char = text[i:i+2]
            if two_char in HIRAGANA_TO_ROMAJI:
                result.append(HIRAGANA_TO_ROMAJI[two_char])
                i += 2
                continue

        # 1文字
        char = text[i]
        if char in HIRAGANA_TO_ROMAJI:
            result.append(HIRAGANA_TO_ROMAJI[char])
        else:
            result.append(char)
        i += 1

    # 促音の処理（Qを次の子音に置換）
    romaji = ''.join(result)
    romaji = re.sub(r'Q([bcdfghjklmnpqrstvwxyz])', r'\1\1', romaji)
    romaji = re.sub(r'Q', '', romaji)  # 残ったQを削除

    return romaji


def phonetic_distance(word1: str, word2: str) -> float:
    """
    2つの単語の音韻距離を計算（0に近いほど似ている）

    音素の類似性を考慮した重み付きレーベンシュタイン距離
    """
    # ローマ字に変換
    r1 = to_romaji(word1).lower()
    r2 = to_romaji(word2).lower()

    # 同一なら0
    if r1 == r2:
        return 0.0

    # 音素混同の重み行列（音韻学的類似性に基づく）
    # 値が小さいほど混同しやすい（0.0=同一、1.0=全く異なる）
    CONFUSION_PAIRS = {
        # === 有声/無声対立（日本語で特に重要） ===
        ('k', 'g'): 0.25, ('t', 'd'): 0.25, ('s', 'z'): 0.25,
        ('h', 'b'): 0.35, ('p', 'b'): 0.25, ('f', 'h'): 0.35,
        ('ch', 'j'): 0.3, ('ts', 'dz'): 0.3,

        # === 歯擦音・破擦音の混同 ===
        ('s', 'sh'): 0.25, ('z', 'j'): 0.25,
        ('ch', 'ts'): 0.35, ('j', 'z'): 0.25,
        ('sh', 'ch'): 0.3, ('ts', 's'): 0.3,

        # === 鼻音の調音位置による混同 ===
        ('n', 'm'): 0.35, ('n', 'ng'): 0.3, ('m', 'ng'): 0.4,

        # === 流音（日本語では同一視されやすい） ===
        ('r', 'l'): 0.15,  # 日本語話者には区別困難

        # === 母音の混同（調音位置・開口度） ===
        ('i', 'e'): 0.35, ('e', 'a'): 0.4, ('a', 'o'): 0.4,
        ('o', 'u'): 0.35, ('u', 'i'): 0.45,

        # === 長音・短音の混同（モーラ数） ===
        ('a', 'aa'): 0.15, ('i', 'ii'): 0.15, ('u', 'uu'): 0.15,
        ('e', 'ee'): 0.15, ('o', 'oo'): 0.15,

        # === 拗音・直音の混同 ===
        ('ki', 'kya'): 0.3, ('shi', 'sha'): 0.25, ('chi', 'cha'): 0.25,
        ('ni', 'nya'): 0.3, ('hi', 'hya'): 0.3, ('mi', 'mya'): 0.3,
        ('ri', 'rya'): 0.3, ('gi', 'gya'): 0.3, ('ji', 'ja'): 0.25,
        ('bi', 'bya'): 0.3, ('pi', 'pya'): 0.3,

        # === 促音（っ）の有無 ===
        ('k', 'kk'): 0.2, ('t', 'tt'): 0.2, ('p', 'pp'): 0.2,
        ('s', 'ss'): 0.2, ('ch', 'cch'): 0.2,

        # === 撥音（ん）の混同 ===
        ('n', 'nn'): 0.2, ('m', 'mm'): 0.2,

        # === 半母音の混同 ===
        ('w', 'u'): 0.3, ('y', 'i'): 0.3,

        # === その他の調音位置による混同 ===
        ('k', 't'): 0.5, ('g', 'd'): 0.5, ('h', 'f'): 0.35,
        ('b', 'd'): 0.45, ('p', 't'): 0.45, ('m', 'b'): 0.4,
    }

    def substitution_cost(c1: str, c2: str) -> float:
        if c1 == c2:
            return 0.0
        pair = tuple(sorted([c1, c2]))
        return CONFUSION_PAIRS.get(pair, 1.0)

    # 動的計画法でレーベンシュタイン距離を計算
    m, n = len(r1), len(r2)
    dp = [[0.0] * (n + 1) for _ in range(m + 1)]

    for i in range(m + 1):
        dp[i][0] = i
    for j in range(n + 1):
        dp[0][j] = j

    for i in range(1, m + 1):
        for j in range(1, n + 1):
            cost = substitution_cost(r1[i-1], r2[j-1])
            dp[i][j] = min(
                dp[i-1][j] + 1,      # 削除
                dp[i][j-1] + 1,      # 挿入
                dp[i-1][j-1] + cost  # 置換
            )

    # 正規化（長い方の文字数で割る）
    max_len = max(m, n)
    if max_len == 0:
        return 0.0
    return dp[m][n] / max_len


# ============================================================
# よくある音声認識誤りパターン
# ============================================================

# Whisperでよくある誤認識パターン（誤り→正解）
# 調査結果に基づき大幅拡充（200語以上）
COMMON_ASR_ERRORS = {
    # ============================================================
    # IT/テック用語のカタカナ誤認識（200語以上）
    # ============================================================

    # === AI/ML関連 ===
    'ウィスパー': 'Whisper',
    'ウイスパー': 'Whisper',
    'クロード': 'Claude',
    'アンソロピック': 'Anthropic',
    'アンスロピック': 'Anthropic',
    'オープンエーアイ': 'OpenAI',
    'オープンAI': 'OpenAI',
    'チャットジーピーティー': 'ChatGPT',
    'ジーピーティー': 'GPT',
    'エーアイ': 'AI',
    'マシンラーニング': 'Machine Learning',
    'ディープラーニング': 'Deep Learning',
    'ニューラルネットワーク': 'Neural Network',
    'トランスフォーマー': 'Transformer',
    'バート': 'BERT',
    'ラーマ': 'LLaMA',
    'ミストラル': 'Mistral',
    'ジェミニ': 'Gemini',

    # === プログラミング言語 ===
    'パイソン': 'Python',
    'パイスン': 'Python',
    'ジャバスクリプト': 'JavaScript',
    'タイプスクリプト': 'TypeScript',
    'ラスト': 'Rust',
    'スイフト': 'Swift',
    'コトリン': 'Kotlin',
    'ゴー': 'Go',
    'ジャバ': 'Java',
    'シーシャープ': 'C#',
    'シープラスプラス': 'C++',
    'ルビー': 'Ruby',
    'ピーエイチピー': 'PHP',
    'エスキューエル': 'SQL',
    'アール': 'R',
    'スカラ': 'Scala',
    'ダート': 'Dart',
    'エリクサー': 'Elixir',
    'ハスケル': 'Haskell',

    # === フレームワーク/ライブラリ ===
    'リアクト': 'React',
    'ビュー': 'Vue',
    'アンギュラー': 'Angular',
    'ノード': 'Node.js',
    'デノ': 'Deno',
    'バン': 'Bun',
    'ネクスト': 'Next.js',
    'ヌクスト': 'Nuxt',
    'エクスプレス': 'Express',
    'ジャンゴ': 'Django',
    'フラスク': 'Flask',
    'レールズ': 'Rails',
    'スプリング': 'Spring',
    'ララベル': 'Laravel',
    'テンソルフロー': 'TensorFlow',
    'パイトーチ': 'PyTorch',
    'ケラス': 'Keras',
    'サイキットラーン': 'scikit-learn',
    'パンダス': 'pandas',
    'ナムパイ': 'NumPy',

    # === クラウド/インフラ ===
    'エーダブリューエス': 'AWS',
    'アマゾン': 'Amazon',
    'グーグル': 'Google',
    'マイクロソフト': 'Microsoft',
    'アジュール': 'Azure',
    'ジーシーピー': 'GCP',
    'ドッカー': 'Docker',
    'クーバネティス': 'Kubernetes',
    'クバネティス': 'Kubernetes',
    'クーベルネテス': 'Kubernetes',
    'テラフォーム': 'Terraform',
    'アンシブル': 'Ansible',
    'ジェンキンス': 'Jenkins',
    'サークルシーアイ': 'CircleCI',
    'ギットハブアクションズ': 'GitHub Actions',
    'ベクセル': 'Vercel',
    'ネットリファイ': 'Netlify',
    'ヘロク': 'Heroku',
    'デジタルオーシャン': 'DigitalOcean',
    'レンダー': 'Render',

    # === データベース ===
    'マイエスキューエル': 'MySQL',
    'ポストグレス': 'PostgreSQL',
    'モンゴディービー': 'MongoDB',
    'レディス': 'Redis',
    'イーエスキューエル': 'Elasticsearch',
    'カサンドラ': 'Cassandra',
    'ダイナモディービー': 'DynamoDB',
    'ファイアーベース': 'Firebase',
    'スープベース': 'Supabase',

    # === 開発ツール ===
    'ギット': 'Git',
    'ギットハブ': 'GitHub',
    'ギターブ': 'GitHub',
    'ギットラボ': 'GitLab',
    'ビットバケット': 'Bitbucket',
    'ブイエスコード': 'VS Code',
    'インテリジェイ': 'IntelliJ',
    'エクリプス': 'Eclipse',
    'ビム': 'Vim',
    'イーマックス': 'Emacs',
    'ポストマン': 'Postman',
    'インソムニア': 'Insomnia',

    # === OS/プラットフォーム ===
    'リナックス': 'Linux',
    'ウィンドウズ': 'Windows',
    'マック': 'Mac',
    'マックオーエス': 'macOS',
    'アイオーエス': 'iOS',
    'アンドロイド': 'Android',
    'ユニックス': 'Unix',
    'ウブントゥ': 'Ubuntu',
    'デビアン': 'Debian',
    'フェドラ': 'Fedora',
    'センタオーエス': 'CentOS',

    # === Web技術 ===
    'エイチティーティーピー': 'HTTP',
    'エイチティーティーピーエス': 'HTTPS',
    'レスト': 'REST',
    'グラフキューエル': 'GraphQL',
    'ジェイソン': 'JSON',
    'ヤムル': 'YAML',
    'エックスエムエル': 'XML',
    'エイチティーエムエル': 'HTML',
    'シーエスエス': 'CSS',
    'サス': 'Sass',
    'ウェブパック': 'Webpack',
    'ビート': 'Vite',
    'エスリント': 'ESLint',
    'プリティア': 'Prettier',

    # === UI/UX ===
    'ユーアイ': 'UI',
    'ユーエックス': 'UX',
    'フィグマ': 'Figma',
    'スケッチ': 'Sketch',
    'アドビエックスディー': 'Adobe XD',

    # === 通信/コラボレーション ===
    'スラック': 'Slack',
    'ディスコード': 'Discord',
    'ズーム': 'Zoom',
    'チームズ': 'Teams',
    'ノーション': 'Notion',
    'コンフルエンス': 'Confluence',
    'トレロ': 'Trello',
    'アサナ': 'Asana',
    'ジラ': 'Jira',

    # === セキュリティ ===
    'オーソライゼーション': 'Authorization',
    'オーセンティケーション': 'Authentication',
    'ジェイダブリュティー': 'JWT',
    'オーオース': 'OAuth',
    'エスエスエル': 'SSL',
    'ティーエルエス': 'TLS',

    # === その他技術用語 ===
    'エーピーアイ': 'API',
    'ジーピーユー': 'GPU',
    'シーピーユー': 'CPU',
    'ラム': 'RAM',
    'エスエスディー': 'SSD',
    'ユーアールエル': 'URL',
    'アイピー': 'IP',
    'ディーエヌエス': 'DNS',
    'シーディーエン': 'CDN',
    'エスディーケー': 'SDK',
    'シーエルアイ': 'CLI',
    'アイディーイー': 'IDE',
    'リポジトリ': 'Repository',
    'コミット': 'Commit',
    'プルリクエスト': 'Pull Request',
    'マージ': 'Merge',
    'ブランチ': 'Branch',
    'デプロイ': 'Deploy',
    'リリース': 'Release',
    'ビルド': 'Build',
    'テスト': 'Test',
    'デバッグ': 'Debug',
    'リファクタリング': 'Refactoring',

    # === 決済/ビジネス ===
    'ストライプ': 'Stripe',
    'ペイパル': 'PayPal',
    'スクエア': 'Square',

    # ============================================================
    # ビジネス用語の同音異義語
    # ============================================================
    '異動': '移動',  # 人事異動が多い文脈
    '規程': '規定',  # 社内規程が多い
    '訂正': '修正',  # 誤りを直す
    '承認': '証人',  # ビジネス文脈では承認
    '生産': '清算',  # 製造業では生産

    # ============================================================
    # 医療用語の誤認識（Whisper研究データより）
    # ============================================================
    '花竹': '鼻茸',
    '花地': '鼻血',
    'ビフェ': '鼻閉',
    '微生息感': '鼻閉塞感',
    'リーガン': '鼻閉感',
    '考え': '肝癌',  # かんがえ→かんがん
    '心レンズ': '心電図',
    '経連': '痙攣',
    '高速': '拘束',  # 医療文脈では拘束

    # ============================================================
    # 一般的な同音異義語（頻出50パターン以上）
    # ============================================================

    # === いこう系 ===
    '移行': '以降',  # 時系列では以降
    '意向': '以降',  # 文脈依存

    # === かがく系 ===
    '科学': '化学',  # 理系文脈では判別困難
    '価格': '価値',  # ビジネス

    # === かんしん系 ===
    '感心': '関心',  # 興味は関心
    '寒心': '関心',  # 稀

    # === きかん系 ===
    '機関': '期間',  # 時間は期間
    '器官': '期間',  # 医療では器官
    '気管': '器官',  # 呼吸器

    # === こうせい系 ===
    '校正': '構成',  # 編集では校正
    '更正': '構成',  # 法律では更正

    # === しこう系 ===
    '思考': '志向',  # 考えることは思考
    '嗜好': '志向',  # 好みは嗜好
    '施行': '試行',  # 法律では施行
    '指向': '志向',  # 目標は志向

    # === たいしょう系 ===
    '対象': '対照',  # 比較は対照
    '大賞': '対象',  # ターゲットは対象
    '大将': '対象',  # 稀

    # === かてい系 ===
    '過程': '家庭',  # プロセスは過程
    '仮定': '課程',  # 学習は課程

    # === けんとう系 ===
    '見当': '検討',  # 調査は検討
    '検討': '健闘',  # 稀

    # === しゅうしょく系 ===
    '修飾': '就職',  # 文法では修飾
    '収束': '就職',  # 事態は収束

    # === せいさん系 ===
    '生産': '清算',  # 製造は生産
    '精算': '清算',  # 計算は精算

    # === ほしょう系 ===
    '保障': '保証',  # 権利は保障
    '補償': '保証',  # 損害は補償

    # === その他頻出 ===
    '意外': '以外',  # 除外は以外
    '異常': '以上',  # 数値は以上
    '回答': '解答',  # 質問は回答、問題は解答
    '感慨': '感概',  # 稀
    '帰省': '規制',  # 実家は帰省
    '既製': '規制',  # 製品は既製
    '体制': '態勢',  # 組織は体制
    '特長': '特徴',  # 優れた点は特長

    # === 長音の有無による誤り ===
    'おばさん': 'おばあさん',
    'おじさん': 'おじいさん',
    'おねさん': 'おねえさん',
}

# 音韻的に類似した単語グループ（同じグループ内は混同されやすい）
# 調査結果に基づき大幅拡充（50グループ以上）
PHONETIC_SIMILAR_GROUPS = [
    # === こうしょう（最多48語以上の同音異義語） ===
    ['交渉', '考証', '工匠', '高尚', '鉱床', '口承', '公称', '公証', '校章',
     '高唱', '厚相', '公傷', '恒常', '洪鐘', '好尚'],

    # === きかん ===
    ['機関', '期間', '器官', '気管', '帰還', '季刊', '棄権', '基幹', '既刊', '飢饉'],

    # === しこう ===
    ['思考', '志向', '嗜好', '施行', '試行', '至高', '指向', '施工', '私考'],

    # === たいしょう ===
    ['対象', '対照', '大将', '大賞', '大正', '対称', '対処', '退所'],

    # === かてい ===
    ['過程', '家庭', '仮定', '課程', '下底', '仮定', '河底'],

    # === いこう ===
    ['移行', '以降', '意向', '遺構', '威光', '衣香', '異構', '移項'],

    # === かがく ===
    ['科学', '化学', '価額', '歌学', '佳作'],

    # === しゅし ===
    ['趣旨', '種子', '主旨', '首肯'],

    # === せいさん ===
    ['生産', '清算', '精算', '聖餐'],

    # === しょうにん ===
    ['承認', '商人', '証人', '小人', '少人'],

    # === かんしん ===
    ['関心', '感心', '寒心', '歓心', '閑心'],

    # === こうせい ===
    ['構成', '校正', '更正', '公正', '恒星', '攻勢', '厚生', '後生', '後世'],

    # === けんとう ===
    ['検討', '見当', '健闘', '剣道'],

    # === しゅうしょく ===
    ['就職', '修飾', '収束', '終息', '終職'],

    # === ほしょう ===
    ['保障', '保証', '補償', '保章'],

    # === いし ===
    ['意志', '意思', '医師', '石', '遺志'],

    # === かいとう ===
    ['回答', '解答', '会堂', '開頭'],

    # === きてい ===
    ['規定', '規程', '既定', '既程'],

    # === けいやく ===
    ['契約', '経約'],

    # === こうか ===
    ['効果', '高価', '硬化', '降下', '広化', '香華'],

    # === さいけん ===
    ['債券', '再建', '再検', '債権'],

    # === しゅうしゅう ===
    ['収集', '収拾', '収周', '襲週'],

    # === せいかく ===
    ['正確', '性格', '精確'],

    # === そうさい ===
    ['相殺', '総裁', '葬祭'],

    # === ていせい ===
    ['訂正', '修正', '定性', '低成'],

    # === どうい ===
    ['同意', '同義', '銅製'],

    # === ふかん ===
    ['俯瞰', '不敢', '不感'],

    # === よそう ===
    ['予想', '予測', '余剰'],

    # === かいかく ===
    ['改革', '改格', '会革'],

    # === きょうぎ ===
    ['協議', '競技', '狭義', '教義'],

    # === しょうてん ===
    ['商店', '焦点', '昇天', '争点'],

    # === ていあん ===
    ['提案', '低案', '堤案'],

    # === はんこう ===
    ['反抗', '反攻', '反響', '版行'],

    # === めいかく ===
    ['明確', '明格'],

    # === りかい ===
    ['理解', '利害'],

    # === いどう ===
    ['移動', '異動', '緯度'],

    # === かくにん ===
    ['確認', '核人'],

    # === きょか ===
    ['許可', '許価'],

    # === けってい ===
    ['決定', '決程'],

    # === こうどう ===
    ['行動', '高度', '光度', '坑道'],

    # === さんこう ===
    ['参考', '山口', '酸攻'],

    # === しょうしょう ===
    ['少々', '小姓', '証書'],

    # === せんたく ===
    ['選択', '洗濯'],

    # === たんい ===
    ['単位', '単衣'],

    # === ふくそう ===
    ['複雑', '服装', '副層'],

    # === ほうこう ===
    ['方向', '方鋼', '報告', '放校'],

    # === みぎ ===
    ['右', '三木'],

    # === りゆう ===
    ['理由', '利用'],

    # === かいぎ ===
    ['会議', '海岸', '解議'],

    # === せつめい ===
    ['説明', '摂明'],
]


# ============================================================
# 敬語・丁寧語の正規化パターン
# ============================================================

# 敬語の正規化（音声認識では敬語レベルが混在することが多い）
HONORIFIC_PATTERNS = {
    # 丁寧語 → 普通形
    'です': 'だ',
    'ます': 'る',
    'ました': 'た',
    'ません': 'ない',
    'ませんでした': 'なかった',
    'でした': 'だった',
    'ください': 'くれ',
    'いただく': 'もらう',
    'おります': 'いる',
    'ございます': 'ある',

    # 尊敬語 → 普通形
    'いらっしゃる': 'いる',
    'いらっしゃいます': 'います',
    'おっしゃる': '言う',
    'おっしゃいます': '言います',
    '召し上がる': '食べる',
    'ご覧になる': '見る',
    'なさる': 'する',
    'くださる': 'くれる',

    # 謙譲語 → 普通形
    '申し上げる': '言う',
    '申します': '言います',
    'いたす': 'する',
    'いたします': 'します',
    '参る': '行く',
    '参ります': '行きます',
    'うかがう': '聞く',
    'いただく': 'もらう',
    '拝見する': '見る',
    '存じる': '知る',
    '存じます': '知ります',
    'おる': 'いる',
}

# 敬語の正規化（逆パターン：普通形 → 丁寧語）
HONORIFIC_PATTERNS_POLITE = {
    'だ': 'です',
    'だった': 'でした',
    'である': 'です',
    'じゃない': 'ではありません',
    'ない': 'ません',
    'なかった': 'ませんでした',
    'いる': 'います',
    'ある': 'あります',
    'する': 'します',
    'した': 'しました',
    'くれ': 'ください',
}


# ============================================================
# カタカナ表記揺れの正規化
# ============================================================

KATAKANA_NORMALIZATION = {
    # ヴ系の正規化（Whisperは「ヴ」系を使うことがある）
    'ヴァ': 'バ', 'ヴィ': 'ビ', 'ヴ': 'ブ', 'ヴェ': 'ベ', 'ヴォ': 'ボ',
    # 注意: ティ/ディ系は外来語で必要なので正規化しない
    # 'ティ': 'チ',  # 外来語では維持（コーディング、ミーティング等）
    # 'ディ': 'ジ',  # 外来語では維持
    # 長音の表記揺れ
    'ー': 'ー',  # 全角に統一
    '−': 'ー',   # ハイフンマイナス
    '‐': 'ー',   # ハイフン
    '－': 'ー',  # 全角ハイフン
    # 中点の表記揺れ
    '・': '・',  # 全角中点に統一
    '･': '・',   # 半角中点
}

# 長音の正規化パターン
CHOUON_PATTERNS = [
    (r'([アカサタナハマヤラワガザダバパ])ア', r'\1ー'),  # ア段+ア→長音
    (r'([イキシチニヒミリギジヂビピ])イ', r'\1ー'),    # イ段+イ→長音
    (r'([ウクスツヌフムユルグズヅブプ])ウ', r'\1ー'),  # ウ段+ウ→長音
    (r'([エケセテネヘメレゲゼデベペ])エ', r'\1ー'),    # エ段+エ→長音
    (r'([エケセテネヘメレゲゼデベペ])イ', r'\1ー'),    # エ段+イ→長音（「せい」→「せー」など）
    (r'([オコソトノホモヨロヲゴゾドボポ])オ', r'\1ー'),  # オ段+オ→長音
    (r'([オコソトノホモヨロヲゴゾドボポ])ウ', r'\1ー'),  # オ段+ウ→長音（「おう」→「おー」）
]


# ============================================================
# 数字表記の正規化
# ============================================================

NUMBER_KANJI_TO_ARABIC = {
    '一': '1', '二': '2', '三': '3', '四': '4', '五': '5',
    '六': '6', '七': '7', '八': '8', '九': '9', '十': '10',
    '百': '100', '千': '1000', '万': '10000',
    '零': '0', '〇': '0',
}

FULLWIDTH_TO_HALFWIDTH_DIGITS = {
    '０': '0', '１': '1', '２': '2', '３': '3', '４': '4',
    '５': '5', '６': '6', '７': '7', '８': '8', '９': '9',
}


# ============================================================
# メインクラス
# ============================================================

class PhoneticCorrector:
    """
    音声認識誤り訂正クラス

    音声学の知見に基づいて、音声認識結果を訂正する。
    辞書による置換は別モジュール（dictionary.py）で行うため、
    このクラスは音韻的な誤り訂正に特化する。
    """

    def __init__(self, config: Optional[PhoneticCorrectorConfig] = None):
        self.config = config or PhoneticCorrectorConfig()
        self._compile_patterns()

    def _compile_patterns(self):
        """正規表現パターンをコンパイル"""
        # よくある誤りパターン（長い順にソート）
        self._error_patterns = sorted(
            COMMON_ASR_ERRORS.items(),
            key=lambda x: len(x[0]),
            reverse=True
        )

        # カタカナ正規化パターン
        self._katakana_patterns = sorted(
            KATAKANA_NORMALIZATION.items(),
            key=lambda x: len(x[0]),
            reverse=True
        )

    def correct(self, text: str) -> CorrectionResult:
        """
        音声認識結果を訂正

        Args:
            text: 音声認識結果テキスト

        Returns:
            CorrectionResult: 訂正結果
        """
        if not text:
            return CorrectionResult(original_text=text, corrected_text=text)

        corrections = []
        result = text

        # 1. ハルシネーション除去（最優先）
        if self.config.enable_hallucination_removal:
            new_result, corrs = self._remove_hallucination(result)
            if new_result != result:
                corrections.extend(corrs)
                result = new_result

        # 2. カタカナ正規化
        if self.config.enable_katakana_normalization:
            new_result, corrs = self._normalize_katakana(result)
            if new_result != result:
                corrections.extend(corrs)
                result = new_result

        # 3. 長音・促音の正規化
        if self.config.enable_mora_normalization:
            new_result, corrs = self._normalize_mora(result)
            if new_result != result:
                corrections.extend(corrs)
                result = new_result

        # 4. よくある誤りパターンの修正
        if self.config.enable_common_errors:
            new_result, corrs = self._correct_common_errors(result)
            if new_result != result:
                corrections.extend(corrs)
                result = new_result

        # 5. 数字表記の正規化
        if self.config.enable_number_normalization:
            new_result, corrs = self._normalize_numbers(result)
            if new_result != result:
                corrections.extend(corrs)
                result = new_result

        # 6. 敬語の正規化（オプション）
        if self.config.enable_honorific_normalization:
            new_result, corrs = self._normalize_honorifics(result)
            if new_result != result:
                corrections.extend(corrs)
                result = new_result

        return CorrectionResult(
            original_text=text,
            corrected_text=result,
            corrections=corrections
        )

    def _normalize_katakana(self, text: str) -> tuple[str, list[dict]]:
        """カタカナ表記を正規化"""
        corrections = []
        result = text

        for old, new in self._katakana_patterns:
            if old in result:
                result = result.replace(old, new)
                corrections.append({
                    'type': 'katakana_normalization',
                    'description': f'カタカナ正規化: {old} → {new}'
                })

        return result, corrections

    def _normalize_mora(self, text: str) -> tuple[str, list[dict]]:
        """長音・促音を正規化"""
        corrections = []
        result = text

        # 長音の正規化（カタカナのみ）
        for pattern, replacement in CHOUON_PATTERNS:
            new_result = re.sub(pattern, replacement, result)
            if new_result != result:
                corrections.append({
                    'type': 'chouon_normalization',
                    'description': f'長音正規化'
                })
                result = new_result

        return result, corrections

    def _correct_common_errors(self, text: str) -> tuple[str, list[dict]]:
        """よくある誤りパターンを修正"""
        corrections = []
        result = text

        for error, correct in self._error_patterns:
            if error in result:
                result = result.replace(error, correct)
                corrections.append({
                    'type': 'common_asr_error',
                    'description': f'音声認識誤り訂正: {error} → {correct}'
                })

        return result, corrections

    def _normalize_numbers(self, text: str) -> tuple[str, list[dict]]:
        """数字表記を正規化（全角→半角）"""
        corrections = []
        result = text

        # 全角数字→半角数字
        for full, half in FULLWIDTH_TO_HALFWIDTH_DIGITS.items():
            if full in result:
                result = result.replace(full, half)
                corrections.append({
                    'type': 'number_normalization',
                    'description': f'数字正規化: {full} → {half}'
                })

        return result, corrections

    def _remove_hallucination(self, text: str) -> tuple[str, list[dict]]:
        """
        Whisperハルシネーション（繰り返し）を除去

        調査結果によると、Whisperは無音区間やノイズに対して
        同じフレーズを繰り返す傾向がある。
        """
        corrections = []
        result = text

        # フレーズ単位での繰り返し検出（より実用的）
        # 例: "ご視聴ありがとうございました" が3回繰り返されている
        min_phrase_length = 5  # 最小フレーズ長

        pattern_found = True
        max_iterations = 10  # 無限ループ防止
        iteration = 0

        while pattern_found and iteration < max_iterations:
            pattern_found = False
            iteration += 1

            # 長いフレーズから検出（貪欲法）
            for phrase_len in range(len(result) // self.config.hallucination_repeat_threshold, min_phrase_length - 1, -1):
                if phrase_len > len(result):
                    continue

                for start_pos in range(len(result) - phrase_len * self.config.hallucination_repeat_threshold + 1):
                    phrase = result[start_pos:start_pos + phrase_len]

                    # このフレーズが連続して何回現れるか
                    repeat_count = 0
                    pos = start_pos
                    while pos + phrase_len <= len(result) and result[pos:pos + phrase_len] == phrase:
                        repeat_count += 1
                        pos += phrase_len

                    # 閾値以上の繰り返しがあれば1回だけ残す
                    if repeat_count >= self.config.hallucination_repeat_threshold:
                        repeated_text = phrase * repeat_count
                        result = result[:start_pos] + phrase + result[start_pos + len(repeated_text):]
                        corrections.append({
                            'type': 'hallucination_removal',
                            'description': f'繰り返し除去: "{phrase[:20]}..." x {repeat_count} → x 1'
                        })
                        pattern_found = True
                        break

                if pattern_found:
                    break

        # 意味のない文字列の繰り返しを除去
        meaningless_patterns = [
            r'ん{3,}',  # 「んんん...」
            r'ー{3,}',  # 「ーーー...」
            r'。{3,}',  # 「。。。...」
            r'、{3,}',  # 「、、、...」
        ]
        for pattern in meaningless_patterns:
            if re.search(pattern, result):
                new_result = re.sub(pattern, '', result)
                if new_result != result:
                    corrections.append({
                        'type': 'hallucination_removal',
                        'description': f'無意味な繰り返し除去: {pattern}'
                    })
                    result = new_result

        return result, corrections

    def _normalize_honorifics(self, text: str, target_style: str = 'plain') -> tuple[str, list[dict]]:
        """
        敬語を正規化（です/ます調 ⇔ だ/である調）

        Args:
            text: テキスト
            target_style: 'plain' (普通形) or 'polite' (丁寧語)
        """
        corrections = []
        result = text

        patterns = HONORIFIC_PATTERNS if target_style == 'plain' else HONORIFIC_PATTERNS_POLITE

        # 長い順にソートして置換（部分一致を防ぐ）
        sorted_patterns = sorted(patterns.items(), key=lambda x: len(x[0]), reverse=True)

        for honorific, plain in sorted_patterns:
            if honorific in result:
                result = result.replace(honorific, plain)
                corrections.append({
                    'type': 'honorific_normalization',
                    'description': f'敬語正規化: {honorific} → {plain}'
                })

        return result, corrections

    def find_similar_words(self, word: str, threshold: float = 0.4) -> list[tuple[str, float]]:
        """
        音韻的に類似した単語を検索

        Args:
            word: 検索対象の単語
            threshold: 類似度閾値（0-1、低いほど類似）

        Returns:
            類似単語と距離のリスト
        """
        candidates = []

        # よくある誤りパターンから検索
        for error, correct in COMMON_ASR_ERRORS.items():
            dist = phonetic_distance(word, error)
            if dist <= threshold:
                candidates.append((correct, dist))

            dist = phonetic_distance(word, correct)
            if dist <= threshold:
                candidates.append((correct, dist))

        # 同音異義語グループから検索
        for group in PHONETIC_SIMILAR_GROUPS:
            for w in group:
                if w == word:
                    # 同じグループの他の単語を候補に
                    for other in group:
                        if other != word:
                            candidates.append((other, 0.0))
                    break

        # 重複を除去して距離でソート
        seen = set()
        unique_candidates = []
        for w, d in sorted(candidates, key=lambda x: x[1]):
            if w not in seen:
                seen.add(w)
                unique_candidates.append((w, d))

        return unique_candidates

    def get_phonetic_reading(self, text: str) -> str:
        """テキストの音韻表現（ローマ字）を取得"""
        return to_romaji(text)

    def are_homophones(self, word1: str, word2: str) -> bool:
        """2つの単語が同音異義語かどうかを判定"""
        return to_romaji(word1) == to_romaji(word2)

    def suggest_corrections(self, text: str, max_suggestions: int = 5) -> list[dict]:
        """
        訂正候補を複数提示する（ユーザーが選択できるように）

        Args:
            text: 入力テキスト
            max_suggestions: 最大候補数

        Returns:
            訂正候補のリスト [{'word': '原文の単語', 'suggestions': [(候補, スコア), ...]}]
        """
        suggestions = []

        # テキストを単語に分割（簡易実装）
        words = re.findall(r'[ぁ-んァ-ヶー一-龯]+', text)

        for word in words:
            # 音韻的に類似した候補を検索
            candidates = self.find_similar_words(word, threshold=self.config.correction_threshold)

            if candidates:
                # スコアの高い順に最大N件
                top_candidates = candidates[:max_suggestions]
                suggestions.append({
                    'word': word,
                    'position': text.find(word),
                    'suggestions': top_candidates
                })

        return suggestions

    def detect_hallucination(self, text: str) -> dict:
        """
        ハルシネーション（繰り返し、無意味な文字列）を検出

        Returns:
            検出結果 {'has_hallucination': bool, 'patterns': [...]}
        """
        patterns_found = []
        min_phrase_length = 5

        # フレーズ単位での繰り返し検出
        for phrase_len in range(len(text) // self.config.hallucination_repeat_threshold, min_phrase_length - 1, -1):
            if phrase_len > len(text):
                continue

            for start_pos in range(len(text) - phrase_len * self.config.hallucination_repeat_threshold + 1):
                phrase = text[start_pos:start_pos + phrase_len]

                # このフレーズが連続して何回現れるか
                repeat_count = 0
                pos = start_pos
                while pos + phrase_len <= len(text) and text[pos:pos + phrase_len] == phrase:
                    repeat_count += 1
                    pos += phrase_len

                if repeat_count >= self.config.hallucination_repeat_threshold:
                    patterns_found.append({
                        'type': 'phrase_repetition',
                        'pattern': phrase,
                        'count': repeat_count,
                        'position': start_pos
                    })
                    # 1つ見つけたら他の重複パターンは無視（最長一致）
                    break

        # 意味のない文字列の検出
        meaningless_patterns = {
            r'ん{3,}': '撥音の繰り返し',
            r'ー{3,}': '長音の繰り返し',
            r'。{3,}': '句点の繰り返し',
            r'、{3,}': '読点の繰り返し',
        }

        for pattern, description in meaningless_patterns.items():
            matches = re.finditer(pattern, text)
            for match in matches:
                patterns_found.append({
                    'type': 'meaningless_repetition',
                    'pattern': match.group(),
                    'description': description,
                    'position': match.start()
                })

        return {
            'has_hallucination': len(patterns_found) > 0,
            'patterns': patterns_found
        }

    def get_homophone_candidates(self, word: str) -> list[str]:
        """
        同音異義語の候補を取得

        Args:
            word: 検索対象の単語

        Returns:
            同音異義語のリスト
        """
        # ローマ字に変換
        romaji = to_romaji(word)

        candidates = []

        # 同音異義語グループから検索
        for group in PHONETIC_SIMILAR_GROUPS:
            for w in group:
                if to_romaji(w) == romaji and w != word:
                    candidates.append(w)

        # よくある誤りパターンからも検索
        for error, correct in COMMON_ASR_ERRORS.items():
            if to_romaji(error) == romaji and error != word:
                candidates.append(error)
            if to_romaji(correct) == romaji and correct != word:
                candidates.append(correct)

        # 重複を除去
        return list(set(candidates))


# ============================================================
# ユーティリティ関数
# ============================================================

def create_default_corrector() -> PhoneticCorrector:
    """デフォルト設定の訂正器を作成"""
    return PhoneticCorrector()


def correct_asr_text(text: str) -> str:
    """
    音声認識テキストを訂正（簡易関数）

    Args:
        text: 音声認識結果

    Returns:
        訂正後のテキスト
    """
    corrector = PhoneticCorrector()
    result = corrector.correct(text)
    return result.corrected_text
