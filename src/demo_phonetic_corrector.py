#!/usr/bin/env python3
"""
phonetic_corrector.py の使用例デモ

拡張機能のデモンストレーション:
1. IT用語の訂正
2. ハルシネーション検出・除去
3. 医療用語の訂正
4. 訂正候補の提示
5. 敬語の正規化
"""

from phonetic_corrector import (
    PhoneticCorrector,
    PhoneticCorrectorConfig,
    phonetic_distance
)


def demo_it_terms():
    """IT用語の訂正デモ"""
    print("=" * 60)
    print("デモ1: IT用語の訂正")
    print("=" * 60)

    config = PhoneticCorrectorConfig()
    corrector = PhoneticCorrector(config)

    text = """
    パイソンとギットハブを使ってエーピーアイを作りました。
    バックエンドはフラスクとマイエスキューエル、
    フロントエンドはリアクトとタイプスクリプトです。
    ドッカーとクーバネティスでデプロイします。
    """

    result = corrector.correct(text)

    print(f"入力テキスト:")
    print(result.original_text)
    print()
    print(f"訂正後:")
    print(result.corrected_text)
    print()
    print(f"訂正数: {len(result.corrections)}")
    for i, corr in enumerate(result.corrections, 1):
        print(f"  {i}. {corr['description']}")
    print()


def demo_hallucination():
    """ハルシネーション検出・除去デモ"""
    print("=" * 60)
    print("デモ2: ハルシネーション検出・除去")
    print("=" * 60)

    config = PhoneticCorrectorConfig()
    corrector = PhoneticCorrector(config)

    # Whisperが無音区間で繰り返すパターン
    text = "ご視聴ありがとうございましたご視聴ありがとうございましたご視聴ありがとうございました"

    # 検出
    hallucination_info = corrector.detect_hallucination(text)
    print(f"入力テキスト:")
    print(text)
    print()
    print(f"ハルシネーション検出: {hallucination_info['has_hallucination']}")
    if hallucination_info['has_hallucination']:
        for pattern in hallucination_info['patterns']:
            print(f"  - タイプ: {pattern['type']}")
            print(f"    パターン: \"{pattern['pattern'][:30]}...\"")
            print(f"    繰り返し回数: {pattern['count']}")
    print()

    # 除去
    result = corrector.correct(text)
    print(f"訂正後:")
    print(result.corrected_text)
    print(f"元の長さ: {len(text)}文字 → 訂正後: {len(result.corrected_text)}文字")
    print()


def demo_medical_terms():
    """医療用語の訂正デモ"""
    print("=" * 60)
    print("デモ3: 医療用語の訂正")
    print("=" * 60)

    config = PhoneticCorrectorConfig()
    corrector = PhoneticCorrector(config)

    text = """
    花竹の治療と心レンズの検査を行いました。
    花地が出ていて、ビフェも見られます。
    """

    result = corrector.correct(text)

    print(f"入力テキスト:")
    print(result.original_text)
    print()
    print(f"訂正後:")
    print(result.corrected_text)
    print()
    for corr in result.corrections:
        print(f"  - {corr['description']}")
    print()


def demo_suggestions():
    """訂正候補の提示デモ"""
    print("=" * 60)
    print("デモ4: 訂正候補の提示")
    print("=" * 60)

    config = PhoneticCorrectorConfig(correction_threshold=0.5)
    corrector = PhoneticCorrector(config)

    text = "かんこうちに機関を訪問しました"

    suggestions = corrector.suggest_corrections(text)

    print(f"入力テキスト: {text}")
    print()
    print("訂正候補:")
    if suggestions:
        for sugg in suggestions:
            word = sugg['word']
            candidates = sugg['suggestions']
            print(f"  単語「{word}」の候補:")
            for cand, score in candidates[:5]:
                print(f"    - {cand} (類似度: {1-score:.3f})")
    else:
        print("  (候補なし)")
    print()


def demo_honorific_normalization():
    """敬語の正規化デモ"""
    print("=" * 60)
    print("デモ5: 敬語の正規化")
    print("=" * 60)

    # 敬語を普通形に
    config = PhoneticCorrectorConfig(enable_honorific_normalization=True)
    corrector = PhoneticCorrector(config)

    text = "明日会議があります。資料を拝見します。"

    result = corrector.correct(text)

    print(f"入力テキスト (敬語あり):")
    print(result.original_text)
    print()
    print(f"訂正後 (普通形):")
    print(result.corrected_text)
    print()


def demo_phonetic_distance():
    """音韻距離の計算デモ"""
    print("=" * 60)
    print("デモ6: 音韻距離の計算")
    print("=" * 60)

    # 音韻的に類似した単語ペア
    pairs = [
        ("きかん", "ききん"),   # 期間 vs 飢饉
        ("こうしょう", "こうじょう"),  # 交渉 vs 工場
        ("かんこう", "かんごう"),  # 観光 vs 勘考
        ("しこう", "しぎょう"),  # 思考 vs 施行
    ]

    print("音韻距離の計算（0に近いほど類似）:")
    print()
    for w1, w2 in pairs:
        dist = phonetic_distance(w1, w2)
        print(f"  「{w1}」と「{w2}」: {dist:.3f}")
    print()


def main():
    """すべてのデモを実行"""
    print()
    print("╔" + "=" * 58 + "╗")
    print("║" + " " * 10 + "phonetic_corrector.py デモ" + " " * 21 + "║")
    print("╚" + "=" * 58 + "╝")
    print()

    demo_it_terms()
    demo_hallucination()
    demo_medical_terms()
    demo_suggestions()
    demo_honorific_normalization()
    demo_phonetic_distance()

    print("=" * 60)
    print("デモ終了")
    print("=" * 60)


if __name__ == "__main__":
    main()
