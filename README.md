# paper-embedding-demo

## 概要

論文を埋め込みに変換して時代ごとの傾向や全体的に人気のあるトピックを把握するためのデモ

## 使い方

環境構築

```bash
poetry install
export SAMBANOVA_API_KEY=YOUR_API_KEY
```

前処理（データの取得から埋め込み計算）

```bash
poetry run python3 preprocess.py
```

デモの立ち上げ

```bash
poetry run streamlit run app.py
```
