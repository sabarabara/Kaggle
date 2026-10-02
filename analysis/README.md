# S6E10 feature analysis

既存のEDAを保存し、特徴量の仮説・データ品質・訓練/テスト差・同じ分割での追加比較を行います。

## 実行

プロジェクトのルートから実行してください。Python 3.12を使用します。

```sh
uv venv .venv-analysis --python python3.12
uv pip install --python .venv-analysis/bin/python -r requirements-analysis.txt
.venv-analysis/bin/python -m ipykernel install --prefix work/jupyter --name kaggle-analysis
.venv-analysis/bin/python analysis/build_notebook.py
.venv-analysis/bin/python analysis/execute_notebook.py
.venv-analysis/bin/python analysis/finalize_report.py
```

`data/raw/train.csv` と `data/raw/test.csv` が必要です。
`build_notebook.py` は追加ノートブックの出力をリセットして生成するため、再実行前に必要な出力を保存してください。
ノートブックには特徴量関数を埋め込み、Kaggleでも単独で実行できます。
Kaggleでは標準のPythonカーネルを選択してください。

全件でのEDAに対し、モデル比較は層化抽出12万行・3-foldの探索用です。
モデルの最適化や提出は行っていません。
すべての派生量は行単位で計算し、目的変数/ID/全件ターゲット統計を特徴量に使いません。
