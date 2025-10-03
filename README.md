# 2025年度 第3回ORセミナー『Pythonによる最適化』

2025年度 第3回ORセミナー『Pythonによる最適化』のハンズオン用教材を提供しています。

## 概要

このリポジトリには、数理最適化の基礎とPython環境構築について学ぶためのサンプルコードが含まれています。

## セットアップ

### 1. 環境構築

#### uvのインストール
- インストール手順は [https://docs.astral.sh/uv/getting-started/](https://docs.astral.sh/uv/getting-started/) に従ってください
- macOS/Linux: `curl -LsSf https://astral.sh/uv/install.sh | sh`

#### 仮想環境の作成と有効化
```bash
uv venv .venv
source .venv/bin/activate  # Windows: .\.venv\Scripts\activate
```

#### 必要なパッケージのインストール
```bash
uv pip install pyomo
uv pip install highspy  # HiGHS ソルバー（推奨）
```

### 2. 代替ソルバー（必要に応じて）

#### macOS
```bash
brew install cbc
# または
brew install glpk
```

#### Ubuntu/Debian
```bash
sudo apt-get install coinor-cbc
# または  
sudo apt-get install glpk-utils
```

#### Windows
- CBC: [COIN-OR Cbc releases](https://github.com/coin-or/Cbc/releases) からダウンロード→展開→`bin`をPATHに追加
- GLPK: [winglpk](https://sourceforge.net/projects/winglpk/) をダウンロード→展開→`wbin`をPATHに追加

## ファイル構成

### 実行可能なPythonファイル

- `session1_complete.py` - 第1回セミナーの全体的なハンズオン用プログラム

### 抽象モデル関連ファイル

- `pyomo_abstract_model.py` - 抽象モデルの定義
- `pyomo_abstract_model.dat` - データセット1（c[1]=2, c[2]=3）
- `pyomo_abstract_model2.dat` - データセット2（c[1]=6, c[2]=3）

## 実行方法

### 全体実行
```bash
python session1_complete.py
```

### 個別実行例

#### 簡単な線形計画問題
```python
from session1_complete import simple_lp_example
simple_lp_example()
```

#### 抽象モデルの例
```python
from session1_complete import abstract_model_example
abstract_model_example()
```

## 学習内容

1. **簡単な線形計画問題** - 基本的な線形計画問題の定式化と求解
2. **詳細な実装例** - ステップバイステップでの実装方法
3. **抽象モデルの例** - データとモデルの分離による柔軟な問題設定
4. **具象モデル vs 抽象モデルの比較** - 使い分けの指針

## 問題設定

### 線形計画問題
- **目的関数**: 最大化 z = 2x₁ + 3x₂
- **制約条件**: 
  - 3x₁ + 4x₂ ≥ 1
  - x₁ + x₂ ≤ 2
  - x₁ ≥ 0, x₂ ≥ 0

### 抽象モデル
- 同じモデル構造で異なるパラメータセットを試すことができます
- 外部データファイルから問題設定を読み込む実践的な方法を学習できます

## トラブルシューティング

### ソルバーが見つからない場合
```python
import pyomo.environ as pyo

# 利用可能なソルバーの確認
for solver_name in ['appsi_highs', 'glpk', 'cbc']:
    try:
        solver = pyo.SolverFactory(solver_name)
        if solver.available():
            print(f"{solver_name}: 利用可能")
        else:
            print(f"{solver_name}: インストールされていません")
    except:
        print(f"{solver_name}: エラー")
```

### 動作確認
```bash
# ソルバーのバージョン確認
highs --version  # highspyは内蔵のため不要
cbc --version
glpsol --version
```

## 次のステップ

1. 各例題を個別に実行してみる
2. パラメータを変更して結果を観察する
3. 独自の最適化問題を定式化してみる

## 参考資料

- [Pyomo公式ドキュメント](https://pyomo.readthedocs.io/)
- [HiGHS ソルバー](https://highs.dev/)
- [uv パッケージマネージャー](https://docs.astral.sh/uv/)

---

© 2025 小林 和博
