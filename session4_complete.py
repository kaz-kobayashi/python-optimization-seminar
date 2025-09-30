#!/usr/bin/env python3
"""
第4回セッション - 非線形計画法と機械学習問題

このプログラムは以下の内容を含みます：
1. 非線形最適化の基礎理論
2. ロジスティック回帰の実装
3. サポートベクターマシン（SVM）の実装
4. 非凸問題への対応（多スタート法）
"""

from pyomo.environ import *
import numpy as np

# ==============================================================================
# 1. 非線形最適化の基礎
# ==============================================================================

def print_theory_background():
    """
    非線形最適化の理論的背景の説明
    """
    print("=" * 70)
    print("1. 非線形最適化の基礎")
    print("=" * 70)
    print()
    
    print("【非線形計画問題 (NLP: Nonlinear Programming)】")
    print("  最小化  f(x)")
    print("  制約   g_i(x) ≤ 0,  i = 1, ..., m")
    print("         h_j(x) = 0,  j = 1, ..., p")
    print()
    print("  f(x): 目的関数（非線形）")
    print("  g_i(x): 不等式制約（非線形可）")
    print("  h_j(x): 等式制約（非線形可）")
    print()
    
    print("【線形関数の例】")
    print("  - f(x) = 2x")
    print("  - f(x,y) = x + 2y")
    print("  - 加法性: f(x + y) = f(x) + f(y)")
    print("  - 斉次性: f(αx) = αf(x)")
    print()
    
    print("【非線形関数の例】")
    print("  - f(x) = x² + 2x + 1")
    print("  - f(x,y) = x² + y²")
    print("  - f(x) = e^x")
    print("  - f(x) = log(x)")
    print("  - f(x,y) = xy")
    print()
    
    print("【凸最適化と非凸最適化】")
    print("  凸最適化問題:")
    print("    - 目的関数が凸関数、実行可能領域が凸集合")
    print("    - 局所最適解 = 大域最適解")
    print("    - 効率的に解ける")
    print()
    print("  非凸最適化問題:")
    print("    - 目的関数または実行可能領域が非凸")
    print("    - 複数の局所最適解が存在する可能性")
    print("    - 大域最適解の保証が困難")
    print()

def print_solver_info():
    """
    主要な非線形ソルバーの情報
    """
    print("=" * 70)
    print("2. 主要な非線形ソルバー")
    print("=" * 70)
    print()
    
    print("【IPOPT (Interior Point OPTimizer)】")
    print("  - アルゴリズム: 主双対内点法")
    print("  - 対象問題: 大規模な連続変数NLP")
    print("  - 特徴:")
    print("    * スパース行列の効率的な処理")
    print("    * 二次収束性")
    print("    * ロバスト性が高い")
    print("  - 制限事項:")
    print("    * 整数変数は扱えない（連続変数のみ）")
    print("    * 非凸問題では局所最適解")
    print()
    
    print("【その他のソルバー】")
    print("  - KNITRO: 商用、SQP/内点法、高速・高精度")
    print("  - SNOPT: 商用、SQP法、スパース問題")
    print("  - Bonmin: 混合整数非線形（MINLP）")
    print("  - BARON: 商用、大域最適化、非凸MINLP")
    print()

# ==============================================================================
# 2. ロジスティック回帰
# ==============================================================================

def logistic_regression_example():
    """
    ロジスティック回帰の実装
    """
    print("=" * 70)
    print("3. ロジスティック回帰の実装")
    print("=" * 70)
    print("問題: 2値分類問題")
    print("確率モデル: P(y=1|x) = 1/(1 + e^(-(β₀ + βᵀx)))")
    print("最適化: 負の対数尤度の最小化")
    print()
    
    # サンプルデータの生成
    np.random.seed(42)
    n_samples = 20
    X = np.vstack([np.random.randn(10, 2) + [2, 2],
                   np.random.randn(10, 2) + [-2, -2]])
    y = np.array([1]*10 + [0]*10)
    
    # Pyomoモデル
    model = ConcreteModel()
    
    # インデックスセット
    model.I = RangeSet(0, n_samples-1)  # サンプル
    model.J = RangeSet(0, 1)            # 特徴量
    
    # 決定変数
    model.beta = Var(model.J, initialize=0)  # 回帰係数
    model.beta0 = Var(initialize=0)          # 切片
    
    # 目的関数（負の対数尤度）
    def negative_log_likelihood(model):
        loss = 0
        for i in model.I:
            # 線形結合
            z = model.beta0 + sum(model.beta[j] * X[i,j] for j in model.J)
            # シグモイド関数
            p = 1 / (1 + exp(-z))
            # 対数尤度
            loss += -y[i]*log(p) - (1-y[i])*log(1-p)
        return loss
    
    model.obj = Objective(rule=negative_log_likelihood, sense=minimize)
    
    # ソルバーの設定と実行
    print("IPOPTソルバーで最適化実行中...")
    solver = SolverFactory('ipopt')
    
    # IPOPTが利用可能かチェック
    if not solver.available():
        print("\n警告: IPOPTが利用できません")
        print("インストール方法:")
        print("  macOS: brew install ipopt")
        print("  Ubuntu: sudo apt-get install coinor-ipopt")
        print("  Windows: conda install -c conda-forge ipopt")
        return None
    
    solver.options['print_level'] = 5  # 詳細ログ
    result = solver.solve(model, tee=True)
    
    # 結果の表示
    print(f"\n最適化状態: {result.solver.termination_condition}")
    print(f"beta0 = {value(model.beta0):.4f}")
    for j in model.J:
        print(f"beta[{j}] = {value(model.beta[j]):.4f}")
    
    # 予測精度の確認
    correct = 0
    for i in range(n_samples):
        z = value(model.beta0) + sum(value(model.beta[j])*X[i,j] for j in model.J)
        pred = 1 if z > 0 else 0
        correct += (pred == y[i])
    print(f"正解率: {correct/n_samples*100:.1f}%")
    
    print("\n【IPOPTの収束特性】")
    print("- 初期段階（iter 0-5）: 急速な減少")
    print("- 中間段階（iter 6-14）: 二次収束")
    print("- 最終段階（iter 15-22）: 高精度収束")
    print()
    
    return model

# ==============================================================================
# 3. サポートベクターマシン
# ==============================================================================

def svm_example():
    """
    サポートベクターマシンの実装
    """
    print("=" * 70)
    print("4. サポートベクターマシン（SVM）の実装")
    print("=" * 70)
    print("問題: ソフトマージンSVMによる2クラス分類")
    print("目的関数: (1/2)||w||² + C∑ξᵢ")
    print("制約: yᵢ(wᵀxᵢ + b) ≥ 1 - ξᵢ, ξᵢ ≥ 0")
    print()
    
    # データ準備（ロジスティック回帰と同じ）
    np.random.seed(42)
    n_samples = 20
    X = np.vstack([np.random.randn(10, 2) + [2, 2],
                   np.random.randn(10, 2) + [-2, -2]])
    y = np.array([1]*10 + [-1]*10)  # SVMでは+1/-1を使用
    
    # Pyomoモデル
    model = ConcreteModel()
    model.I = RangeSet(0, n_samples-1)
    model.J = RangeSet(0, 1)
    
    # 正則化パラメータ
    C = 1.0
    
    # 決定変数
    model.w = Var(model.J, initialize=0)        # 重みベクトル
    model.b = Var(initialize=0)                 # バイアス項
    model.xi = Var(model.I, domain=NonNegativeReals)  # スラック変数
    
    # 目的関数
    def objective_rule(model):
        regularization = 0.5 * sum(model.w[j]**2 for j in model.J)
        slack_penalty = C * sum(model.xi[i] for i in model.I)
        return regularization + slack_penalty
    
    model.obj = Objective(rule=objective_rule, sense=minimize)
    
    # 制約条件
    def margin_constraint_rule(model, i):
        return y[i] * (sum(model.w[j]*X[i,j] for j in model.J) + model.b) >= 1 - model.xi[i]
    
    model.margin_constraint = Constraint(model.I, rule=margin_constraint_rule)
    
    # 求解
    print("IPOPTソルバーで最適化実行中...")
    solver = SolverFactory('ipopt')
    
    if not solver.available():
        print("\n警告: IPOPTが利用できません")
        return None
    
    result = solver.solve(model, tee=True)
    
    print(f"\n最適化状態: {result.solver.termination_condition}")
    
    # 決定境界の表示
    print(f"w = [{value(model.w[0]):.4f}, {value(model.w[1]):.4f}]")
    print(f"b = {value(model.b):.4f}")
    
    # サポートベクターの特定
    support_vectors = []
    for i in model.I:
        margin = y[i] * (sum(value(model.w[j])*X[i,j] for j in model.J) + value(model.b))
        if abs(margin - 1) < 1e-3 or value(model.xi[i]) > 1e-3:
            support_vectors.append(i)
    print(f"サポートベクター数: {len(support_vectors)}")
    
    print("\n【SVMの特徴】")
    print("- 目的関数: 二次関数（凸関数）")
    print("- 制約条件: 線形不等式制約（凸制約）")
    print("- 問題全体: 凸二次計画問題（QP）")
    print("- 大域最適解が保証される")
    print()
    
    return model

# ==============================================================================
# 4. 非凸問題への対応
# ==============================================================================

def multi_start_method_example():
    """
    非凸問題への対応: 多スタート法
    """
    print("=" * 70)
    print("5. 非凸問題への対応 - 多スタート法")
    print("=" * 70)
    print("複数の初期点から複数の局所最適解を求める")
    print()
    
    # 簡単な非凸問題を作成
    model = ConcreteModel()
    model.x = Var(bounds=(-5, 5))
    model.y = Var(bounds=(-5, 5))
    
    # 非凸目的関数の例
    model.obj = Objective(
        expr=(model.x - 1)**2 + (model.y - 1)**2 + 0.1*sin(10*model.x)*sin(10*model.y),
        sense=minimize
    )
    
    print("目的関数: (x-1)² + (y-1)² + 0.1*sin(10x)*sin(10y)")
    print("真の最適解: (1, 1) 付近")
    print()
    
    solver = SolverFactory('ipopt')
    if not solver.available():
        print("警告: IPOPTが利用できません")
        return
    
    best_obj = float('inf')
    best_solution = None
    
    print("10個の初期点から最適化を実行:")
    for trial in range(10):
        # ランダムな初期値
        for v in model.component_data_objects(Var):
            v.set_value(np.random.randn())
        
        result = solver.solve(model)
        if value(model.obj) < best_obj:
            best_obj = value(model.obj)
            best_solution = {v.name: value(v) for v in model.component_data_objects(Var)}
        
        print(f"  試行{trial+1}: 目的関数値 = {value(model.obj):.6f}")
    
    print(f"\n最良解:")
    print(f"  x = {best_solution['x']:.4f}")
    print(f"  y = {best_solution['y']:.4f}")
    print(f"  目的関数値 = {best_obj:.6f}")
    print()

# ==============================================================================
# メイン実行関数
# ==============================================================================

def main():
    """メイン実行関数"""
    print("=" * 70)
    print("第4回セッション - 非線形計画法と機械学習問題")
    print("=" * 70)
    print()
    
    # 1. 理論的背景
    print_theory_background()
    
    print("\n" + "-" * 70 + "\n")
    
    # 2. ソルバー情報
    print_solver_info()
    
    print("\n" + "-" * 70 + "\n")
    
    # 3. ロジスティック回帰
    logistic_model = logistic_regression_example()
    
    print("\n" + "-" * 70 + "\n")
    
    # 4. サポートベクターマシン
    svm_model = svm_example()
    
    print("\n" + "-" * 70 + "\n")
    
    # 5. 非凸問題への対応
    multi_start_method_example()
    
    print("=" * 70)
    print("すべての実行が完了しました！")
    print()
    print("【本セッションで学んだこと】")
    print("1. 非線形最適化の基礎")
    print("   - 凸最適化と非凸最適化の違い")
    print("   - 大域最適解と局所最適解")
    print()
    print("2. 非線形ソルバー")
    print("   - IPOPTを中心とした各種ソルバーの特徴")
    print("   - 問題に応じた選択基準")
    print()
    print("3. Pyomoによる実装")
    print("   - ロジスティック回帰（凸最適化）")
    print("   - SVM（二次計画問題）")
    print("=" * 70)

if __name__ == "__main__":
    main()