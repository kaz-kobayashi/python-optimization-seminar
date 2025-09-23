#!/usr/bin/env python3
"""
セッション1: 数理最適化の基礎とPython環境構築
orseminar1.mdのスライドで示した例題を実行可能な形式で提供

このプログラムは以下の内容を含みます：
1. 簡単な線形計画問題の例
2. 詳細な実装例（ステップバイステップ）
3. 抽象モデルの例（データとモデルの分離）
4. 具象モデル vs 抽象モデルの比較
"""
import pyomo.environ as pyo
import runpy

def simple_lp_example():
    """
    簡単な線形計画問題の例
    最大化: z = 2x₁ + 3x₂
    制約条件: 3x₁ + 4x₂ ≥ 1
             x₁ + x₂ ≤ 2
             x₁ ≥ 0, x₂ ≥ 0
    """
    print("=== 簡単な線形計画問題 ===")
    print("最大化: z = 2x₁ + 3x₂")
    print("制約条件: 3x₁ + 4x₂ ≥ 1")
    print("         x₁ + x₂ ≤ 2")
    print("         x₁ ≥ 0, x₂ ≥ 0")
    print()
    
    # モデルの作成
    model = pyo.ConcreteModel()
    
    # 変数の定義
    model.x = pyo.Var([1,2], domain=pyo.NonNegativeReals)
    
    # 目的関数の定義（最大化）
    model.OBJ = pyo.Objective(expr=2*model.x[1] + 3*model.x[2], sense=pyo.maximize)
    
    # 制約条件の定義
    model.Constraint1 = pyo.Constraint(expr=3*model.x[1] + 4*model.x[2] >= 1)
    model.Constraint2 = pyo.Constraint(expr=model.x[1] + model.x[2] <= 2)
    
    # ソルバーの実行（HiGHS推奨）
    opt = pyo.SolverFactory('appsi_highs')
    
    print("ソルバー実行中...")
    results = opt.solve(model)
    
    # 結果の表示
    print("\n=== 結果 ===")
    print(f"ソルバー状態: {results.solver.termination_condition}")
    print(f"x[1] = {pyo.value(model.x[1]):.6f}")
    print(f"x[2] = {pyo.value(model.x[2]):.6f}")
    print(f"目的関数値: z = {pyo.value(model.OBJ):.6f}")
    print()
    
    return model, results

def detailed_example():
    """
    より詳細な実装例（変数・目的関数・制約条件の定義を分けて説明）
    """
    print("=== 詳細な実装例 ===")
    
    # Step 1: モデルの作成
    print("Step 1: モデルの作成")
    from pyomo.environ import ConcreteModel, Var, NonNegativeReals, Objective, maximize, Constraint
    model = ConcreteModel()
    
    # Step 2: 変数の定義
    print("Step 2: 変数の定義")
    model.x = Var([1,2], domain=NonNegativeReals)  # x[1] ≥ 0, x[2] ≥ 0
    print("  - x[1], x[2]: 非負の実数変数")
    
    # Step 3: 目的関数の定義
    print("Step 3: 目的関数の定義")
    model.OBJ = Objective(expr=2*model.x[1] + 3*model.x[2], sense=maximize)
    print("  - 最大化: 2*x[1] + 3*x[2]")
    
    # Step 4: 制約条件の定義
    print("Step 4: 制約条件の定義")
    model.Constraint1 = Constraint(expr=3*model.x[1] + 4*model.x[2] >= 1)
    model.Constraint2 = Constraint(expr=model.x[1] + model.x[2] <= 2)
    print("  - 制約1: 3*x[1] + 4*x[2] >= 1")
    print("  - 制約2: x[1] + x[2] <= 2")
    
    # Step 5: ソルバーの実行
    print("\nStep 5: ソルバーの実行")
    opt = pyo.SolverFactory('appsi_highs')
    results = opt.solve(model)
    
    # Step 6: 結果の表示
    print("\nStep 6: 結果の表示")
    print(f"  - x[1] = {pyo.value(model.x[1]):.6f}")
    print(f"  - x[2] = {pyo.value(model.x[2]):.6f}")
    print(f"  - 目的関数値 = {pyo.value(model.OBJ):.6f}")
    
    # より詳細な結果情報
    print("\n詳細な結果情報:")
    print(results)
    
    return model, results


def abstract_model_example():
    """
    抽象モデルの例
    orseminar1.mdのスライドで示された抽象モデルとデータ分離の例
    """
    print("=== 抽象モデルの例 ===")
    print("データとモデルを分離した実装方法")
    print("同じモデル構造で異なるデータセットを解く")
    print()
    
    # 外部ファイルからモデルを読み込み
    print("Step 1: 外部ファイルからモデルを読み込み")
    data_path = "./pyomo_abstract_model.dat"
    model_path = "./pyomo_abstract_model.py"
    
    model = runpy.run_path(str(model_path))["model"]
    instance = model.create_instance(str(data_path))
    
    print(f"  - モデルファイル: {model_path}")
    print(f"  - データファイル: {data_path}")
    
    # データセット1での実行
    print("\nStep 2: データセット1での実行")
    print("  - c[1] = 2, c[2] = 3")
    
    opt = pyo.SolverFactory('appsi_highs')
    res = opt.solve(instance)
    
    print("状態:", res.solver.status)
    print("最適値:", pyo.value(instance.OBJ))
    print("最適解 (x):")
    for j in instance.J:
        print(f"  x[{j}] = {pyo.value(instance.x[j]):.4f}")
    
    # パラメータ変更による再実行
    print("\nStep 3: パラメータ変更による再実行")
    
    data_path2 = "./pyomo_abstract_model2.dat"
    model2 = runpy.run_path(str(model_path))["model"]  # モデルの定義
    
    instance1 = model2.create_instance(str(data_path))   
    instance2 = model2.create_instance(str(data_path2))  # modelは同じでデータは異なる
    
    res1 = opt.solve(instance1)
    res2 = opt.solve(instance2)
    
    print("状態1:", res1.solver.status, " 状態2:", res2.solver.status)
    print("最適値1:", pyo.value(instance1.OBJ))
    print("最適値2:", pyo.value(instance2.OBJ))
    
    print("\n抽象モデルの利点:")
    print("  - 同じモデル構造で異なるデータを簡単に解ける")
    print("  - データ変更のたびにモデルを再構築する必要がない")
    print("  - 大規模な問題やデータが外部から読み込まれる場合に有効")
    
    print()
    return model, instance1, instance2

def concrete_vs_abstract_comparison():
    """
    具象モデルと抽象モデルの使い分けについて説明
    """
    print("=== 具象モデル vs 抽象モデル ===")
    print()
    
    print("【具象モデル (ConcreteModel)】")
    print("  - データを直接モデルに組み込む")
    print("  - 小規模な問題向け")
    print("  - プログラム内でデータが生成される場合")
    print("  - 実装が直感的")
    print()
    
    print("【抽象モデル (AbstractModel)】")
    print("  - データとモデルを分離")
    print("  - 大規模な問題向け")
    print("  - 外部データファイルを使用")
    print("  - 同じモデル構造で複数のデータセットを解ける")
    print("  - パラメータ変更が容易")
    print()
    
    print("【使い分けの指針】")
    print("  - 学習・試作段階 → 具象モデル")
    print("  - 実務・運用段階 → 抽象モデル")
    print("  - データサイズが大きい → 抽象モデル")
    print("  - パラメータスタディ → 抽象モデル")
    print()


def main():
    """メイン実行関数"""
    print("=" * 60)
    print("セッション1: 数理最適化の基礎とPython環境構築")
    print("ハンズオン用プログラム")
    print("=" * 60)
    print()
    
    # 1. 簡単な線形計画問題
    simple_lp_example()
    
    print("\n" + "-" * 60 + "\n")
    
    # 2. 詳細な実装例
    detailed_example()
    
    print("\n" + "-" * 60 + "\n")
    
    # 3. 抽象モデルの例
    abstract_model_example()
    
    print("\n" + "-" * 60 + "\n")
    
    # 4. 具象モデル vs 抽象モデルの比較
    concrete_vs_abstract_comparison()
    
    print("\n" + "=" * 60)
    print("実行完了！")
    print()
    print("【次のステップ】")
    print("1. 各例題を個別に実行してみる")
    print("2. パラメータを変更して結果を観察する")
    print("3. 独自の最適化問題を定式化してみる")
    print("=" * 60)

if __name__ == "__main__":
    main()