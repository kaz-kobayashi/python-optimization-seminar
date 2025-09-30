#!/usr/bin/env python3
"""
セッション2: 線形計画法と輸送問題
orseminar2.mdのスライドで示した全ての例題を実行可能な形式で提供

このプログラムは以下の内容を含みます：
1. 生産計画問題の実装と感度分析
2. 輸送問題の実装と感度分析
3. 異なるソルバーでの実行比較
"""
import pyomo.environ as pyo
from pyomo.environ import ConcreteModel, Var, NonNegativeReals, Objective, maximize, minimize, Constraint, SolverFactory, value
import time

def production_planning_problem():
    """
    生産計画問題の実装
    2種類の製品A, Bを生産して利益を最大化
    - 製品A: 利益 3, 資源消費量 2
    - 製品B: 利益 2, 資源消費量 1
    - 資源の総量: 4
    """
    print("=== 生産計画問題 ===")
    print("問題設定:")
    print("  - 製品A: 利益 3, 資源消費量 2")
    print("  - 製品B: 利益 2, 資源消費量 1")
    print("  - 資源の総量: 4")
    print()
    
    # モデルの作成
    model = ConcreteModel()
    
    # 変数の定義
    model.x = pyo.Var(['A','B'], domain=NonNegativeReals)
    print("変数の定義:")
    print("  - x['A']: 製品Aの生産量")
    print("  - x['B']: 製品Bの生産量")
    
    # 目的関数の定義
    model.profit = Objective(expr=3*model.x['A'] + 2*model.x['B'], sense=maximize)
    print("目的関数: 最大化 3*x['A'] + 2*x['B']")
    
    # 制約条件の定義
    model.resource = Constraint(expr=2*model.x['A'] + 1*model.x['B'] <= 4)
    print("制約条件: 2*x['A'] + 1*x['B'] <= 4")
    
    # 双対変数（シャドウプライス）の取得設定
    model.dual = pyo.Suffix(direction=pyo.Suffix.IMPORT)
    
    # ソルバーの実行
    print("\nソルバー実行中...")
    solver = SolverFactory('appsi_highs')
    res = solver.solve(model)
    
    # 結果の表示
    print("\n=== 結果 ===")
    print(f"状態: {res.solver.status}")
    print(f"製品A生産量: {value(model.x['A'])}")
    print(f"製品B生産量: {value(model.x['B'])}")
    print(f"最大利益: {value(model.profit)}")
    
    # 感度分析（シャドウプライス）
    print("\n=== 感度分析 ===")
    lam = model.dual[model.resource]
    print(f"資源制約のシャドウプライス: {lam}")
    print("解釈: 資源が1単位増加すると最大利益が{:.1f}増加".format(lam))
    
    # 感度分析の検証（資源を1増加）
    print("\n=== 感度分析の検証 ===")
    print("資源の総量を5に変更して再実行...")
    
    # 新しいモデルで検証
    model2 = ConcreteModel()
    model2.x = pyo.Var(['A','B'], domain=NonNegativeReals)
    model2.profit = Objective(expr=3*model2.x['A'] + 2*model2.x['B'], sense=maximize)
    model2.resource = Constraint(expr=2*model2.x['A'] + 1*model2.x['B'] <= 5)
    
    res2 = solver.solve(model2)
    new_profit = value(model2.profit)
    print(f"新しい最大利益: {new_profit}")
    print(f"利益の増加: {new_profit - value(model.profit)}")
    print(f"シャドウプライス: {lam}")
    
    print()
    return model, res

def transportation_problem():
    """
    輸送問題の実装
    複数の供給地から複数の需要地への輸送コスト最小化
    """
    print("=== 輸送問題 ===")
    print("問題設定: 複数の供給地から複数の需要地に製品を輸送")
    print("目標: 総輸送コストの最小化")
    print()
    
    # モデルの作成
    model = ConcreteModel()
    
    # ノード集合の定義
    model.S = pyo.Set(initialize=["Tokyo", "Osaka", "Nagoya"])  # 供給地
    model.D = pyo.Set(initialize=["Yokohama", "Kyoto", "Sapporo", "Fukuoka"])  # 需要地
    
    print("供給地:", list(model.S))
    print("需要地:", list(model.D))
    
    # パラメータの定義
    # 供給量
    supply = {"Tokyo": 150, "Osaka": 200, "Nagoya": 125}
    model.supply = pyo.Param(model.S, initialize=supply, mutable=True)
    
    # 需要量
    demand = {"Yokohama": 120, "Kyoto": 80, "Sapporo": 150, "Fukuoka": 100}
    model.demand = pyo.Param(model.D, initialize=demand)
    
    print("\n供給量:", {s: value(model.supply[s]) for s in model.S})
    print("需要量:", {d: value(model.demand[d]) for d in model.D})
    
    # 輸送コスト（単位量あたり）
    costs = {
        ("Tokyo", "Yokohama"): 20, ("Tokyo", "Kyoto"): 80,
        ("Tokyo", "Sapporo"): 120, ("Tokyo", "Fukuoka"): 180,
        ("Osaka", "Yokohama"): 70, ("Osaka", "Kyoto"): 40,
        ("Osaka", "Sapporo"): 150, ("Osaka", "Fukuoka"): 100,
        ("Nagoya", "Yokohama"): 50, ("Nagoya", "Kyoto"): 60,
        ("Nagoya", "Sapporo"): 140, ("Nagoya", "Fukuoka"): 120,
    }
    model.cost = pyo.Param(model.S, model.D, initialize=costs)
    
    # 輸送容量制限
    capacities = {
        ("Tokyo", "Yokohama"): 100, ("Tokyo", "Kyoto"): 60,
        ("Tokyo", "Sapporo"): 100, ("Tokyo", "Fukuoka"): 80,
        ("Osaka", "Yokohama"): 100, ("Osaka", "Kyoto"): 100,
        ("Osaka", "Sapporo"): 120, ("Osaka", "Fukuoka"): 100,
        ("Nagoya", "Yokohama"): 80, ("Nagoya", "Kyoto"): 60,
        ("Nagoya", "Sapporo"): 60, ("Nagoya", "Fukuoka"): 60,
    }
    model.cap = pyo.Param(model.S, model.D, initialize=capacities, mutable=True)
    
    # 変数の定義
    model.x = pyo.Var(model.S, model.D, domain=pyo.NonNegativeReals)
    
    # 目的関数の定義（総輸送コスト最小化）
    def total_cost(m):
        return sum(m.cost[i, j] * m.x[i, j] for i in m.S for j in m.D)
    model.obj = pyo.Objective(rule=total_cost, sense=minimize)
    
    # 制約条件の定義
    # 1. 供給量制約
    def supply_rule(m, i):
        return sum(m.x[i, j] for j in m.D) <= m.supply[i]
    model.supply_con = pyo.Constraint(model.S, rule=supply_rule)
    
    # 2. 需要量制約
    def demand_rule(m, j):
        return sum(m.x[i, j] for i in m.S) >= m.demand[j]
    model.demand_con = pyo.Constraint(model.D, rule=demand_rule)
    
    # 3. 容量制約
    def capacity_rule(m, i, j):
        return m.x[i, j] <= m.cap[i, j]
    model.capacity_con = pyo.Constraint(model.S, model.D, rule=capacity_rule)
    
    # 双対変数の取得設定
    model.dual = pyo.Suffix(direction=pyo.Suffix.IMPORT)
    
    # ソルバーの実行
    print("\nソルバー実行中...")
    solver = pyo.SolverFactory('appsi_highs')
    res = solver.solve(model)
    
    # 結果の表示
    print("\n=== 結果 ===")
    print("状態:", res.solver.status)
    print("最適値（総コスト）:", pyo.value(model.obj))
    print("輸送計画 (非ゼロのみ):")
    for i in model.S:
        for j in model.D:
            v = pyo.value(model.x[i, j])
            if v is not None and v > 1e-8:
                print(f"  {i} -> {j}: {v:.2f}")
    
    return model, res, solver

def transportation_sensitivity_analysis(model, solver):
    """
    輸送問題の感度分析
    """
    print("\n=== 感度分析（容量制約） ===")
    print("容量制約の双対変数（シャドウプライス）:")
    
    for (i, j) in model.capacity_con:
        mu = model.dual.get(model.capacity_con[i, j], None)
        if mu and abs(mu) > 1e-8:
            print(f"  dual[{i}->{j}] = {mu:.6g}")
            print(f"  {i}->{j}の容量を1増やすとコストが{-mu:.6g}減少します。")
    
    print("\n=== 感度分析の検証（容量制約） ===")
    print("Nagoya->Sapporo間の容量を1増やして再実行...")
    
    # 容量を1増加
    original_cap = pyo.value(model.cap["Nagoya", "Sapporo"])
    model.cap["Nagoya", "Sapporo"] = original_cap + 1
    
    res = solver.solve(model)
    
    print("状態:", res.solver.status)
    print("最適値（総コスト）:", pyo.value(model.obj))
    print("輸送計画 (非ゼロのみ):")
    for i in model.S:
        for j in model.D:
            v = pyo.value(model.x[i, j])
            if v is not None and v > 1e-8:
                print(f"  {i} -> {j}: {v:.2f}")

    model.cap["Nagoya", "Sapporo"] = original_cap 

    print("\n=== 感度分析（需給バランス） ===")
    print("供給制約の双対変数:")
    
    for i in model.supply_con:
        mu = model.dual.get(model.supply_con[i], None)
        if mu and abs(mu) > 1e-8:
            print(f"  dual[{i}] = {mu:.6g}")
            print(f"  {i}の供給を1増やすとコストが{-mu:.6g}減少します。")
    
    print("\n=== 感度分析の検証（供給量） ===")
    print("Tokyoの供給量を1増やして再実行...")
    
    # 供給量を1増加
    original_supply = pyo.value(model.supply["Tokyo"])
    model.supply["Tokyo"] = original_supply + 1
    
    res = solver.solve(model)
    
    print("状態:", res.solver.status)
    print("最適値（総コスト）:", pyo.value(model.obj))
    print("輸送計画 (非ゼロのみ):")
    for i in model.S:
        for j in model.D:
            v = pyo.value(model.x[i, j])
            if v is not None and v > 1e-8:
                print(f"  {i} -> {j}: {v:.2f}")

def solver_comparison_demo():
    """
    異なるソルバーでの実行比較のデモ
    """
    print("\n=== ソルバー比較デモ ===")
    
    # 簡単な線形計画問題を作成
    def create_demo_model():
        model = ConcreteModel()
        model.x = pyo.Var(['A', 'B'], domain=NonNegativeReals)
        model.profit = Objective(expr=3*model.x['A'] + 2*model.x['B'], sense=maximize)
        model.resource = Constraint(expr=2*model.x['A'] + 1*model.x['B'] <= 4)
        return model
    
    # 利用可能なソルバーのリスト
    solvers_to_test = []
    
    # HiGHSのテスト
    try:
        solver = pyo.SolverFactory('appsi_highs')
        if solver.available():
            solvers_to_test.append('appsi_highs')
    except:
        pass
    
    # GLPKのテスト
    try:
        solver = pyo.SolverFactory('glpk')
        if solver.available():
            solvers_to_test.append('glpk')
    except:
        pass
        
    print(f"利用可能なソルバー: {solvers_to_test}")
    
    # 各ソルバーで実行
    for solver_name in solvers_to_test:
        print(f"\n--- {solver_name} での実行 ---")
        model = create_demo_model()
        solver = pyo.SolverFactory(solver_name)
        
        # 計算時間の測定
        solve_start = time.time()
        
        try:
            # ソルバーオプション設定（例）
            if hasattr(solver, 'options'):
                if 'time_limit' in dir(solver.options):
                    solver.options['time_limit'] = 10  # 10秒の制限
                if 'mip_rel_gap' in dir(solver.options):
                    solver.options['mip_rel_gap'] = 0.01  # 1%のギャップで停止
            
            results = solver.solve(model)
            solve_time = time.time() - solve_start
            
            print(f"  状態: {results.solver.termination_condition}")
            print(f"  製品A: {value(model.x['A']):.6f}")
            print(f"  製品B: {value(model.x['B']):.6f}")
            print(f"  最大利益: {value(model.profit):.6f}")
            print(f"  計算時間: {solve_time:.4f}秒")
            
        except Exception as e:
            print(f"  エラー: {e}")
    
    print("\n=== ソルバー性能比較の考慮事項 ===")
    print("- 問題規模と時間制約: 大規模問題や速度が重要な場合は商用ソルバーを検討")
    print("- 予算: 商用ソルバーは高価だが、Academic Licenseが利用可能な場合も")
    print("- 問題の特性: 特定の問題タイプに強いソルバーがある")
    print("- 使いやすさ: インターフェース、ドキュメント、サポート体制")
    
    return solvers_to_test

def theoretical_background():
    """
    線形計画法の理論的背景の説明
    """
    print("=== 線形計画法の理論的背景 ===")
    print()
    
    print("【線形計画問題の標準形】")
    print("  目的関数: 最小化 c^T x")
    print("  制約条件: Ax = b, x ≥ 0")
    print("  ここで:")
    print("    c: 目的関数係数ベクトル")
    print("    x: 決定変数ベクトル")
    print("    A: 制約条件係数行列")
    print("    b: 制約条件右辺ベクトル")
    print()
    
    print("【双対問題】")
    print("  主問題:")
    print("    目的関数: 最大化 c^T x")
    print("    制約条件: Ax ≤ b, x ≥ 0")
    print()
    print("  双対問題:")
    print("    目的関数: 最小化 b^T y")
    print("    制約条件: A^T y ≥ c, y ≥ 0")
    print()
    
    print("【双対問題の意義】")
    print("  - 主問題と双対問題の最適値は一致する（強双対定理）")
    print("  - 双対問題から主問題の解に関する情報が得られる（感度分析）")
    print("  - 一部のアルゴリズムは、主問題と双対問題を同時に解く")
    print()

def main():
    """メイン実行関数"""
    print("=" * 60)
    print("セッション2: 線形計画法と輸送問題")
    print("ハンズオン用プログラム")
    print("=" * 60)
    print()
    
    # 1. 理論的背景
    theoretical_background()
    
    print("\n" + "-" * 60 + "\n")
    
    # 2. 生産計画問題
    production_planning_problem()
    
    print("\n" + "-" * 60 + "\n")
    
    # 3. 輸送問題
    model, res, solver = transportation_problem()
    
    # 4. 輸送問題の感度分析
    transportation_sensitivity_analysis(model, solver)
    
    print("\n" + "-" * 60 + "\n")
    
    # 5. ソルバー比較
    solver_comparison_demo()
    
    print("\n" + "=" * 60)
    print("実行完了！")
    print()
    print("【学習内容】")
    print("1. 生産計画問題の定式化と感度分析")
    print("2. 輸送問題の定式化と最適化")
    print("3. 双対変数（シャドウプライス）による感度分析")
    print("4. 異なるソルバーでの性能比較")
    print("=" * 60)

if __name__ == "__main__":
    main()