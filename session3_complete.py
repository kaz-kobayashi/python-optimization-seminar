#!/usr/bin/env python3
"""
第3回セッション - 整数計画法と施設配置問題（完全版）
orseminar3.mdのすべての例題を含む統合ファイル

このプログラムは以下の内容を含みます：
1. 分枝限定法のコンセプトとステップバイステップ実行
2. 基本的な施設配置問題
3. 複雑な施設配置問題（分枝限定法のログ確認）
4. 混合整数計画法の理論と実装
"""

import pyomo.environ as pyo
from pyomo.environ import ConcreteModel, Set, Param, Var, Objective, Constraint
from pyomo.environ import Binary, NonNegativeReals, NonNegativeIntegers
from pyomo.environ import minimize, maximize, SolverFactory, value

# ==============================================================================
# 1. 分枝限定法のコンセプト
# ==============================================================================

def branch_and_bound_concept():
    """
    分枝限定法の理論的背景の説明
    """
    print("=" * 70)
    print("1. 分枝限定法の理論的背景")
    print("=" * 70)
    print()
    
    print("【混合整数計画問題 (MIP)】")
    print("  決定変数の一部が整数値に制限された最適化問題")
    print("  - 整数変数: 0, 1, 2, ... のような整数値のみをとる変数")
    print("  - 0-1変数（バイナリ変数）: 0 または 1 の値のみをとる変数")
    print("  - 連続変数: 実数値をとる変数")
    print()
    
    print("【整数変数の導入による効果】")
    print("  - より現実的なモデル（人数、機械台数など）")
    print("  - 複雑な制約の表現（論理的な条件や条件分岐）")
    print()
    
    print("【問題の複雑性】")
    print("  - 線形計画問題: 多項式時間で解ける")
    print("  - 混合整数計画問題: NP困難（計算時間が指数関数的に増加する可能性）")
    print()
    
    print("【分枝限定法の基本的な考え方】")
    print("  1. 整数変数を無視して、線形計画問題（緩和問題）を解く")
    print("  2. 緩和問題の解が整数解であれば、それが最適解")
    print("  3. 整数解でなければ、整数変数の値を分岐させて、部分問題を作成")
    print("  4. 各部分問題に対して、再帰的に分枝限定法を適用")
    print()

def branch_and_bound_example():
    """
    分枝限定法のステップバイステップ実行例
    問題: max 2x + 3y s.t. x + 2y ≤ 8, 2x + y ≤ 9, x ≥ 0, y ≥ 0, x整数
    """
    print("=" * 70)
    print("2. 分枝限定法のステップバイステップ実行")
    print("=" * 70)
    print()
    print("問題:")
    print("  最大化: 2x + 3y")
    print("  制約条件:")
    print("    x + 2y ≤ 8")
    print("    2x + y ≤ 9")
    print("    x ≥ 0, y ≥ 0")
    print("    x は整数")
    print()
    
    def solve_subproblem(x_lower=None, x_upper=None, name="", relax_integer=True):
        """部分問題を解く"""
        model = ConcreteModel()
        
        # 変数の定義
        if relax_integer:
            model.x = Var(domain=NonNegativeReals)
        else:
            model.x = Var(domain=NonNegativeIntegers)
        model.y = Var(domain=NonNegativeReals)
        
        # 目的関数
        model.obj = Objective(expr=2*model.x + 3*model.y, sense=maximize)
        
        # 制約条件
        model.const1 = Constraint(expr=model.x + 2*model.y <= 8)
        model.const2 = Constraint(expr=2*model.x + model.y <= 9)
        
        # 分枝のための追加制約
        if x_lower is not None:
            model.branch_lower = Constraint(expr=model.x >= x_lower)
        if x_upper is not None:
            model.branch_upper = Constraint(expr=model.x <= x_upper)
        
        # 求解
        solver = SolverFactory('appsi_highs')
        result = solver.solve(model)
        
        x_val = value(model.x)
        y_val = value(model.y)
        obj_val = value(model.obj)
        
        print(f"{name}:")
        print(f"  最適解: x = {x_val:.3f}, y = {y_val:.3f}")
        print(f"  最適値: {obj_val:.3f}")
        
        if abs(x_val - round(x_val)) < 1e-6:
            print(f"  ★ x は整数です！")
            return x_val, y_val, obj_val, True
        else:
            print(f"  × x は整数ではありません（分枝が必要）")
            return x_val, y_val, obj_val, False
    
    # ステップ1: 線形緩和問題
    print("【ステップ1】線形緩和問題")
    x0, y0, obj0, is_integer0 = solve_subproblem(name="根ノード（線形緩和）")
    print()
    
    if not is_integer0:
        x_branch = int(x0)
        print(f"分枝: x ≤ {x_branch} と x ≥ {x_branch + 1}")
        print()
        
        # ステップ2: 部分問題1
        print(f"【ステップ2】部分問題1: x ≤ {x_branch}")
        x1, y1, obj1, is_integer1 = solve_subproblem(x_upper=x_branch, name=f"部分問題1 (x ≤ {x_branch})")
        print()
        
        # ステップ3: 部分問題2
        print(f"【ステップ3】部分問題2: x ≥ {x_branch + 1}")
        x2, y2, obj2, is_integer2 = solve_subproblem(x_lower=x_branch + 1, name=f"部分問題2 (x ≥ {x_branch + 1})")
        print()
        
        # ステップ4: 最適解の決定
        print("【ステップ4】最適解の決定")
        if obj1 > obj2:
            print(f"部分問題1の方が良い目的関数値: {obj1:.3f} > {obj2:.3f}")
            print(f"最適解: x = {x1:.0f}, y = {y1:.3f}, 最適値 = {obj1:.3f}")
        else:
            print(f"部分問題2の方が良い目的関数値: {obj2:.3f} > {obj1:.3f}")
            print(f"最適解: x = {x2:.0f}, y = {y2:.3f}, 最適値 = {obj2:.3f}")
        print()
        
        # 検証: 元の混合整数問題を直接解く
        print("【検証】元の混合整数問題を直接解く")
        x_mip, y_mip, obj_mip, _ = solve_subproblem(name="MIP（整数制約あり）", relax_integer=False)
        print("分枝限定法の手動計算と一致することを確認してください。")
        print()

# ==============================================================================
# 2. 施設配置問題
# ==============================================================================

def simple_facility_location():
    """
    基本的な施設配置問題
    2つの施設候補地と3つの顧客
    """
    print("=" * 70)
    print("3. 基本的な施設配置問題")
    print("=" * 70)
    print("施設候補:")
    print("  F1: 建設コスト 10, 容量 20")
    print("  F2: 建設コスト 15, 容量 30")
    print("顧客:")
    print("  A: 需要 5")
    print("  B: 需要 10")
    print("  C: 需要 15")
    print("目的: 建設コスト + 輸送コストの最小化")
    print()
    
    # データ
    facility_cost = {1: 10, 2: 15}
    facility_capacity = {1: 20, 2: 30}
    customer_demand = {1: 5, 2: 10, 3: 15}
    transport_cost = {
        (1, 1): 2, (1, 2): 3, (1, 3): 4,  # 施設1から各顧客へ
        (2, 1): 3, (2, 2): 2, (2, 3): 1   # 施設2から各顧客へ
    }
    
    # モデルの作成
    model = ConcreteModel()
    
    # 集合の定義
    model.I = Set(initialize=facility_cost.keys())  # 施設集合
    model.J = Set(initialize=customer_demand.keys())  # 顧客集合
    
    # 変数の定義
    model.y = Var(model.I, domain=Binary)  # y_i: 施設iを建設するか
    model.x = Var(model.I, model.J, domain=NonNegativeReals)  # x_{ij}: 施設iから顧客jへの供給量
    
    # 目的関数
    model.cost = Objective(
        expr=sum(facility_cost[i]*model.y[i] for i in model.I) +
             sum(transport_cost[i,j]*model.x[i,j] for i in model.I for j in model.J),
        sense=minimize
    )
    
    # 容量制約
    def capacity_rule(model, i):
        return sum(model.x[i,j] for j in model.J) <= facility_capacity[i]*model.y[i]
    model.capacity = Constraint(model.I, rule=capacity_rule)
    
    # 需要制約
    def demand_rule(model, j):
        return sum(model.x[i,j] for i in model.I) == customer_demand[j]
    model.demand = Constraint(model.J, rule=demand_rule)
    
    # 求解
    print("ソルバー実行中...")
    solver = SolverFactory('appsi_highs')
    result = solver.solve(model, tee=True)  # tee=Trueでログ表示
    
    # 結果の表示
    print("\n=== 結果 ===")
    print(f"状態: {result.solver.status}")
    print(f"最小総コスト: {value(model.cost):.1f}")
    
    print("\n建設される施設:")
    for i in model.I:
        if value(model.y[i]) > 0.5:
            print(f"  施設{i}: 建設する (コスト: {facility_cost[i]})")
        else:
            print(f"  施設{i}: 建設しない")
    
    print("\n輸送計画 (非ゼロのみ):")
    for i in model.I:
        for j in model.J:
            transport_amount = value(model.x[i,j])
            if transport_amount > 1e-8:
                print(f"  施設{i} -> 顧客{j}: {transport_amount:.2f} (単価: {transport_cost[i,j]})")
    
    # コストの内訳
    facility_cost_total = sum(facility_cost[i]*value(model.y[i]) for i in model.I)
    transport_cost_total = sum(transport_cost[i,j]*value(model.x[i,j]) for i in model.I for j in model.J)
    print(f"\nコストの内訳:")
    print(f"  建設コスト: {facility_cost_total:.1f}")
    print(f"  輸送コスト: {transport_cost_total:.1f}")
    print(f"  総コスト: {facility_cost_total + transport_cost_total:.1f}")
    print()

def complex_facility_location():
    """
    より複雑な施設配置問題
    分枝限定法が実際に動作する例
    """
    print("=" * 70)
    print("4. 複雑な施設配置問題（分枝限定法の動作確認）")
    print("=" * 70)
    print("施設候補: 3箇所")
    print("顧客: 4箇所")
    print("目的: 分枝限定法のログを観察")
    print()
    
    # より複雑なデータ
    facility_cost = {1: 100, 2: 120, 3: 110}  # 3つの施設
    facility_capacity = {1: 25, 2: 30, 3: 35}
    customer_demand = {1: 12, 2: 18, 3: 15, 4: 20}  # 4つの顧客
    transport_cost = {
        (1, 1): 8, (1, 2): 6, (1, 3): 10, (1, 4): 12,
        (2, 1): 13, (2, 2): 7, (2, 3): 9, (2, 4): 14,
        (3, 1): 10, (3, 2): 11, (3, 3): 8, (3, 4): 9
    }
    
    # モデルの作成
    model = ConcreteModel()
    
    model.I = Set(initialize=facility_cost.keys())
    model.J = Set(initialize=customer_demand.keys())
    
    model.y = Var(model.I, domain=Binary)
    model.x = Var(model.I, model.J, domain=NonNegativeReals)
    
    model.cost = Objective(
        expr=sum(facility_cost[i]*model.y[i] for i in model.I) +
             sum(transport_cost[i,j]*model.x[i,j] for i in model.I for j in model.J),
        sense=minimize
    )
    
    def capacity_rule(model, i):
        return sum(model.x[i,j] for j in model.J) <= facility_capacity[i]*model.y[i]
    model.capacity = Constraint(model.I, rule=capacity_rule)
    
    def demand_rule(model, j):
        return sum(model.x[i,j] for i in model.I) == customer_demand[j]
    model.demand = Constraint(model.J, rule=demand_rule)
    
    print("分枝限定法のログ（HiGHSソルバー）:")
    print("=" * 60)
    
    # 求解（詳細ログ付き）
    solver = SolverFactory('appsi_highs')
    result = solver.solve(model, tee=True)
    
    print("=" * 60)
    print("\n=== 結果 ===")
    print(f"最適目的関数値: {value(model.cost):.1f}")
    
    print("\n建設される施設:")
    for i in model.I:
        if value(model.y[i]) > 0.5:
            print(f"  施設{i}: 建設する (コスト: {facility_cost[i]}, 容量: {facility_capacity[i]})")
        else:
            print(f"  施設{i}: 建設しない")
    
    print("\n輸送計画 (非ゼロのみ):")
    for i in model.I:
        for j in model.J:
            transport_amount = value(model.x[i,j])
            if transport_amount > 1e-8:
                print(f"  施設{i} -> 顧客{j}: {transport_amount:.2f}")
    
    # 容量使用状況
    print("\n容量使用状況:")
    for i in model.I:
        if value(model.y[i]) > 0.5:
            used_capacity = sum(value(model.x[i,j]) for j in model.J)
            print(f"  施設{i}: {used_capacity:.1f} / {facility_capacity[i]} (使用率: {used_capacity/facility_capacity[i]*100:.1f}%)")
    
    print()
    
    # ログの解説
    print("【MIPソルバーのログの読み方】")
    print("- Nodes: 分枝限定法で探索されたノード数")
    print("- B&B Tree: 分枝限定木の状態")
    print("- BestBound: LP緩和による下界（最小化問題では下界）")
    print("- BestSol: 見つかった最良の整数解")
    print("- Gap: (BestSol - BestBound) / BestSol × 100%")
    print("- Gap = 0%で最適解が証明される")
    print()

def facility_location_formulation():
    """
    施設配置問題の数式による定式化の説明
    """
    print("=" * 70)
    print("5. 施設配置問題の数式による定式化")
    print("=" * 70)
    print()
    
    print("【集合】")
    print("  I: 施設の候補地の集合")
    print("  J: 顧客の集合")
    print()
    
    print("【パラメータ】")
    print("  f_i: 施設 i の建設コスト")
    print("  c_{ij}: 顧客 j の需要を施設 i から満たす単位輸送コスト")
    print("  d_j: 顧客 j の需要量")
    print("  M_i: 施設 i の容量上限")
    print()
    
    print("【決定変数】")
    print("  y_i: 施設 i を建設するかどうか（1: 建設する, 0: 建設しない）")
    print("  x_{ij}: 顧客 j の需要を施設 i で満たす量")
    print()
    
    print("【目的関数】")
    print("  最小化: Σ_i f_i * y_i + Σ_i Σ_j c_{ij} * x_{ij}")
    print()
    
    print("【制約条件】")
    print("  1. 容量制約: Σ_j x_{ij} ≤ M_i * y_i  ∀i ∈ I")
    print("  2. 需要充足: Σ_i x_{ij} = d_j  ∀j ∈ J")
    print("  3. 非負制約: x_{ij} ≥ 0  ∀i ∈ I, j ∈ J")
    print("  4. 0-1変数制約: y_i ∈ {0, 1}  ∀i ∈ I")
    print()

# ==============================================================================
# メイン実行関数
# ==============================================================================

def main():
    """メイン実行関数"""
    print("=" * 70)
    print("第3回セッション - 整数計画法と施設配置問題（完全版）")
    print("=" * 70)
    print()
    
    # 1. 分枝限定法の理論
    branch_and_bound_concept()
    
    print("\n" + "-" * 70 + "\n")
    
    # 2. 分枝限定法のステップバイステップ実行
    branch_and_bound_example()
    
    print("\n" + "-" * 70 + "\n")
    
    # 3. 基本的な施設配置問題
    simple_facility_location()
    
    print("\n" + "-" * 70 + "\n")
    
    # 4. 複雑な施設配置問題
    complex_facility_location()
    
    print("\n" + "-" * 70 + "\n")
    
    # 5. 数式による定式化
    facility_location_formulation()
    
    print("=" * 70)
    print("すべての実行が完了しました！")
    print()
    print("【学習内容のまとめ】")
    print("1. 分枝限定法の動作原理を理解")
    print("2. 混合整数計画問題の定式化方法を習得")
    print("3. 0-1変数を使った施設配置問題の実装")
    print("4. MIPソルバーのログの読み方を学習")
    print("5. 整数制約が問題の複雑性に与える影響を確認")
    print("=" * 70)

if __name__ == "__main__":
    main()