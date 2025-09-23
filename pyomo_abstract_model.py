import pyomo.environ as pyo

model = pyo.AbstractModel()

model.I = pyo.Set()  # 制約のインデックス
model.J = pyo.Set()  # 変数のインデックス

model.a = pyo.Param(model.I, model.J)  # 制約係数行列
model.b = pyo.Param(model.I)           # 制約右辺
model.c = pyo.Param(model.J)           # 目的関数係数

model.x = pyo.Var(model.J, domain=pyo.NonNegativeReals)

def obj_expression(m):
    return pyo.summation(m.c, m.x)

model.OBJ = pyo.Objective(rule=obj_expression, sense=pyo.maximize)

def ax_constraint_rule(m, i):
    return sum(m.a[i,j]*m.x[j] for j in m.J) >= m.b[i]

model.AxbConstraint = pyo.Constraint(model.I, rule=ax_constraint_rule)