# import pyomo.environ as pyo
# import pandas as pd
#
# nodes = [1, 2, 3]
# flowcost = pd.DataFrame(data=[[0, 1.4, 2.7], [0, 0, 0], [0, 1.6, 0]], index=nodes, columns=nodes)
# demand = pd.DataFrame(data=[0, 1, 1], index=nodes)
# supply =
#
# model = pyo.AbstractModel()
# model.Nodes = pyo.Set(initialize=nodes)
# model.Arcs = pyo.Set(dimen=2)
#
#
# def NodesOut_init(m, node):
#     for i, j in m.Arcs:
#         if i == node:
#             yield j
#
#
# def NodesIn_init(m, node):
#     for i, j in m.Arcs:
#         if j == node:
#             yield i
#
#
# model.NodesIn = pyo.Set(model.Nodes, initialize=NodesIn_init)
# model.NodesOut = pyo.Set(model.Nodes, initialize=NodesOut_init)
#
#
# model.Flow = pyo.Var(model.Arcs, domain=pyo.NonNegativeReals)
# model.FlowCost = pyo.Param(model.Arcs, initialize=flowcost.stack().to_dict())
#
# model.Demand = pyo.Param(model.Nodes, initialize=demand.to_dict())
# model.Supply = pyo.Param(model.Nodes)
#
#
# def Obj_rule(m):
#     return pyo.summation(m.FlowCost, m.Flow)
#
# model.Obj = pyo.Objective(rule=Obj_rule, sense=pyo.minimize)
#
#
# def FlowBalance_rule(m, node):
#     return m.Supply[node] \
#     + sum(m.Flow[i, node] for i in m.NodesIn[node]) \
#     - m.Demand[node] \
#     - sum(m.Flow[node, j] for j in m.NodesOut[node]) \
#     == 0
#
#
# model.FlowBalance = pyo.Constraint(model.Nodes, rule=FlowBalance_rule)