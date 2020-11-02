from __future__ import division
import geopandas as gpd
import numpy as np
import pandas as pd
from pyomo.environ import *
from pyomo.opt import SolverFactory, TerminationCondition
from pyomo.network import Port, Arc
import pyomo.kernel as pmo
import random


# CP = 5
house_or_not = [1, 1, 1, 1, 0]
source_or_not = [0, 0, 0, 0, 1]
p2p = [[1, 2], [0, 2], [0, 1, 3], [2, 4], [3]]
cost_street = [[1.5, 3.5], [1.5, 2.5], [3.5, 2.5, 4.5], [4.5, 5.5], [5.5]]


# house_or_not = [1, 1, 0]
# source_or_not = [0, 0, 1]
# p2p = [[1, 2], [0, 2], [0, 1]]
# cost_street = [[1.5, 3.5], [1.5, 2.5], [3.5, 2.5]]


m = AbstractModel()
m.x = RangeSet(0, 4)


def junction(b, x):
    b.h = Param(initialize=house_or_not[x])
    b.s = Param(initialize=source_or_not[x])

    b.i = Set(initialize=p2p[x])
    b.Q_to = Var(b.i, domain=pmo.Binary, initialize=dict(zip(b.i, np.zeros(len(b.i)))))
    b.Q_from = Var(b.i, domain=pmo.Binary, initialize=dict(zip(b.i, np.zeros(len(b.i)))))
    b.total_Q = Var(domain=pmo.NonNegativeReals, initialize=0)

    b.cost_street = Param(b.i, initialize=dict(zip(b.i, cost_street[x])), domain=pmo.NonNegativeReals)

    def junction_balance_rule(b):
        return (sum(b.Q_from[i] for i in b.i) + b.h) * (1 - sum(b.Q_to[i] for i in b.i)) == b.s

    def junction_streets_rule(b):
        return b.total_Q == sum(b.Q_to[i] * b.cost_street[i] for i in b.i)

    b.junction_balance = Constraint(rule=junction_balance_rule)
    b.junction_streets = Constraint(rule=junction_streets_rule)


m.p = Block(m.x, rule=junction)


def link_rule(m, x, x2):
    if x2 in m.p[x].i:
        return m.p[x].Q_to[x2] == m.p[x2].Q_from[x]
    else:
        return Constraint.Skip


def opp_directions_rule(m, x, x2):
    if (x2 in m.p[x].i) and (x < x2):
        return m.p[x].Q_to[x2] * m.p[x2].Q_to[x] == 0
    else:
        return Constraint.Skip


m.linking_points = Constraint(m.x, m.x, rule=link_rule)
m.opposite_directions = Constraint(m.x, m.x, rule=opp_directions_rule)


def obj_rule(m):
    return sum(m.p[x].total_Q for x in m.x)


m.obj = Objective(rule=obj_rule, sense=minimize)

opt = SolverFactory('ipopt')
instance = m.create_instance()
instance.pprint()
results = opt.solve(instance,  tee=True)
# results = opt.solve(instance, mip_solver='glpk', nlp_solver='ipopt', strategy='OA', tee=True)
instance.display(filename='results_12_block_simple.txt')