from __future__ import division
import geopandas as gpd
import numpy as np
import pandas as pd
from pyomo.environ import *
from pyomo.opt import SolverFactory, TerminationCondition
from pyomo.network import Port, Arc
import pyomo.kernel as pmo
import random

# def step_func(x):
#     delta = 0.1
#     z = ((2 + pmo.exp(-50 * (x - delta))) / (1 + pmo.exp(-50 * (x - delta)))) - 1
#     return z


T_in = 60
T_out = 40
ro = 1
sw = 4.18

# CP = 5
p2p = [[1, 2], [0, 2], [0, 1, 3], [2, 4], [3]]
p2h = [[0],[1, 2],[],[],[]]
cost_street = [[1.5, 3.5], [1.5, 2.5], [3.5, 2.5, 4.5], [4.5, 5.5], [5.5]]

house_or_not = [True, True, False, False, False]
source_or_not = [False, False, False, False, True]
wv = [200, 200, 300]
pkhuis = [50, 50, 50]
p_h = [30, 35, 35]
# p2p = [[1, 2], [0, 2], [0, 1]]
# cost_street = [[1.5, 3.5], [1.5, 2.5], [3.5, 2.5]]


m = AbstractModel()
m.x = RangeSet(0, 4)
m.h = RangeSet(0, 2)
# m.streets = Set(dimen=2)

m.P_grid = Param(domain=pmo.NonNegativeReals, initialize=30)


def junction(b, x):

    b.i = Set(initialize=p2p[x])
    b.c = Var(b.i, domain=pmo.Boolean)
    b.Q_to = Var(b.i, domain=pmo.NonNegativeReals, initialize=dict(zip(b.i, np.zeros(len(b.i)))))
    b.street_open = Var(b.i, domain=pmo.Binary, initialize=dict(zip(b.i, np.zeros(len(b.i)))))
    b.Q_from = Var(b.i, domain=pmo.NonNegativeReals, initialize=dict(zip(b.i, np.zeros(len(b.i)))))
    b.street_cost = Var(domain=pmo.NonNegativeReals, initialize=0)
    b.h = Param(domain=pmo.Boolean, initialize=house_or_not[x])
    b.s = Param(domain=pmo.Boolean, initialize=source_or_not[x])

    if b.h:
        b.Q_from_h = Var(domain=pmo.NonNegativeReals, initialize=0)
        b.IDh = Set(initialize=p2h[x])
    if b.s:
        b.Q_to_s = Var(domain=pmo.NonNegativeReals, initialize=0)

    b.cost_street = Param(b.i, initialize=dict(zip(b.i, cost_street[x])), domain=pmo.NonNegativeReals)

    def junction_balance_rule(b):
        if b.h:
            return (sum(b.Q_from[i] for i in b.i) + b.Q_from_h) == sum(b.Q_to[i] for i in b.i)
        elif b.s:
            return sum(b.Q_from[i] for i in b.i) == b.Q_to_s
        else:
            return sum(b.Q_from[i] for i in b.i) == sum(b.Q_to[i] for i in b.i)

    def convergence_rule(b):
        if b.s:
            return Constraint.Skip
        else:
            return sum(b.c[i] * b.Q_to[i] for i in b.i) == sum(b.Q_to[i] for i in b.i)

    def choice_rule(b):
        return prod(1 - b.c[i] for i in b.i) == 0

    def choice_sum_rule(b):
        return sum(b.c[i] for i in b.i) == 1

    def junction_streets_rule(b):
        return b.street_cost == sum(b.street_open[i] * b.cost_street[i] for i in b.i)

    def street_open_rule(b, i):
        return (1 - b.street_open[i]) * b.Q_to[i] == 0

    b.junction_balance = Constraint(rule=junction_balance_rule)
    b.junction_streets = Constraint(rule=junction_streets_rule)
    b.street_open_con = Constraint(b.i, rule=street_open_rule)
    b.convergence_con = Constraint(rule=convergence_rule)
    b.choice_con = Constraint(rule=choice_rule)
    b.choice_sum_con = Constraint(rule=choice_sum_rule)

def house(b, h):
    b.T_in = Param(domain=pmo.NonNegativeReals, initialize=T_in)
    b.T_out = Param(domain=pmo.NonNegativeReals, initialize=T_out)
    b.ro = Param(within=pmo.NonNegativeReals, initialize=ro)  # dichtheid | density
    b.sw = Param(within=pmo.NonNegativeReals, initialize=sw)
    b.PKhuis = Param(domain=pmo.NonNegativeReals, initialize=pkhuis[h])
    b.WV = Param(domain=pmo.NonNegativeReals, initialize=wv[h])
    b.P_h = Param(domain=pmo.NonNegativeReals, initialize=p_h[h])

    b.Q_h = Var()

    def Power_Con(b):
        return b.Q_h * (b.ro * b.sw * (b.T_in - b.T_out)) >= \
               b.PKhuis

    b.PowerConstraint = Constraint(rule=Power_Con)


m.p = Block(m.x, rule=junction)
m.d = Block(m.h, rule=house)


def link_rule(m, x, x2):
    if x2 in m.p[x].i:
        return m.p[x].Q_to[x2] == m.p[x2].Q_from[x]
    else:
        return Constraint.Skip
m.linking_points = Constraint(m.x, m.x, rule=link_rule)


def Q_initial(m, x):
    if m.p[x].h:
        return sum(m.d[h].Q_h for h in m.p[x].IDh) == m.p[x].Q_from_h
    else:
        return Constraint.Skip
m.house_to_point = Constraint(m.x, rule=Q_initial)

def opp_directions_rule(m, x, x2):
    if (x2 in m.p[x].i) and (x < x2):
        return m.p[x].Q_to[x2] * m.p[x2].Q_to[x] == 0
    else:
        return Constraint.Skip
m.opposite_directions = Constraint(m.x, m.x, rule=opp_directions_rule)


def obj_rule(m):
    return sum(m.P_grid * m.d[h].WV for h in m.h) - sum(m.p[x].street_cost for x in m.x)
m.obj = Objective(rule=obj_rule, sense=maximize)


# opt = SolverFactory('baron', executable="/home/rogier/PycharmProjects/solvers/baron-lin64/baron")
# opt = SolverFactory('mindtpy')
opt =  SolverFactory('ipopt')
instance = m.create_instance()
results = opt.solve(instance,  tee=True)
# results = opt.solve(instance, mip_solver='glpk', nlp_solver='ipopt', strategy='OA', tee=True)

instance.solutions.store_to(results)
instance.pprint()
instance.display()

# results.write(filename='results_12_block_simple.json', format='json')
