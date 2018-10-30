from __future__ import division
from pyomo.environ import *
from pyomo.opt import SolverFactory

m = ConcreteModel()

m.x3 = Var(within=NonNegativeReals)
m.u = Var(within=NonNegativeReals)


def _con(m):
    return m.x3 >= 3


m.con = Constraint(rule=_con)


def _con2(m):
    return 4 >= m.u >= 1


m.con2 = Constraint(rule=_con2)

m.obj = Objective(expr=m.x3 * m.u)
opt = SolverFactory('Ipopt')
results = opt.solve(m)
m.solutions.store_to(results)
results.write()