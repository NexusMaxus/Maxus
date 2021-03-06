from __future__ import division
from pyomo.environ import *

m = AbstractModel()

m.I= Param(within=NonNegativeIntegers)
m.T = Param(within=NonNegativeIntegers)
m.Y = Param(within=NonNegativeIntegers)

m.i=RangeSet(1,m.I)
m.t=RangeSet(1,m.T)
m.y=RangeSet(1,m.Y)
m.c=Set()
m.g=Set()

m.t_seed=Param(m.c,m.g,m.t)
m.t_harvest=Param(m.c,m.g,m.t)
m.L_req_Fg=Param(m.i,m.c,m.g)
m.L_Agr=Param(m.i)

m.P=Param(m.c)
m.C_Fg=Param(m.c)
m.GS=Param(m.c, m.g, m.t)
m.t_seedmonth=Param(m.c,m.g)

m.F_har=Var(m.i,m.c,m.g,m.y,m.t,domain=NonNegativeReals)
m.Fg = Var(m.i,m.c,m.g,m.y,m.t, domain=NonNegativeReals)

def obj_expression(m):
    return sum(sum(sum(sum(sum(m.P[c] * m.F_har[i,c,g,y,t]- m.C_Fg[c] * m.Fg[i,c,g,y,t] for i in m.i)for c in m.c)for g in m.g) for y in m.y)for t in m.t)

m.OBJ = Objective(rule=obj_expression, sense=maximize)

def harvest_constraint(m, i, c, g, y, t):
    if y==m.y.first():
        return m.F_har[i,c,g,y,t] == sum(m.Fg[i, c, g, y, x] for x in range(1, m.t_seed[c, g, t])) * m.t_harvest[c,g,t]
    return m.F_har[i,c,g,y,t] == (sum(m.Fg[i, c, g, y-1, x] for x in range(m.t_seed[c, g, t], 13)) + sum(m.Fg[i, c, g, y, x] for x in range(1, m.t_seed[c, g, t])))\
           * m.t_harvest[c,g,t]

        #m.F_har[i,c,g,y,t] == sum(m.Fg[i, c, g, y, x] for x in range(m.t_seed[c, g, t], t+1)) * m.t_harvest[c,g,t]
        #(m.t_seed[c, g, t] - t)
def land_constraint(m,i,y,t):
    return sum(sum(m.Fg[i,c,g,y,t]* m.L_req_Fg[i,c,g] for c in m.c) for g in m.g ) <= m.L_Agr[i]
def growthseason_constraint(m,i,c,g,y, t):
    return m.Fg[i,c,g,y,t] <= m.GS[c,g,t]*m.Fg[i,c,g,y,m.t_seedmonth[c,g]]


m.harvestConstraint = Constraint(m.i,m.c,m.g,m.y,m.t,rule=harvest_constraint)
m.landConstraint = Constraint(m.i,m.y, m.t, rule=land_constraint)
m.growthseasonConstraint = Constraint(m.i,m.c,m.g,m.y,m.t,rule=growthseason_constraint)
