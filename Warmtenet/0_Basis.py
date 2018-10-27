from __future__ import division
from pyomo.environ import *

m = AbstractModel()

m.B = Param(within=NonNegativeIntegers)
m.H = Param(within=NonNegativeIntegers)

#sets
m.b = RangeSet(1,m.B)
m.h = RangeSet(1,m.H)
m.w = Set()

#parameters
m.WV = Param(m.w,m.h, within=NonNegativeReals)
m.PKhuis = Param(m.w,m.h, within=NonNegativeReals)
m.T_in = Param(within=NonNegativeReals)
m.Ef = Param(within=NonNegativeReals) #energyuse per kWh heat)
m.U = Param(within=NonNegativeReals) #stroomsnelheid | flow velocity
m.ro = Param(within=NonNegativeReals) #dichtheid | density
m.sw = Param(within=NonNegativeReals) # ...  | soortelijke warmte

#parameters: cost

m.C_bron = Param(within=NonNegativeReals)
m.C_e = Param(within=NonNegativeReals)
m.C_h = Param(m.w, within=NonNegativeReals)
m.C_b = Param(within=NonNegativeReals)

#variables
m.A = Var(m.b, domain=NonNegativeReals)
m.Q = Var(m.b, domain=NonNegativeReals)
m.T_uit = Var(m.h, domain=NonNegativeReals)
#m.PK_bron = Var(domain=NonNegativeReals)
m.E_use = Var(m.h, domain = NonNegativeReals)
m.Type = Var(m.w, m.h, domain = NonNegativeIntegers)

#variables: cost
m.CostTubes = Var(domain=NonNegativeReals)
m.CostEnergy = Var(domain=NonNegativeReals)
m.CostHouses = Var(domain=NonNegativeReals)
m.CostSingleHouse = Var(domain=NonNegativeReals)

def obj_expression(m):
    return summation(m.CostTubes, m.CostEnergy, m.CostHouses)

m.OBJ = Objective(rule=obj_expression, sense=minimize)



#Constraints

def Energy_Use_House(m, h): #1
    for w in m.w:
        if m.Type[w,h] == 1:
           return m.E_use[h] == m.WV[w,h] * m.Ef

def Energy_Use_Tot(m):  #2
    return Energy_Use_Tot == sum(m.E_use[h] for h in m.h)

def Discharge_Con(m, b): #3
    return m.A[b] == m.Q[b]/m.U

def Power_Con(m,b,h):  #4
    for w in m.w:
        if m.Type[w, h] == 1:
            return m.Q[b] == m.PKhuis[w,h]/( m.ro * m.sw * (m.T_in - m.T_uit[h]))

def Type_Con(m,h):  #5
    return sum(m.Type[w,h] for w in m.w) == 1

#cost
def CostTubes_Con(m):  #6
    return m.CostTubes == sum(m.A[b] for b in m.b) * m.C_b

def CostEnergy_Con(m):   #7
    return m.CostEnergy == m.Energy_Use_Tot * m.C_e

def CostSingleHouse_Con(m,h):   #8
    for w in m.w:
        if m.Type[w,h] == 1:
            return m.CostSingleHouse == m.C_h[w]

def CostHouses_Con(m): #9
    return m.CostHouses == sum(m.CostSingleHouse[h] for h in m.h)



#initialize constraints

m.EnergyHouseConstraint = Constraint(m, m.h, rule=Energy_Use_House) #1
m.EnergyTotalConstraint = Constraint(m, rule = Energy_Use_Tot) #2
m.DischargeConstraint = Constraint(m, m.b, rule= Discharge_Con) #3
m.PowerConstraint = Constraint(m,m.b,m.h, rule=Power_Con) #4
m.TypeConstraint = Constraint(m, m.h, rule=Type_Con) #5

m.CostTubesConstraint = Constraint(m, rule =CostTubes_Con) #6
m.CostEnergyConstraint = Constraint(m, rule=CostEnergy_Con) #7
m.CostSingleHouseConstraint = Constraint(m, m.h, rule=CostSingleHouse_Con) #8
m.CostHousesConstraint = Constraint(m, rule=CostSingleHouse_Con) #9



























