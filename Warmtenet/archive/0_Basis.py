from __future__ import division
from pyomo.environ import *
from pyomo.opt import SolverFactory

m = AbstractModel()


m.B = Param(within=NonNegativeIntegers)
m.H = Param(within=NonNegativeIntegers)

#sets
m.b = RangeSet(1,m.B)
m.h = RangeSet(1,m.H)
m.w = Set()

#parameters
m.WV = Param(m.w, m.h, within=NonNegativeReals)
m.PKhuis = Param(m.w, m.h, within=NonNegativeReals)

m.T_in = Param(within=NonNegativeReals)
m.T_source_in = Param(within=NonNegativeReals)
m.T_out = Param(m.h, within=NonNegativeReals)
m.T_source_out = Param(within=NonNegativeReals)

m.Ef = Param(within=NonNegativeReals) #energyuse pump per kWh heat)
m.U = Param(within=NonNegativeReals) #stroomsnelheid | flow velocity
m.ro = Param(within=NonNegativeReals) #dichtheid | density
m.sw = Param(within=NonNegativeReals) # ...  | soortelijke warmte



m.Con_pos = Param(within=NonNegativeIntegers)

#parameters: cost

m.C_bron = Param(within=NonNegativeReals)
m.C_e = Param(within=NonNegativeReals)
m.C_h = Param(m.w, m.h, within=NonNegativeReals)
m.C_b = Param(within=NonNegativeReals)



m.load('0_basis.dat')

#variables
m.Const_b= Var(m.b, domain=NonNegativeIntegers)
m.A = Var(m.b, domain=NonNegativeReals)
m.Q = Var(m.b, domain=NonNegativeReals)
m.PK_bron = Var(domain=NonNegativeReals)
m.E_use = Var(m.h, domain = NonNegativeReals)
m.Type = Var(m.w, m.h, domain = NonNegativeIntegers, bounds=(-1,2))
m.Q_h = Var(m.h, domain=NonNegativeReals)
m.Q_source = Var( domain=NonNegativeReals)

#variables: cost
m.CostTubes = Var(domain=NonNegativeReals)
m.CostEnergy = Var(domain=NonNegativeReals)
m.CostHouses = Var(domain=NonNegativeReals)
m.CostSingleHouse = Var(m.h, domain=NonNegativeReals)
m.CostSource = Var(domain=NonNegativeReals)

def obj_expression(m):
    return m.CostTubes + m.CostEnergy + m.CostHouses + m.CostSource

m.OBJ = Objective(rule=obj_expression, sense=minimize)



#cost
def CostEnergy_Con(m):  #2
    return m.CostEnergy == sum(m.E_use[h] for h in m.h) * m.C_e

def CostTubes_Con(m):  #6
    return m.CostTubes == sum(m.A[b] for b in m.b) * m.C_b

def CostHouses_Con(m):   #8
    return m.CostHouses == sum(sum(m.Type[w,h] * m.C_h[w,h] for w in m.w)for h in m.h)

def Cost_Source_Con(m): #13
    return m.CostSource == m.PK_bron * m.C_bron



#Constraints

def Energy_Use_House_Con(m, h): #1
    return m.E_use[h] ==  sum(m.Type[w,h]* m.WV[w,h] * m.Ef for w in m.w)

def Discharge_Con(m, b): #3
    return m.A[b]*m.U == m.Q[b]

def Power_Con(m,h):  #4
    return m.Q_h[h] * (m.ro * m.sw * (m.T_in - m.T_out[h])) ==  \
           sum(m.Type[w,h] * m.PKhuis[w,h] for w in m.w)

def Type_Con(m,h):  #5
    return sum(m.Type[w,h] for w in m.w) == 1

def Q_Con(m,h):  #11
    return m.Q[h] == m.Q_h[h]

def Q_source_Con(m):
    return sum(m.Q[b] for b in m.b) * (m.T_in - m.T_source_in) == \
                     m.Q_source * (m.T_source_in-m.T_source_out)


def Source_Con(m): #12
    return m.PK_bron == m.Q_source * (m.ro * m.sw * (m.T_source_in - m.T_source_out))



#initialize constraints

m.EnergyHouseConstraint = Constraint( m.h, rule=Energy_Use_House_Con) #1
m.DischargeConstraint = Constraint( m.b, rule=Discharge_Con) #3
m.PowerConstraint = Constraint( m.h, rule=Power_Con) #4
m.TypeConstraint = Constraint( m.h, rule=Type_Con) #5


m.CostEnergyConstraint = Constraint( rule=CostEnergy_Con) #2
m.CostTubesConstraint = Constraint(rule=CostTubes_Con) #6
m.CostHousesConstraint = Constraint(rule=CostHouses_Con) #9

m.Q_sourceConstraint = Constraint(rule=Q_source_Con)
m.Q_Constraint = Constraint(m.h, rule = Q_Con)
m.SourceConstrant = Constraint(rule=Source_Con)
m.CostSourceConstraint = Constraint(rule=Cost_Source_Con)

if __name__ == '__main__':
    # This replicates what the pyomo command-line tools does
    from pyomo.opt import SolverFactory
    import pyomo.environ


    opt = SolverFactory('Ipopt')
    instance= m.create_instance()
    results = opt.solve(instance)
    m.solutions.store_to(results)
    results.write()



























