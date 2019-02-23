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

m.Link_pos_b=Param(m.b,m.b,within=NonNegativeIntegers)



m.Con_pos = Param(within=NonNegativeIntegers)

#parameters: cost

m.C_bron = Param(within=NonNegativeReals)
m.C_e = Param(within=NonNegativeReals)
m.C_h = Param(m.w, m.h, within=NonNegativeReals)
m.C_b = Param(within=NonNegativeReals)



#variables
m.Const_b= Var(m.b, domain=NonNegativeIntegers)
m.A = Var(m.b, domain=NonNegativeReals)
m.Q = Var(m.b, domain=NonNegativeReals)
m.PK_bron = Var(domain=NonNegativeReals)
m.E_use = Var(m.h, domain = NonNegativeReals)
m.Type = Var(m.w, m.h, domain = NonNegativeIntegers, bounds=(-1,2))
m.Q_h = Var(m.h, domain=NonNegativeReals)
m.Q_source = Var( domain=NonNegativeReals)
m.Link_b=Var(m.b,m.b,domain=NonNegativeIntegers)


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

def Q_source_Con(m):
    return m.Q[5] * (m.T_in - m.T_source_in) == \
                     m.Q_source * (m.T_source_in-m.T_source_out)

def pipes_connection_Con(m,b):
    if (b > m.H) and (b < m.B):
        return sum(m.Q[b2]* m.Link_b[b2,b] for b2 in m.b) == sum(m.Q[b]*m.Link_b[b,b2] for b2 in m.b)
    elif b==m.B:
        return m.Q[b] == sum(m.Q[b2]* m.Link_b[b2,b] for b2 in m.b)
    elif b <= m.H:
        return sum(m.Q[b]* m.Link_b[b,b2] for b2 in m.b) == m.Q_h[b]

def pipe_sum_con(m,b):
    if b < m.B:
        return sum(m.Link_b[b,b2] for b2 in m.b) <= 1
    else:
        return Constraint.Skip

def pipes_construction_Con(m,b,b2):
    return m.Link_b[b2,b] <= m.Link_pos_b[b2,b]

def water_balance_con(m):
    return m.Q[m.B] == sum(m.Q_h[h] for h in m.h)

def one_direction_con(m,b):
    if b < m.B:
        return prod((1-m.Link_b[b,b2]) for b2 in m.b)  == 0;
    else:
        return Constraint.Skip

def one_direction_con2(m,b,b2):
    if (b < m.B) and (b != b2):
        return m.Link_b[b,b2] * m.Link_b[b2,b]  == 0;
    else:
        return Constraint.Skip

def Source_Con(m): #12
    return m.PK_bron == m.Q_source * (m.ro * m.sw * (m.T_source_in - m.T_source_out))



#initialize constraints

m.EnergyHouseConstraint = Constraint( m.h, rule=Energy_Use_House_Con) #1
m.DischargeConstraint = Constraint( m.b, rule=Discharge_Con) #3
m.PowerConstraint = Constraint( m.h, rule=Power_Con) #4
m.TypeConstraint = Constraint( m.h, rule=Type_Con) #5

m.pipesumConstraint = Constraint(m.b, rule=pipe_sum_con)
m.CostEnergyConstraint = Constraint( rule=CostEnergy_Con) #2
m.CostTubesConstraint = Constraint(rule=CostTubes_Con) #6
m.CostHousesConstraint = Constraint(rule=CostHouses_Con) #9

m.onedirectionConstraint = Constraint(m.b,rule= one_direction_con)
m.onedirectionConstraint2 = Constraint(m.b,m.b,rule= one_direction_con2)
m.waterbalanceConstraint = Constraint(rule = water_balance_con)
m.PipesConstructionConstraint= Constraint(m.b,m.b, rule=pipes_construction_Con)
m.PipesConstraint = Constraint(m.b, rule=pipes_connection_Con)
m.Q_sourceConstraint = Constraint(rule=Q_source_Con)
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

