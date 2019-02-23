from __future__ import division
from pyomo.environ import *
from pyomo.opt import SolverFactory

m = AbstractModel()


m.H = Param(within=NonNegativeIntegers)
m.X = Param(within=NonNegativeIntegers)

#sets
m.h = RangeSet(1,m.H)
m.x = RangeSet(1,m.X)

#parameters
m.P_h = Param( m.h, within=NonNegativeReals )
m.WV = Param(m.h, within=NonNegativeReals)
m.PKhuis = Param( m.h, within=NonNegativeReals)

m.T_in = Param(within=NonNegativeReals)
m.T_source_in = Param(within=NonNegativeReals)
m.T_out = Param(m.h, within=NonNegativeReals)
m.T_source_out = Param(within=NonNegativeReals)

m.Ef = Param(within=NonNegativeReals) #energyuse pump per kWh heat)
m.U = Param(within=NonNegativeReals) #stroomsnelheid | flow velocity
m.ro = Param(within=NonNegativeReals) #dichtheid | density
m.sw = Param(within=NonNegativeReals) # ...  | soortelijke warmte

m.Q_poss=Param(m.x,m.x,within=NonNegativeIntegers)

#parameters: cost

m.C_bron = Param(within=NonNegativeReals)
m.C_e = Param(within=NonNegativeReals)
m.C_h = Param( m.h, within=NonNegativeReals)
m.C_b = Param(within=NonNegativeReals)
m.C_Street = Param(m.x, m.x, within=NonNegativeReals)

m.load('8_1temperatuur.dat')

#variables
m.A = Var(m.x, m.x, domain=NonNegativeReals)
m.Q = Var(m.x, m.x, domain=NonNegativeReals)
m.PK_bron = Var(domain=NonNegativeReals)
m.E_use = Var(m.h, domain = NonNegativeReals)
m.Conn = Var(m.h, domain = NonNegativeIntegers, bounds=(0,1))
m.Q_h = Var(m.h, domain=NonNegativeReals)
m.Q_source = Var( domain=NonNegativeReals)
m.P_grid = Var( domain = NonNegativeReals)


#variables: cost
m.CostTubes = Var(domain=NonNegativeReals)
m.CostEnergy = Var(domain=NonNegativeReals)
m.Revenue = Var(domain=Reals)
m.CostSource = Var(domain=NonNegativeReals)
m.CostStreets = Var(domain=NonNegativeReals)

def obj_expression(m):
    return m.Revenue - (m.CostTubes + m.CostEnergy + m.CostSource + m.CostStreets)

m.OBJ = Objective(rule=obj_expression, sense=maximize)


def Revenue_Con(m):
    return m.Revenue == sum(m.P_grid *m.WV[h] * m.Conn[h] for h in m.h)

#cost
def CostEnergy_Con(m):  #2
    return m.CostEnergy == sum(m.E_use[h] for h in m.h) * m.C_e

def CostTubes_Con(m):  #6
    return m.CostTubes == sum(sum(m.A[x,x2] for x in m.x)for x2 in m.x) * m.C_b

def Cost_Source_Con(m): #13
    return m.CostSource == m.PK_bron * m.C_bron

def Cost_Streets_Con(m):
    return m.CostStreets == sum(sum((m.Q[x,x2]*150)**2/((m.Q[x,x2]*150)**2+1) * m.C_Street[x,x2] for x in m.x)for x2 in m.x)



#Constraints
def Pgrid_Con(m,h):
    return  m.Conn[h] *  (m.P_grid - m.P_h[h])  <= 0


def Energy_Use_House_Con(m, h): #1
    return m.E_use[h] ==  m.WV[h] * m.Conn[h] * m.Ef

def Discharge_Con(m, x, x2): #3
    return m.A[x,x2]* m.U == m.Q[x,x2]

def Power_Con(m,h):  #4
    return m.Q_h[h] * (m.ro * m.sw * (m.T_in - m.T_out[h])) ==  \
           m.Conn[h] * m.PKhuis[h]

def Q_source_Con(m):
    return m.Q[m.X-1,m.X] * (m.T_in - m.T_source_in) == \
                     m.Q_source * (m.T_source_in-m.T_source_out)

def Q_mass_balance(m,x):
    if x <= m.H:
        return Constraint.Skip
    if (x > m.H) and (x<m.X):
        return sum(m.Q[x,x2]for x2 in m.x) == sum(m.Q[x2,x] for x2 in m.x)
    if x == m.X:
        return sum(m.Q_h[h] for h in m.h) == sum(m.Q[x2,x] for x2 in m.x)

def Q_initial(m,h):
    return sum(m.Q[h,x2] for x2 in m.x) == m.Q_h[h]


def one_direction_con(m,x,x2):
    if x!=x2:
        return m.Q[x,x2] * m.Q[x2,x]  == 0;
    else:
        return Constraint.Skip


def pipes_construction_Con(m,x,x2):
    return m.Q[x,x2] <= m.Q_poss[x,x2]

def Source_Con(m): #12
    return m.PK_bron == m.Q_source * (m.ro * m.sw * (m.T_source_in - m.T_source_out))




#initialize constraints
#cost
m.CostEnergyConstraint = Constraint( rule=CostEnergy_Con) #2
m.CostTubesConstraint = Constraint(rule=CostTubes_Con) #6
m.RevenueConstraint = Constraint(rule=Revenue_Con) #9
m.CostSourceConstraint = Constraint(rule=Cost_Source_Con)
m.CostStreetsConstraint = Constraint(rule=Cost_Streets_Con)
m.PgridConstraint = Constraint(m.h,rule=Pgrid_Con)

#house
m.EnergyHouseConstraint = Constraint( m.h, rule=Energy_Use_House_Con) #1
m.DischargeConstraint = Constraint( m.x, m.x, rule=Discharge_Con) #3
m.PowerConstraint = Constraint( m.h, rule=Power_Con) #4


#pipes
m.Q_sourceConstraint = Constraint(rule=Q_source_Con)
m.Q_massBalanceConstraint = Constraint( m.x, rule=Q_mass_balance)
m.onedirectionConstraint = Constraint(m.x,m.x,rule= one_direction_con)
m.PipesConstructionConstraint= Constraint(m.x,m.x, rule=pipes_construction_Con)
m.pipesInitialConstraint= Constraint(m.h, rule=Q_initial)

#Source
m.SourceConstrant = Constraint(rule=Source_Con)


if __name__ == '__main__':
    # This replicates what the pyomo command-line tools does
    from pyomo.opt import SolverFactory
    import pyomo.environ


    opt = SolverFactory('bonmin')
    # opt.options['linear_solver'] = 'ma57'
    instance = m.create_instance()
    results = opt.solve(instance, tee=True,  options={'tol': 1e-5, 'max_iter': 1000})
    instance.solutions.store_to(results)
    results.write(filename='results_1temperatuur.json', format='json')
    instance.pprint()