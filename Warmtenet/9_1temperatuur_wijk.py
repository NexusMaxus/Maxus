from __future__ import division
import geopandas as gpd
import numpy as np
import pandas as pd
from pyomo.environ import *
from pyomo.opt import SolverFactory, TerminationCondition
import random


def step_func(x):
    delta = 0.05
    z = ((2 + exp(-10 * (x - delta))) / (1 + exp(-10* (x - delta)))) - 1\
        + ((2 + exp(-10 * (-x - delta))) / (1 + exp(-10 * (-x - delta)))) - 1
    return z

def asymptot_func(x):
    # z = 1 / (((x + 0.05) * 10)** 2 + 0.05) + 1
    z = 1/((exp(10*x)-1)+1) +1
    return z


optimize=True


points = gpd.read_file(r'C:\Users\Rogier\OneDrive\Warmtenet\All_Points_copy.shp')
roads = gpd.read_file(r'C:\Users\Rogier\OneDrive\Warmtenet\assen_test\wegen_wijk.shp')
buildings = gpd.read_file(r'C:\Users\Rogier\OneDrive\Warmtenet\assen_test\shape\buildings.shp')


idx = points.index.tolist()
source = points.loc[points['osm_id'] == -1].index.values[0]
idx.pop(source)
points = points.reindex(idx+[source])
points.reset_index(drop = True)

PointsInRoads = np.zeros((len(roads.geometry),len(points.geometry)))
PointsInRoadsShort = np.ones((len(roads.geometry),2))*-1

i=0;
j=0;

for road in roads.geometry:
    x = 0;
    for point in points.geometry:
        if road.distance(point) < 1e-8:
            PointsInRoads[i,j] = 1
            PointsInRoadsShort[i,x] = j
            x += 1;
        j += 1
    i += 1
    j = 0

PointsConnected = np.zeros((len(points.geometry),len(points.geometry)))
length_roads = 0.01 * np.ones((len(points.geometry),len(points.geometry)))


for i in range(0, np.size(PointsInRoadsShort,0)):
    if np.logical_and(PointsInRoadsShort[i,0] != -1, PointsInRoadsShort[i,1] != -1):
        PointsConnected[int(PointsInRoadsShort[i,0]),int(PointsInRoadsShort[i,1])] = 1
        PointsConnected[int(PointsInRoadsShort[i, 1]), int(PointsInRoadsShort[i, 0])] = 1
        length_roads[int(PointsInRoadsShort[i, 1]), int(PointsInRoadsShort[i, 0])] = roads['length'][i]
        length_roads[int(PointsInRoadsShort[i,0]),int(PointsInRoadsShort[i,1])]= roads['length'][i]


StreetsConnected = np.zeros(np.size(PointsConnected,0))

for i in range(0, np.size(PointsConnected,0)):
    StreetsConnected[i] = sum(PointsConnected[i,:])




if optimize == True:

    #buildings
    X = len(points)-1
    H = len(buildings)-1

    P_h = pd.Series(data = 50 * np.ones(int(len(buildings))))#random.sample(np.arange(50,50.5,0.001).tolist(), int(len(buildings))))

    PKhuis = pd.Series(data = buildings['area']/ (500 * np.ones(int(len(buildings))))) #random.sample(np.arange(1000,1000.5,0.001).tolist(), int(len(buildings))))

    WV = pd.Series(data = buildings['area']/(10 * np.ones(int(len(buildings))))) #data = buildings['area'] / random.sample(np.arange(15,15.5,0.001).tolist(), int(len(buildings))))

    T_in = 60
    T_out = pd.Series(data=40 * np.ones(len(buildings)))

    T_source_in = 30
    T_source_out = 10
    Ef = 0.07
    U = 3
    ro = 1
    sw = 4.18
    C_bron = 20 #20
    C_e = 50 #50
    C_b = 100
    C_constant_source = 20000
    # C_street = pd.DataFrame(data=PointsConnected * 2)
    Q_poss = pd.DataFrame(data=PointsConnected * 999)
    length_roads_pd = pd.DataFrame(data=length_roads)

    del (points)
    del (roads)
    del (buildings)

    m = AbstractModel()


    m.H = Param(within=NonNegativeIntegers, initialize=H)
    m.X = Param(within=NonNegativeIntegers, initialize=X)

    #sets
    m.h = RangeSet(0,m.H)
    m.x = RangeSet(0,m.X)

    #parameters


    m.P_h = Param(m.h, within=NonNegativeReals, initialize=P_h.to_dict())
    m.WV = Param(m.h, within=NonNegativeReals, initialize=WV.to_dict())
    m.PKhuis = Param( m.h, within=NonNegativeReals, initialize = PKhuis.to_dict())


    m.T_in = Param(within=NonNegativeReals, initialize=T_in)
    m.T_source_in = Param(within=NonNegativeReals, initialize=T_source_in)
    m.T_out = Param(m.h, within=NonNegativeReals, initialize=T_out.to_dict())
    m.T_source_out = Param(within=NonNegativeReals, initialize=T_source_out)

    m.Ef = Param(within=NonNegativeReals, initialize=Ef)  # energyuse pump per kWh heat)
    m.U = Param(within=NonNegativeReals, initialize=U)  # stroomsnelheid | flow velocity
    m.ro = Param(within=NonNegativeReals, initialize=ro)  # dichtheid | density
    m.sw = Param(within=NonNegativeReals, initialize=sw)  # ...  | soortelijke warmte

    m.Q_poss = Param(m.x, m.x, within=NonNegativeReals, initialize=Q_poss.stack().to_dict())

    m.length = Param(m.x, m.x, within=NonNegativeReals, initialize=length_roads_pd.stack().to_dict())
    # parameters: cost
    m.C_constant_source = Param(within=NonNegativeReals, initialize=C_constant_source)
    m.C_bron = Param(within=NonNegativeReals, initialize=C_bron)
    m.C_e = Param(within=NonNegativeReals, initialize=C_e)
    m.C_b = Param(within=NonNegativeReals, initialize=C_b)
    # m.C_Street = Param(m.x, m.x, within=NonNegativeReals, initialize=C_street.stack().to_dict())


    #variables
    m.A = Var(m.x, m.x, domain=NonNegativeReals)
    m.Q = Var(m.x, m.x, domain=NonNegativeReals)
    m.PK_bron = Var(domain=NonNegativeReals)
    m.E_use = Var(m.h, domain = NonNegativeReals)
    m.Conn = Var(m.h, domain= NonNegativeIntegers, bounds=(0,1))
    m.Q_h = Var(m.h, domain=NonNegativeReals)
    m.Q_source = Var( domain=NonNegativeReals)
    m.P_grid = Var( domain = NonNegativeReals)


    #variables: cost
    m.CostTubes = Var(domain=NonNegativeReals)
    m.CostEnergy = Var(domain=NonNegativeReals)
    m.Revenue = Var(domain=Reals)
    m.CostSource = Var(domain=NonNegativeReals)
    # m.CostStreets = Var(domain=NonNegativeReals)


    def obj_expression(m):
        return m.Revenue - (m.CostTubes + m.CostEnergy + m.CostSource ) #- m.CostStreets


    m.OBJ = Objective(rule=obj_expression, sense=maximize)


    def Revenue_Con(m):
        return m.Revenue == sum(m.P_grid * m.WV[h] * m.Conn[h] for h in m.h)


    # cost
    def CostEnergy_Con(m):  # 2
        return m.CostEnergy == sum(m.E_use[h] for h in m.h) * m.C_e


    def CostTubes_Con(m):  # 6
        return m.CostTubes == sum(sum(asymptot_func(m.A[x, x2])* m.A[x,x2] * m.length[x,x2] for x in m.x) for x2 in m.x) * m.C_b


    def Cost_Source_Con(m):  # 13
        return m.CostSource == m.PK_bron * m.C_bron + m.C_constant_source * step_func(m.Q_source)


    # def Cost_Streets_Con(m):
    #     return m.CostStreets == sum(sum(step_func(m.A[x,x2]*30)* m.length[x,x2] * m.C_Street[x,x2] for x in m.x)for x2 in m.x)
    #         # sum(sum((m.Q[x, x2] * 100) ** 2 / ((m.Q[x, x2] * 100) ** 2 + 1) * m.C_Street[x, x2] for x in m.x) for x2 in m.x)


    # Constraints
    def Pgrid_Con(m, h):
        return m.Conn[h] * (m.P_grid - m.P_h[h]) <= 0


    def Energy_Use_House_Con(m, h):  # 1
        return m.E_use[h] == m.WV[h] * m.Conn[h] * m.Ef


    def Discharge_Con(m, x, x2):  # 3
        return m.A[x, x2] * m.U == m.Q[x, x2]


    def Power_Con(m, h):  # 4
        return m.Q_h[h] * (m.ro * m.sw * (m.T_in - m.T_out[h])) == \
               m.Conn[h] * m.PKhuis[h]


    def Q_source_Con(m):
        return m.Q[36, m.X] * (m.T_in - m.T_source_in) - \
               m.Q_source * (m.T_source_in - m.T_source_out) == 0


    def Q_mass_balance(m, x):
        if x <= m.H:
            return Constraint.Skip
        if (x > m.H) and (x < m.X):
            return sum(m.Q[x, x2] for x2 in m.x) == sum(m.Q[x2, x] for x2 in m.x)
        if x == m.X:
            return sum(m.Q_h[h] for h in m.h) == sum(m.Q[x2, x] for x2 in m.x)


    def Q_initial(m, h):
        return sum(m.Q[h, x2] for x2 in m.x) == m.Q_h[h]


    def one_direction_con(m, x, x2):
        if x != x2:
            return m.Q[x, x2] * m.Q[x2, x] == 0;
        else:
            return Constraint.Skip


    def pipes_construction_Con(m, x, x2):
        return m.Q[x, x2] <= m.Q_poss[x, x2]


    def Source_Con(m):  # 12
        return m.PK_bron == m.Q_source * (m.ro * m.sw * (m.T_source_in - m.T_source_out))


    # initialize constraints
    # cost
    m.CostEnergyConstraint = Constraint(rule=CostEnergy_Con)  # 2
    m.CostTubesConstraint = Constraint(rule=CostTubes_Con)  # 6
    m.RevenueConstraint = Constraint(rule=Revenue_Con)  # 9
    m.CostSourceConstraint = Constraint(rule=Cost_Source_Con)
    # m.CostStreetsConstraint = Constraint(rule=Cost_Streets_Con)
    m.PgridConstraint = Constraint(m.h, rule=Pgrid_Con)

    # house
    m.EnergyHouseConstraint = Constraint(m.h, rule=Energy_Use_House_Con)  # 1
    m.DischargeConstraint = Constraint(m.x, m.x, rule=Discharge_Con)  # 3
    m.PowerConstraint = Constraint(m.h, rule=Power_Con)  # 4

    # pipes
    m.Q_sourceConstraint = Constraint(rule=Q_source_Con)
    m.Q_massBalanceConstraint = Constraint(m.x, rule=Q_mass_balance)
    m.onedirectionConstraint = Constraint(m.x, m.x, rule=one_direction_con)
    m.PipesConstructionConstraint = Constraint(m.x, m.x, rule=pipes_construction_Con)
    m.pipesInitialConstraint = Constraint(m.h, rule=Q_initial)

    # Source
    m.SourceConstrant = Constraint(rule=Source_Con)



    opt = SolverFactory('ipopt')
    # opt.options['linear_solver'] = 'ma57'
    instance = m.create_instance()
    results = opt.solve(instance, tee=True,  options={'tol': 0.01, 'max_iter': 10000,  'hessian_approximation' : 'limited-memory'})
    #instance.pprint('network_constructed')
    instance.solutions.store_to(results)
    results.write(filename='results_9_1temperatuur_wijk.json', format='json')
