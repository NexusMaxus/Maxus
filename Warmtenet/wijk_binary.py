from __future__ import division
import geopandas as gpd
import numpy as np
import pandas as pd
from pyomo.environ import *
from pyomo.opt import SolverFactory, TerminationCondition
import random

points = gpd.read_file('/home/rogier/earlybirds/assen_test/AllPoints.shp')
roads = gpd.read_file(r'/home/rogier/earlybirds/assen_test/wegen_wijk.shp')
buildings = gpd.read_file(r'/home/rogier/earlybirds/assen_test/shape/buildings.shp')


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

optimize = True

if optimize:

    X = len(points) - 1
    H = len(buildings) - 1
    P_h = pd.Series(
        data=random.sample(np.arange(10, 100, 0.1).tolist(), int(len(buildings))))  # 50 * np.ones(int(len(buildings))))
    PKhuis = pd.Series(data=buildings['area'] / (1000 * np.ones(
        int(len(buildings)))))  # random.sample(np.arange(1000,1000.5,0.001).tolist(), int(len(buildings))))
    WV = pd.Series(data=buildings['area'] / (15 * np.ones(int(len(buildings)))))

    T_in = 60
    T_out = pd.Series(data=40 * np.ones(len(buildings)))
    T_source_in = 30
    T_source_out = 10

    ro = 1
    sw = 4.18

    C_constant_source = 20000

    C_street = pd.DataFrame(data=PointsConnected * 2)
    Q_poss = pd.DataFrame(data=PointsConnected * 1, dtype=int)
    length_roads_pd = pd.DataFrame(data=length_roads)

    # del (points)
    # del (roads)
    # del (buildings)

    m = AbstractModel()

    m.H = Param(within=NonNegativeIntegers, initialize=H)
    m.X = Param(within=NonNegativeIntegers, initialize=X)

    # sets
    m.h = RangeSet(0, m.H)
    m.x = RangeSet(0, m.X)

    # parameters
    m.P_h = Param(m.h, within=NonNegativeReals, initialize=P_h.to_dict())
    m.WV = Param(m.h, within=NonNegativeReals, initialize=WV.to_dict())
    m.PKhuis = Param(m.h, within=NonNegativeReals, initialize=PKhuis.to_dict())

    m.T_in = Param(within=NonNegativeReals, initialize=T_in)
    m.T_out = Param(m.h, within=NonNegativeReals, initialize=T_out.to_dict())
    m.ro = Param(within=NonNegativeReals, initialize=ro)  # dichtheid | density
    m.sw = Param(within=NonNegativeReals, initialize=sw)  # ...  | soortelijke warmte
    m.CostSource = Param(domain=NonNegativeReals, initialize=C_constant_source)

    m.Q_poss = Param(m.x, m.x, within=Binary, initialize=Q_poss.stack().to_dict(), mutable=True)
    m.length = Param(m.x, m.x, within=NonNegativeReals, initialize=length_roads_pd.stack().to_dict())

    # parameters: cost
    m.C_Street = Param(m.x, m.x, within=NonNegativeReals, initialize=C_street.stack().to_dict())

    # variables
    m.Q = Var(m.x, m.x, domain=Binary)
    m.Conn = Var(m.h, within=Binary)
    m.Q_h = Var(m.h, domain=NonNegativeReals)
    m.P_grid = Var(domain=NonNegativeReals, initialize=0)

    # variables: cost
    m.Revenue = Var(domain=Reals, initialize=0)
    m.CostStreets = Var(domain=NonNegativeReals, initialize=0)


    def obj_expression(m):
        return m.Revenue - m.CostSource - m.CostStreets


    m.OBJ = Objective(rule=obj_expression, sense=maximize)


    def Revenue_Con(m):
        return m.Revenue == sum(m.P_grid * m.WV[h] * m.Conn[h] for h in m.h)


    def Cost_Streets_Con(m):
        return m.CostStreets == sum(sum(m.Q[x, x2] * m.length[x, x2] * m.C_Street[x, x2] for x in m.x) for x2 in m.x)


    # Constraints
    def Pgrid_Con(m, h):
        return m.Conn[h] * (m.P_grid - m.P_h[h]) <= 0


    def Power_Con(m, h):  # 4
        return m.Q_h[h] * (m.ro * m.sw * (m.T_in - m.T_out[h])) == \
               m.Conn[h] * m.PKhuis[h]


    def Q_mass_balance(m, x):
        if x <= m.H:
            return Constraint.Skip
        if (x > m.H) and (x < m.X):
            return sum(1 - m.Q[x, x2] for x2 in m.x) * (sum(m.Q[x2, x] for x2 in m.x)) == 0
        if x == m.X:
            return sum(m.Q_h[h] for h in m.h) * (1 - sum(m.Q[x2, x] for x2 in m.x)) == 0

    def one_direction_con(m, x, x2):
        if x != x2:
            return m.Q[x, x2] * m.Q[x2, x] == 0
        else:
            return Constraint.Skip

    def Q_initial(m, h):
        return m.Q_h[h] * (1 - sum(m.Q[h, x2] for x2 in m.x)) == 0

    def pipes_construction_Con(m, x, x2):
        return m.Q[x, x2] * (1 - m.Q_poss[x, x2]) == 0


    # initialize constraints
    # cost

    m.RevenueConstraint = Constraint(rule=Revenue_Con)  # 9
    m.CostStreetsConstraint = Constraint(rule=Cost_Streets_Con)
    m.PgridConstraint = Constraint(m.h, rule=Pgrid_Con)

    # house
    m.PowerConstraint = Constraint(m.h, rule=Power_Con)  # 4

    # pipes
    m.Q_massBalanceConstraint = Constraint(m.x, rule=Q_mass_balance)
    m.onedirectionConstraint = Constraint(m.x, m.x, rule=one_direction_con)
    m.pipesInitialConstraint = Constraint(m.h, rule=Q_initial)
    m.PipesConstructionConstraint = Constraint(m.x, m.x, rule=pipes_construction_Con)

    opt = SolverFactory('mindtpy')
    # opt.options['linear_solver'] = 'ma86'
    instance = m.create_instance()
    results = opt.solve(instance, mip_solver='glpk', nlp_solver='ipopt', tee=True)
    # instance.pprint('network_constructed')
    instance.solutions.store_to(results)
    results.write(filename='results_11_wijk_binary.json', format='json')
