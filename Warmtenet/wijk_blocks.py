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
points.reset_index(drop=True, inplace=True)

points_in_roads = np.zeros((len(roads.geometry),len(points.geometry)))
points_in_roads_short = np.ones((len(roads.geometry), 2), dtype=int)*-1


for i, road in enumerate(roads.geometry):
    x = 0;
    for j, point in enumerate(points.geometry):
        if road.distance(point) < 1e-8:
            points_in_roads[i, j] = 1
            points_in_roads_short[i, x] = j
            x += 1

points_connected = np.zeros((len(points.geometry), len(points.geometry)))
length_roads = 0.01 * np.ones((len(points.geometry), len(points.geometry)))


for i in range(len(points_in_roads_short)):
    if np.logical_and(points_in_roads_short[i, 0] != -1, points_in_roads_short[i, 1] != -1):
        points_connected[points_in_roads_short[i, 0], points_in_roads_short[i, 1]] = 1
        points_connected[points_in_roads_short[i, 1], points_in_roads_short[i, 0]] = 1
        length_roads[points_in_roads_short[i, 1], points_in_roads_short[i, 0]] = roads['length'][i]
        length_roads[points_in_roads_short[i, 0], points_in_roads_short[i, 1]] = roads['length'][i]

streets_connected = np.zeros(np.size(points_connected,0))
points_connected_to_points = {}

for i in range(np.size(points_connected,0)):
    streets_connected[i] = sum(points_connected[i, :])
    points_connected_to_points[i] = np.squeeze(np.argwhere(points_connected[i, :] == 1))

optimize = True

if optimize:

    X = len(points) -1
    H = len(buildings) - 1
    P_h = pd.Series(
        data=random.sample(np.arange(10, 100, 0.1).tolist(), int(len(buildings))))  # 50 * np.ones(int(len(buildings))))
    PKhuis = pd.Series(data=buildings['area'] / (1000 * np.ones(
        int(len(buildings)))))  # random.sample(np.arange(1000,1000.5,0.001).tolist(), int(len(buildings))))
    WV = pd.Series(data=buildings['area'] / (15 * np.ones(int(len(buildings)))))

    C_constant_source = 20000
    C_street = pd.DataFrame(data=points_connected * 2)

    Q_ini = pd.DataFrame(data=np.zeros_like(points_connected), dtype=int)
    length_roads_pd = pd.DataFrame(data=length_roads)
    number_of_points_connected = pd.DataFrame(data=streets_connected, dtype=int)

    # del (points)
    # del (roads)
    # del (buildings)

    m = AbstractModel()

    m.H = Param(within=NonNegativeIntegers, initialize=H)
    m.X = Param(within=NonNegativeIntegers, initialize=X)
    m.X2 = Param(within=NonNegativeIntegers, initialize=p2p.shape[1])

    # sets
    m.h = RangeSet(0, m.H)
    m.x = RangeSet(0, m.X)
    m.x2 = RangeSet(0, m.X2)

    # parameters
    m.Num = Param(m.x, within=NonNegativeIntegers, initialize=number_of_points_connected.to_dict())
    m.P_h = Param(m.h, within=NonNegativeReals, initialize=P_h.to_dict())
    m.WV = Param(m.h, within=NonNegativeReals, initialize=WV.to_dict())
    m.CostSource = Param(domain=NonNegativeReals, initialize=C_constant_source)

    # variables
    m.Conn = Var(m.h, within=Binary)
    m.P_grid = Var(domain=NonNegativeReals, initialize=0)
    m.Q = Var(m.x, m.x, domain=Binary, initialize=Q_ini.stack().to_dict())

    # blocks
    def junction(p, x):
        p.ID = x
        p.h = Param(initialize=house_or_not[x])
        p.i = Set(p2p[x])
        p.Q_to = Var(p.i, within=Binary)
        p.Q_from = Var(p.i, within=Binary)
        p.junction_balance = Constraint(expr=(sum(p.Q_to[i] for i in p.i) + p.h) * (1-sum(p.Q_from[i] for i in p.i) == 0))


        def opp_directions(i):
            return p.Q_to[i] * p.Q_to[i] == 0
        p.opp_directions = Constraint(p.i, rule=opp_directions)


    m.junction = Block(m.x, rule=junction)


    def linking_rule(m, x):
        return m.p[x].Q_to[m.p[x].i] == m.p[m.p[x].i].Q_from[x]



    # variables: cost
    m.Revenue = Var(domain=NonNegativeReals, initialize=0)
    m.CostStreets = Var(domain=NonNegativeReals, initialize=0)





  # .............................
    def obj_expression(m):
        return m.Revenue - m.CostSource #- m.CostStreets


    m.OBJ = Objective(rule=obj_expression, sense=maximize)


    def revenue_con(m):
        return m.Revenue == sum(m.P_grid * m.WV[h] * m.Conn[h] for h in m.h)

    # Constraints
    def p_grid_con(m, h):
        return m.Conn[h] * (m.P_grid - m.P_h[h]) <= 0

    def cost_streets_con(m):
        return m.CostStreets == sum(sum(m.Q[x, x2] * m.C_Street[x, x2] for x in m.x) for x2 in m.x)



    def pipes_construction_con(m, x, x2):
        return m.Q[x, x2] <= m.Q_poss[x, x2]
    #
    #
    # def one_direction_con(m, x, x2):
    #     if x != x2:
    #         return m.Q[x, x2] * m.Q[x2, x] == 0
    #     else:
    #         return Constraint.Skip

    m.Linking = Constraint(m.x, )
    m.Q_massBalanceConstraint = Constraint(m.x, rule=q_mass_balance)
    m.RevenueConstraint = Constraint(rule=revenue_con)  # 9
    m.PgridConstraint = Constraint(m.h, rule=p_grid_con)
    m.CostStreetsConstraint = Constraint(rule=cost_streets_con)
    m.PipesConstructionConstraint = Constraint(m.x, m.x, rule=pipes_construction_con)
    # m.OneDirectionConstraint = Constraint(m.x, m.x, rule=one_direction_con)

    # opt = SolverFactory('ipopt')
    # # opt.options['linear_solver'] = 'ma57'
    # instance = m.create_instance()
    # results = opt.solve(instance, tee=True,
    #                     options={'tol': 0.01, 'max_iter': 10000, 'hessian_approximation': 'limited-memory',
    #                              'print_level': 5})

    opt = SolverFactory('mindtpy')
    # opt.options['linear_solver'] = 'ma86'
    instance = m.create_instance()
    results = opt.solve(instance, mip_solver='glpk', nlp_solver='ipopt', tee=True)
    instance.display(filename='results_11_wijk_binary.txt' )
    # # instance.pprint('network_constructed')
    # instance.solutions.store_to(results)
    # results.write(filename='results_11_wijk_binary.json', format='json')
