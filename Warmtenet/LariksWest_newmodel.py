from __future__ import division
import geopandas as gpd
import numpy as np
import pandas as pd
from pyomo.environ import *
from pyomo.opt import SolverFactory, TerminationCondition
from shapely.geometry import Point, LineString
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


def cut(line, distance):
    """Cuts a line in two at a distance from its starting point"""
    if distance <= 0.0 or distance >= line.length:
        return [LineString(line)]
    coords = list(line.coords)
    for i, p in enumerate(coords):
        pd = line.project(Point(p))
        if pd == distance:
            return [
                LineString(coords[:i+1]),
                LineString(coords[i:])]
        if pd > distance:
            cp = line.interpolate(distance)
            return [
                LineString(coords[:i] + [(cp.x, cp.y)]),
                LineString([(cp.x, cp.y)] + coords[i:])]


def split_line_with_points(line, points):
    """Splits a line string in several segments considering a list of points.

    The points used to cut the line are assumed to be in the line string
    and given in the order of appearance they have in the line string.
    ['LINESTRING (1 2, 8 7, 4 5, 2 4)', 'LINESTRING (2 4, 4 7, 8 5, 9 18)', 'LINESTRING (9 18, 1 2, 12 7, 4 5, 6 5)', 'LINESTRING (6 5, 4 9)']

    """
    segments = []
    current_line = line
    for p in points:
        d = current_line.project(p)
        seg, current_line = cut(current_line, d)
        segments.append(seg)
    segments.append(current_line)
    return segments


crs = {'init': 'epsg:28992'}
optimize = True


points = gpd.read_file('./data/All_Points.shp')
roads = gpd.read_file('./data/All_Roads.shp')
buildings = gpd.read_file('./data/Lariks-west-pand.shp')

buildings.sort_values(by='identifica', inplace=True)
buildings.reset_index(drop=True, inplace=True)

buildings_csv = pd.read_csv('./data/DEMO-gebouwgegevens.csv')
buildings_csv.sort_values(by='identifica', inplace=True)
buildings_csv.reset_index(drop=True, inplace=True)
buildings = buildings.join(buildings_csv, lsuffix='l')

points.sort_values(by=['point_type'], inplace=True)
points = points.reset_index(drop=True)
idx = points.index.tolist()
source = points.loc[points['point_type'] == 'b'].index.values[0]
idx.pop(source)
points = points.reindex(idx+[source])
points = points.reset_index(drop=True)

# see what roads contain multiple points (roads that do not go only from one to another point)
long_roads = {}
road_point_count = np.zeros(roads.shape[0], dtype=int)

for i, road in enumerate(roads.geometry):
    x = 0
    p_array = []
    for j, point in enumerate(points.geometry):
        if road.distance(point) < 1e-1:
            x += 1
            p_array.append(j)
        road_point_count[i] = x
    if x > 2:
        long_roads[i] = p_array

# split roads that contain multiple points in smaller parts
smaller_parts = []
for road_number, segments in long_roads.items():
    points_in_long_road = []
    distance = []
    for j in segments:
        distance.append(roads.geometry[road_number].project(points.geometry[j][0]))
        points_in_long_road.append(points.geometry[j][0])

    df_points_in_road = pd.DataFrame({'dis': distance})
    points_gpd = gpd.GeoDataFrame(df_points_in_road , crs=crs, geometry=points_in_long_road)
    points_gpd.sort_values('dis', inplace=True)
    points_gpd.reset_index(drop=True, inplace=True)
    smaller_parts.extend(split_line_with_points(roads.geometry[road_number], points_gpd.geometry[1:-1]))

# merge split roads with other roads
roads.drop(axis=0, index=long_roads.keys(), inplace=True)
gdf_long_roads_split = gpd.GeoDataFrame(crs=crs, geometry=smaller_parts)
gdf_small_roads = gpd.GeoDataFrame(crs=crs, geometry=roads.geometry)
roads = pd.concat([gdf_small_roads, gdf_long_roads_split], axis=0, ignore_index=True, sort=False)

# see what points are connected by the roads
points_in_road = np.zeros((len(roads.geometry), len(points.geometry)), dtype=int)
points_in_road_short = np.ones((len(roads.geometry), 2), dtype=int)*-1
for i, road in enumerate(roads.geometry):
    x = 0
    for j, point in enumerate(points.geometry):
        if road.distance(point) < 1e-1:
            points_in_road[i, j] = 1
            points_in_road_short[i, x] = j
            x += 1

# see what points can connect to other points
points_connected = np.zeros((len(points.geometry), len(points.geometry)))
length_roads = 0.01 * np.ones((len(points.geometry), len(points.geometry)))

for i in range(len(points_in_road_short)):
    if np.logical_and(points_in_road_short[i, 0] != -1, points_in_road_short[i, 1] != -1):
        points_connected[points_in_road_short[i, 0], points_in_road_short[i, 1]] = 1
        points_connected[points_in_road_short[i, 1], points_in_road_short[i, 0]] = 1
        length_roads[points_in_road_short[i, 1], points_in_road_short[i, 0]] = roads.geometry[i].length
        length_roads[points_in_road_short[i, 0], points_in_road_short[i,1]] = roads.geometry[i].length

# check to how many streets the points are connected
streets_connected = np.zeros(np.size(points_connected, 0))
for i in range(0, np.size(points_connected, 0)):
    streets_connected[i] = sum(points_connected[i, :])

# check which points are in a building
points_mask = np.ones([len(points), 1])-2
building_mask = np.zeros([len(buildings), 1])-2
for i, point in enumerate(points.geometry):
    for j, polygon in enumerate(buildings.geometry):
        if polygon.contains(point):
            points_mask[i] = j
            building_mask[j] = i

mask = points_mask > -1
points['house'] = mask

# sort such that all points in buildings are on top (don't know why this was implemented?')
points.sort_values('house', inplace=True, ascending=False)
points.reset_index(inplace=True, drop=True)

if optimize:

    H = np.sum(mask)-1
    if H != len(buildings)-1:
        print(len(buildings), H)

    X = len(points)-1

    P_h = pd.Series(
        data=buildings['60-warmtep'] )
    PKhuis = pd.Series(data=buildings['60-piekcap'] )
    WV = pd.Series(data=buildings['60-warmtev'] )

    T_in = 60
    T_out = pd.Series(data=40 * np.ones(len(buildings)))

    T_source_in = 30
    T_source_out = 10

    ro = 1
    sw = 4.18
    C_constant_source = 20000

    C_street = pd.DataFrame(data=points_connected * 2)
    Q_poss = pd.DataFrame(data=points_connected * 999)
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

    m.Q_poss = Param(m.x, m.x, within=NonNegativeReals, initialize=Q_poss.stack().to_dict())
    m.length = Param(m.x, m.x, within=NonNegativeReals, initialize=length_roads_pd.stack().to_dict())

    # parameters: cost
    m.C_Street = Param(m.x, m.x, within=NonNegativeReals, initialize=C_street.stack().to_dict())

    # variables
    m.Q = Var(m.x, m.x, domain=NonNegativeReals)
    m.Conn = Var(m.h, within=Binary)
    m.Q_h = Var(m.h, domain=NonNegativeReals)
    m.P_grid = Var(domain=NonNegativeReals)

    # variables: cost
    m.Revenue = Var(domain=Reals)
    m.CostStreets = Var(domain=NonNegativeReals)


    def obj_expression(m):
        return m.Revenue - m.CostSource - m.CostStreets

    m.OBJ = Objective(rule=obj_expression, sense=maximize)

    def Revenue_Con(m):
        return m.Revenue == sum(m.P_grid * m.WV[h] * m.Conn[h] for h in m.h)

    def Cost_Streets_Con(m):
        return m.CostStreets == sum(sum(step_func(m.Q[x,x2]*10) * m.length[x,x2] * m.C_Street[x,x2] for x in m.x) for x2 in m.x)

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
            return sum(m.Q[x, x2] for x2 in m.x) == sum(m.Q[x2, x] for x2 in m.x)
        if x == m.X:
            return sum(m.Q_h[h] for h in m.h) == sum(m.Q[x2, x] for x2 in m.x)


    def Q_initial(m, h):
        return sum(m.Q[h, x2] for x2 in m.x) == m.Q_h[h]

    def one_direction_con(m, x, x2):
        if x != x2:
            return m.Q[x, x2] * m.Q[x2, x] == 0
        else:
            return Constraint.Skip

    def pipes_construction_Con(m, x, x2):
        return m.Q[x, x2] <= m.Q_poss[x, x2]

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
    results.write(filename='results_11_LariksWest.json', format='json')

