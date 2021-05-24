from __future__ import division
import geopandas as gpd
import pandas as pd
import numpy as np
from shapely.geometry import Point, LineString
import matplotlib.pyplot as plt
from pyomo.environ import *
import geopandas as gpd
import numpy as np
import pandas as pd
from pyomo.opt import SolverFactory, TerminationCondition
import pyomo.kernel as pmo
from pandas.io.json import json_normalize
import json
import io

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

points_with_house_and_source = gpd.read_file('./totaalgebied/Aansluitpunten.geojson')
points_with_house_and_source.drop(169, inplace=True)
points_with_house = points_with_house_and_source[points_with_house_and_source['pandidentificatie'] != 'BRON']
points_with_house.loc[:, 'pandidentificatie'] = [str(p[1:]) for p in points_with_house['pandidentificatie']]

junctions = gpd.read_file('./totaalgebied/Kruispunten.geojson')
points = pd.concat([points_with_house_and_source, junctions], ignore_index=True, sort=False)
roads = gpd.read_file('./totaalgebied/Wegen.geojson')

# points_unique_geometry = gpd.read_file('./deelgebied/Kruispunten_Ap_deel.geojson')
points_unique_geometry = points[~points.geometry.duplicated()].reset_index()

with io.open('./totaalgebied/data_Wnet_0.json', encoding='utf-8-sig') as f:
    houses_data = json.load(f)
houses_data = json_normalize(houses_data['dataWnet'])

# see what roads contain multiple points (roads that do not go only from one to another point)
long_roads = {}
road_point_count = np.zeros(roads.shape[0], dtype=int)

for i, road in enumerate(roads.geometry):
    x = 0
    p_array = []
    for j, point in enumerate(points_unique_geometry.geometry):
        if road.distance(point) < 1e-1:
            x += 1
            p_array.append(j)
    if x > 2:
        long_roads[i] = p_array
    road_point_count[i] = x

# split roads that contain multiple points in smaller parts
smaller_parts = []
for road_number, segments in long_roads.items():
    points_in_long_road = []
    distance = []
    for j in segments:
        distance.append(roads.geometry[road_number].project(points_unique_geometry.geometry[j]))
        points_in_long_road.append(points_unique_geometry.geometry[j])

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


points_in_road = np.zeros((len(roads.geometry),len(points_unique_geometry.geometry)), dtype=np.int16)
points_in_road_short = np.ones((len(roads.geometry),2),dtype=np.int16)*-1

for i, road in enumerate(roads.geometry):
    x = 0
    for j, point in enumerate(points_unique_geometry.geometry):
        if road.distance(point) < 1e-1:
            points_in_road[i, j] = 1
            points_in_road_short[i, x] = j
            x += 1

# see what points can connect to other points
p2p = {}
cost_streets = {}

for i in range(len(points_in_road_short)):
    if np.logical_and(points_in_road_short[i, 0] != -1, points_in_road_short[i, 1] != -1):
        if points_in_road_short[i, 0] not in p2p.keys():
            p2p[points_in_road_short[i, 0]] = [points_in_road_short[i, 1]]
            cost_streets[points_in_road_short[i, 0]] = [roads.geometry[i].length * 100]
        else:
            p2p[points_in_road_short[i, 0]].append(points_in_road_short[i, 1])
            cost_streets[points_in_road_short[i, 0]].append(roads.geometry[i].length * 100)

        if points_in_road_short[i, 1] not in p2p.keys():
            p2p[points_in_road_short[i, 1]] = [points_in_road_short[i, 0]]
            cost_streets[points_in_road_short[i, 1]] = [roads.geometry[i].length * 100]
        else:
            p2p[points_in_road_short[i, 1]].append(points_in_road_short[i, 0])
            cost_streets[points_in_road_short[i, 1]].append(roads.geometry[i].length * 100)


houses_connected = {}
for j, point in points_unique_geometry.geometry.items():
    houses = []
    for i, h in points_with_house.geometry.items():
        if h.distance(point) < 1e-1:
            houses.append(i)
    houses_connected[j] = houses


houses_in_district = []
for pand in houses_data['PandBAGnr'].values:
    if str(pand) in list(points_with_house.loc[:, 'pandidentificatie'].values):
        houses_in_district.append(True)
    else:
        houses_in_district.append(False)

houses_data = houses_data[houses_in_district]
houses_data = houses_data.reset_index()

p2a = {}
for pand in points_with_house.loc[:,'pandidentificatie']:
    p2a[pand] = houses_data[houses_data['PandBAGnr'] == pand].index[0]

p2h = {}
for key, list_of_houses in houses_connected.items():
    p2h[key] = [np.squeeze(p2a[points_with_house.loc[house, 'pandidentificatie']].values[0]) for house in list_of_houses]


house_or_not = [len(x) > 0 for x in houses_connected.values()]
source_or_not = list((points_unique_geometry['pandidentificatie'] == 'BRON').values)
pkhuis = houses_data["GJLT"]/365*3
wvhuis = houses_data["GJLT"]
p_h = houses_data["WprijsLT"]

# def step_func(x):
#     delta = 0.1
#     z = ((2 + pmo.exp(-5 * (x - delta))) / (1 + pmo.exp(-5 * (x - delta)))) - 1
#     return z

T_in = 60
T_out = 40
ro = 1
sw = 4.18
period = 30

m = AbstractModel()
m.x = RangeSet(0, len(points_unique_geometry)-1)
m.h = RangeSet(0, len(houses_data)-1)


m.P_grid = Var(domain=pmo.NonNegativeReals, initialize=15, bounds=(0, 20))
# m.P_grid = Param(domain=pmo.NonNegativeReals, initialize=10)


def junction(b, x):

    b.i = Set(initialize=p2p[x])
    b.Q_to = Var(b.i, domain=pmo.NonNegativeReals, initialize=dict(zip(b.i, np.zeros(len(b.i)))))
    b.Q_from = Var(b.i, domain=pmo.NonNegativeReals, initialize=dict(zip(b.i, np.zeros(len(b.i)))))
    b.street_cost = Var(domain=pmo.NonNegativeReals, initialize=0)
    b.h = Param(domain=pmo.Boolean, initialize=house_or_not[x])
    b.s = Param(domain=pmo.Boolean, initialize=source_or_not[x])

    if b.h:
        b.Q_from_h = Var(domain=pmo.NonNegativeReals, initialize=0)
        b.IDh = Set(initialize=p2h[x])
    if b.s:
        b.Q_to_s = Var(domain=pmo.NonNegativeReals, initialize=0)

    b.cost_street = Param(b.i, initialize=dict(zip(b.i, cost_streets[x])), domain=pmo.NonNegativeReals)

    b.street_open = Var(b.i, domain=pmo.Binary)


    def junction_balance_rule(b):
        if b.h:
            return sum(b.Q_from[i] for i in b.i) + b.Q_from_h == sum(b.Q_to[i] for i in b.i)
        elif b.s:
            return sum(b.Q_from[i] for i in b.i) == sum(b.Q_to[i] for i in b.i) + b.Q_to_s
        else:
            return sum(b.Q_from[i] for i in b.i) == sum(b.Q_to[i] for i in b.i)

    def junction_streets_rule(b):
        return b.street_cost == sum(b.street_open[i] * b.cost_street[i] for i in b.i)

    def street_open_rule(b, i):
        return b.Q_to[i] * (1 - b.street_open[i]) == 0

    b.junction_balance = Constraint(rule=junction_balance_rule)
    b.junction_streets = Constraint(rule=junction_streets_rule)
    b.street_open_con = Constraint(b.i, rule=street_open_rule)


def house(b, h):
    b.T_in = Param(domain=pmo.NonNegativeReals, initialize=T_in)
    b.T_out = Param(domain=pmo.NonNegativeReals, initialize=T_out)
    b.ro = Param(within=pmo.NonNegativeReals, initialize=ro)  # dichtheid | density
    b.sw = Param(within=pmo.NonNegativeReals, initialize=sw)
    b.PKhuis = Param(domain=pmo.NonNegativeReals, initialize=pkhuis[h])
    b.WV = Param(domain=pmo.NonNegativeReals, initialize=wvhuis[h])
    b.P_h = Param(domain=pmo.Reals, initialize=p_h[h])

    b.Q_h = Var(domain=pmo.NonNegativeReals, initialize=0)
    b.Conn = Var(domain=pmo.Binary, initialize=0)

    def Power_Con(b):
        return b.Q_h * (b.ro * b.sw * (b.T_in - b.T_out)) == \
               b.Conn * b.PKhuis

    b.PowerConstraint = Constraint(rule=Power_Con)


m.p = Block(m.x, rule=junction)
m.d = Block(m.h, rule=house)


def link_rule(m, x, x2):
    if x2 in m.p[x].i:
        return m.p[x].Q_to[x2] == m.p[x2].Q_from[x]
    else:
        return Constraint.Skip
m.linking_points = Constraint(m.x, m.x, rule=link_rule)


def Q_initial(m, x):
    if m.p[x].h:
        return sum(m.d[h].Q_h for h in m.p[x].IDh) == m.p[x].Q_from_h
    else:
        return Constraint.Skip
m.house_to_point = Constraint(m.x, rule=Q_initial)


def opp_directions_rule(m, x, x2):
    if (x2 in m.p[x].i) and (x < x2):
        return m.p[x].Q_to[x2] * m.p[x2].Q_to[x] == 0
    else:
        return Constraint.Skip
m.opposite_directions = Constraint(m.x, m.x, rule=opp_directions_rule)

def Pgrid_Con(m, h):
    return m.d[h].Conn * (m.P_grid - m.d[h].P_h) <= 0
m.PgridConstraint = Constraint(m.h, rule=Pgrid_Con)

def obj_rule(m):
    return sum(m.P_grid * m.d[h].WV * m.d[h].Conn for h in m.h)*10 - sum(m.p[x].street_cost for x in m.x)
m.obj = Objective(rule=obj_rule, sense=maximize)


# opt = SolverFactory('baron', executable="/home/rogier/PycharmProjects/solvers/baron-lin64/baron")
# opt = SolverFactory('gurobi')
# opt = SolverFactory('couenne')
# opt = SolverFactory('mindtpy')
opt = SolverFactory('ipopt', solver_args = {'acceptable_tol': 0.0001, 'constr_viol_tol':0.0001, 'acceptable_constr_viol_tol':0.0001})
instance = m.create_instance()
# results = opt.solve(instance, mip_solver='glpk', nlp_solver='ipopt', strategy='OA', tee=True)
# opt.options['MaxTime']=-1
# opt.options['threads']=4
results = opt.solve(instance, tee=True)
# results = opt.solve(instance,  solver='ipopt',  strategy='rand_guess_and_bound', stopping_mass=0.8, HCS_tolerance = 0.00001)

# options=)
instance.solutions.store_to(results)
# instance.pprint()
instance.display('results_14_block_simple_constraints_ipopt.txt')

results.write(filename='results_14_block_simple_ipopt.json', format='json')


