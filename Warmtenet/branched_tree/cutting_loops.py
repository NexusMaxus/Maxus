import geopandas as gpd
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt

import random
from branched_tree.network_functions import create_unique_points_and_merge_panden, \
    get_all_connections, get_all_connected_points, store_connected_points_per_point
from branched_tree.profit_functions import calculate_revenue

points_with_house_and_source = gpd.read_file('../data/loops_district/Aansluitpunten.geojson')
all_points = gpd.read_file('../data/loops_district/Kruispunten.geojson')

kruispunten_mask = []
for cell in all_points.id:
    if isinstance(cell, str):
        kruispunten_mask.append('kruispunt' in cell)
    else:
        kruispunten_mask.append(False)

junctions = all_points[kruispunten_mask]

points_with_house = points_with_house_and_source[points_with_house_and_source['pandidentificatie'] != 'BRON']
points_with_house.loc[:, 'pandidentificatie'] = [str(p[1:]) for p in points_with_house['pandidentificatie']]

# generate price for each house
price_threshold = []
warmtevraag = []
for index in points_with_house_and_source.index:
    if points_with_house_and_source.loc[index, 'pandidentificatie'] != 'BRON':
        price_threshold.append(random.randint(50, 60))
        warmtevraag.append(random.randint(300, 400))
    else:
        price_threshold.append(999)
        warmtevraag.append(0)

# select houses above price threshold
points_with_house_and_source['threshold'] = price_threshold
points_with_house_and_source['warmtevraag'] = warmtevraag
points_with_house_and_source = points_with_house_and_source[points_with_house_and_source.threshold >= 58]

# load junctions and put all selected points in dataframe
points = pd.concat([points_with_house_and_source, junctions], ignore_index=True, sort=False)

# load roads
roads = gpd.read_file('../data/loops_district/Wegen.geojson')

# see what point are so near to one another that they should be treated as one
points_unique_geometry = create_unique_points_and_merge_panden(points)

# points_unique_geometry = gpd.read_file('./point_unique_geometry.geojson')
connections = get_all_connections(roads=roads, points=points_unique_geometry)

connected_points = get_all_connected_points(connections=connections, points=points_unique_geometry)

# see what points can connect to other points
p2p, streets, cost_streets = store_connected_points_per_point(connected_points, connections)
# cut_loops
print(p2p)

index_bron = points_unique_geometry[points_unique_geometry['pandidentificatie'] == 'BRON'].index.values[0]


def find_loops(p2p, index_bron):
    paths = {}
    x = 0
    paths[x] = [index_bron]
    active_keys = [x]
    cuts = []
    loops = []
    number_in_row = 0
    num_loops = 0

    while len(active_keys) > 0:
        if num_loops == len(loops):
            number_in_row += 1
        else:
            number_in_row = 0

        if number_in_row > 50:
            break

        num_loops = len(loops)

        print('loops_found:', len(loops))
        rounds = active_keys.copy()
        for key in rounds:
            path_orig = paths[key].copy()
            p_conn = np.array(p2p[path_orig[-1]])
            if len(p_conn) == 1 and path_orig[-1] != index_bron:
                active_keys.remove(key)
            else:
                if len(p_conn) == 1:
                    paths[key].append(p_conn[0])
                else:
                    for i, p_index in enumerate(p_conn[p_conn != path_orig[-2]]):
                        if ((path_orig[-1], p_index) not in cuts) and ((p_index, path_orig[-1]) not in cuts):

                            if i == 0:  # continue existing path
                                if p_index not in paths[key]:
                                    paths[key].append(p_index)
                                else:
                                    active_keys.remove(key)
                                    loop = path_orig[path_orig.index(p_index):] + [p_index]
                                    if (loop not in loops) and (loop.reverse() not in loops):
                                        loops.append(loop)
                                        cut = find_cut(loop, cost_streets)
                                        cuts.append(cut)

                            if i > 0:   # start new path if not loop
                                x += 1
                                paths[x] = path_orig + [p_index]
                                if p_index not in path_orig:
                                    active_keys.append(x)
                                else:
                                    loop = path_orig[path_orig.index(p_index):] + [p_index]
                                    if (loop not in loops) and (loop.reverse() not in loops):
                                        loops.append(loop)
                                        cut = find_cut(loop, cost_streets)
                                        cuts.append(cut)
                        else:
                            if key in active_keys:
                                active_keys.remove(key)
    return cuts
        # for key in active_keys:

def find_cut(loop, cost_streets):
    costs_segments_in_loop = []
    for i in range(len(loop) - 1):
        costs_segments_in_loop.append(cost_streets[loop[i], loop[i+1]])

    index_exp = np.argmax(costs_segments_in_loop)
    return loop[index_exp], loop[index_exp + 1]


def plot_loop(new_connections):
    f, ax = plt.subplots()
    new_connections.plot(ax=ax)
    # roads.loc[roads_to_plot].to_file('~/PycharmProjects/Maxus/Warmtenet/data/loops_district/output_cut_network.geojson', driver='GeoJSON')
    points_unique_geometry.plot(ax=ax, color='r')
    plt.show()


cuts = find_loops(p2p, index_bron)
street_index = [streets[(cut[0], cut[1])] for cut in cuts]
streets_branched = list(set(connections.index) - set(street_index))
new_connections = connections.loc[streets_branched]
plot_loop(new_connections)
print(cuts)

new_connected_points = get_all_connected_points(new_connections, points_unique_geometry)
p2p, streets_branched, cost_streets_branched = store_connected_points_per_point(new_connected_points, connections)
end_point_branches = {key: value for key, value in p2p.items() if len(value) == 1}
end_point_branches.pop(index_bron)
junctions_branched = {key: value for key, value in p2p.items() if len(value) > 2}
junctions_branched_status = {key: [False for v in value] for key, value in junctions_branched.items()}
junctions_branched_income = {key: [0 for v in value] for key, value in junctions_branched.items()}
junctions_branched_cost = {key: [0 for v in value] for key, value in junctions_branched.items()}
junctions_branched_profit = {key: [0 for v in value] for key, value in junctions_branched.items()}
junctions_branched_points = {key: [[] for v in value] for key, value in junctions_branched.items()}

paths = {}
income = {}
cost = {}
profit = {}
finished_points = []
losing_points = []

for x, key in enumerate(end_point_branches.keys()):
    paths[x] = [key]
x = len(paths)

active_keys = list(paths.keys())

while len(active_keys) > 0:
    rounds = active_keys.copy()
    for key in rounds:
        print('round:', key)
        if len(paths[key]) > 1:
            income[key] += calculate_revenue(paths[key][-2], points_unique_geometry, points_with_house_and_source)
            cost[key] += cost[key] + cost_streets_branched[paths[key][-1], paths[key][-2]]
            profit[key] += income[key] - cost[key]
        else:
            profit[key] = 0
            cost[key] = 0
            income[key] = 0

        if profit[key] < 0:
            print('popping key because of profit:', key)
            active_keys.remove(key)
            if len(p2p[paths[key][-1]]) > 2:
                p_index = p2p[paths[key][-1]].index(paths[key][-2])
                junctions_branched_status[paths[key][-1]][p_index] = True
            else:
                x += 1
                paths[x] = [paths[key][-1]]
                finished_points.extend(paths[key][:-1])
                losing_points.extend(paths[key][:-1])
                active_keys.append(x)


        else:
            if len(p2p[paths[key][-1]]) == 1:
                if not p2p[paths[key][-1]] in paths[key]:
                    next_point = p2p[paths[key][-1]]
                    paths[key].extend(next_point)
                else:
                    print('BRON found')
                    active_keys.remove(key)
                    print('profit:', profit[key])

            elif len(p2p[paths[key][-1]]) == 2:
                next_point = None
                for point in p2p[paths[key][-1]]:
                    if (point != paths[key][-2]) and (point not in finished_points):
                        next_point = point

                if next_point is not None:
                    paths[key].append(next_point)
                else:
                    raise ValueError(f'couldnt find next point for {paths[key][-1]}')

            elif len(p2p[paths[key][-1]]) > 2:
                print('popping key because of junction:', key)
                active_keys.remove(key)
                finished_points.extend(paths[key][:-1])
                p_index = p2p[paths[key][-1]].index(paths[key][-2])
                junctions_branched_status[paths[key][-1]][p_index] = True
                junctions_branched_income[paths[key][-1]][p_index] = income[key]
                junctions_branched_cost[paths[key][-1]][p_index] = cost[key]
                junctions_branched_profit[paths[key][-1]][p_index] = profit[key]
                junctions_branched_points[paths[key][-1]][p_index] = paths[key][:-1]

                if all(junctions_branched_status[paths[key][-1]]):
                    x += 1
                    paths[x] = [item for sublist in junctions_branched_points[paths[key][-1]] for item in sublist] + [paths[key][-1]]
                    income[x] = sum(junctions_branched_income[paths[key][-1]])
                    cost[x] = sum(junctions_branched_cost[paths[key][-1]])
                    profit[x] = sum(junctions_branched_profit[paths[key][-1]])
                    active_keys.append(x)

f, ax = plt.subplots()
points_unique_geometry.plot(ax=ax)
points_unique_geometry.loc[losing_points].plot(ax=ax, color='r')
new_connections.plot(ax=ax)
plt.show()