import geopandas as gpd
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import random
from branched_tree.network_functions import create_unique_points_and_merge_panden, \
    get_all_connections, get_all_connected_points, store_connected_points_per_point
from branched_tree.profit_functions import calculate_revenue

points_with_house_and_source = gpd.read_file('/home/rogier/PycharmProjects/Maxus/Warmtenet/data/loops_district/Aansluitpunten.geojson')
all_points = gpd.read_file('/home/rogier/PycharmProjects/Maxus/Warmtenet/data/loops_district/Kruispunten.geojson')

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
prices = np.fromfile('/home/rogier/PycharmProjects/Maxus/Warmtenet/branched_tree/prices.dat', dtype=int)
wvs = np.fromfile('/home/rogier/PycharmProjects/Maxus/Warmtenet/branched_tree/wv.dat', dtype=int)
# prices = np.random.randint(50, 60, len(points_with_house_and_source))
# wvs = np.random.randint(50, 60, len(points_with_house_and_source))
for index, price, wv in zip(points_with_house_and_source.index, prices, wvs):
    if points_with_house_and_source.loc[index, 'pandidentificatie'] != 'BRON':
        price_threshold.append(price)
        warmtevraag.append(wv)
    else:
        price_threshold.append(999)
        warmtevraag.append(0)

# select houses above price threshold
points_with_house_and_source['threshold'] = price_threshold
points_with_house_and_source['warmtevraag'] = warmtevraag
points_with_house_and_source = points_with_house_and_source[points_with_house_and_source.threshold >= 57]

# load junctions and put all selected points in dataframe
points = pd.concat([points_with_house_and_source, junctions], ignore_index=True, sort=True)

# load roads
roads = gpd.read_file('/home/rogier/PycharmProjects/Maxus/Warmtenet/data/loops_district/Wegen.geojson')

# see what point are so near to one another that they should be treated as one
points_unique_geometry = create_unique_points_and_merge_panden(points)

# income
income = [calculate_revenue(index, points_unique_geometry, points_with_house_and_source)
          for index, row in points_unique_geometry.iterrows()]
points_unique_geometry['income'] = income


# points_unique_geometry = gpd.read_file('./point_unique_geometry.geojson')
connections = get_all_connections(roads=roads, points=points_unique_geometry)

connected_points = get_all_connected_points(connections=connections, points=points_unique_geometry)

# assign costs to connections
connections['A'] = connected_points[:, 0]
connections['B'] = connected_points[:, 1]
connections['costs'] = connections.geometry.length * 100

# both directions
connections_by_points = connections.set_index(['A', 'B'])
connections_by_points_reverse = connections.set_index(['B', 'A'])
conns_both_directions = connections_by_points.append(connections_by_points_reverse)

connections_dict = {}
for k, g in conns_both_directions.groupby(level=(0, 1)):
    connections_dict[k] = g.to_dict('r')

# see what points can connect to other points
p2p = store_connected_points_per_point(connections)
# cut_loops
print(p2p)

index_bron = points_unique_geometry[points_unique_geometry['pandidentificatie'] == 'BRON'].index.values[0]


def find_loops(p2p, index_bron, connections, connections_dict, plot=False):
    paths = {}
    x = 0
    paths[x] = [index_bron]
    active_keys = [x]
    cuts = []
    index_cuts = []
    loops = []

    while len(active_keys) > 0:
        rounds = active_keys.copy()
        print(f'cuts: {len(cuts)}, active_keys: {len(active_keys)}, loops: {len(loops)}')
        if plot:
            plot_path_loop(paths={k: v for k, v in paths.items() if k in active_keys}, connections=connections, points=points_unique_geometry)
        for key in rounds:
            path_orig = paths[key].copy()
            p_conn = np.array(p2p[path_orig[-1]])
            if len(p_conn) == 1 and path_orig[-1] != index_bron:
                active_keys.remove(key)
            else:
                if len(p_conn) == 1:
                    paths[key].append(p_conn[0])
                else:
                    # remove if loop detected
                    if len(p_conn[p_conn == path_orig[-2]]) == 2:
                        p_index = path_orig[-2]
                        loop = path_orig[path_orig.index(p_index):] + [p_index]
                        if (loop not in loops) and (loop.reverse() not in loops):
                            loops.append(loop)
                            cut, index_cut = find_cut(loop, connections_dict, index_cuts)
                            cuts.extend(cut)
                        if len(p_conn[p_conn != path_orig[-2]]) == 0:
                            active_keys.remove(key)

                    for i, p_index in enumerate(p_conn[p_conn != path_orig[-2]]):
                        if len(list(filter(lambda x: (x[0] == path_orig[-1] and x[1] == p_index) or (x[1] == path_orig[-1] and x[0] == p_index), index_cuts))) == 0:
                            if i == 0:  # continue existing path
                                if p_index not in paths[key]:
                                    paths[key].append(p_index)
                                else:
                                    active_keys.remove(key)
                                    loop = path_orig[path_orig.index(p_index):] + [p_index]
                                    if (loop not in loops) and (loop.reverse() not in loops):
                                        loops.append(loop)
                                        cut, index_cut = find_cut(loop, connections_dict, index_cuts)
                                        cuts.extend(cut)

                            else:   # start new path if not loop
                                x += 1
                                paths[x] = path_orig + [p_index]
                                if p_index not in path_orig:
                                    active_keys.append(x)
                                else:
                                    loop = path_orig[path_orig.index(p_index):] + [p_index]
                                    if (loop not in loops) and (loop.reverse() not in loops):
                                        loops.append(loop)
                                        cut, index_cuts = find_cut(loop, connections_dict, index_cuts)
                                        cuts.extend(cut)
                        else:
                            if key in active_keys:
                                active_keys.remove(key)
    return cuts, len(loops)
        # for key in active_keys:

def find_cut(loop, connections_dict, index_cuts_done):

    cuts = []
    x = 0
    paths = {x: []}
    roads = {x: []}
    index_segment = {x: []}
    index_cuts = []

    for i in range(len(loop) - 1):
        original_keys = paths.copy().keys()
        for key in original_keys:
            segments = connections_dict[(loop[i], loop[i+1])]
            if len(segments) > 1:
                for j, segment in enumerate(segments):
                    if j == 0:
                        paths[key].append(segment['costs'])
                        roads[key].append(segment['geometry'])
                        index_segment[key].append(j)
                    else:
                        x += 1
                        paths[x] = paths[key][:-1] + [segment['costs']]
                        roads[x] = roads[key][:-1] + [segment['geometry']]
                        index_segment[x] = index_segment[key][:-1] + [j]
            else:
                paths[key].append(segments[0]['costs'])
                roads[key].append(segments[0]['geometry'])
                index_segment[key].append(0)

    for i, path in paths.items():
        index_exp = np.argmax(path)
        cuts.append(roads[i][index_exp])
        index_cuts.append((loop[index_exp], loop[index_exp + 1], index_segment[i][index_exp]))


    new_cuts = []
    for ind, cut in zip(index_cuts, cuts):
        if ind not in index_cuts_done:
            new_cuts.append(cut)
            index_cuts_done.append(ind)

    return new_cuts, index_cuts_done

def plot_path_loop(connections, paths, points):
    f, ax = plt.subplots()
    points.plot(color='grey', alpha=0.2, ax=ax)
    connections.plot(ax=ax)
    for k, path in paths.items():
        points.loc[path].plot(ax=ax)
    plt.show()

def plot_loop(new_connections, points):
    f, ax = plt.subplots()
    new_connections.plot(ax=ax)
    # for k, v in points.iterrows():
    #     plt.annotate(s=k, xy=v.geometry.coords[:][0], fontsize = 10)
    # roads.loc[roads_to_plot].to_file('~/PycharmProjects/Maxus/Warmtenet/data/loops_district/output_cut_network.geojson', driver='GeoJSON')
    points.plot(ax=ax, color='r')
    plt.show()


cuts, number_of_loops = find_loops(p2p, index_bron, connections, connections_dict, plot=False)

mask_connections = []
for i, conn in connections.iterrows():
    if conn.geometry in cuts:
        mask_connections.append(False)
    else:
        mask_connections.append(True)

new_connections = connections[mask_connections]
# plot_loop(new_connections, points_unique_geometry)
iteration = 0
print(cuts)

while number_of_loops > 0:
    print(f'iteration {iteration}:, {number_of_loops} loops detected')
    iteration += 1
    new_connected_points = get_all_connected_points(new_connections, points_unique_geometry)
    p2p = store_connected_points_per_point(new_connections)

    cuts2, number_of_loops = find_loops(p2p, index_bron, new_connections, connections_dict, plot=False)
    cuts = cuts + cuts2

    mask_connections = []
    for i, conn in connections.iterrows():
        if conn.geometry in cuts:
            mask_connections.append(False)
        else:
            mask_connections.append(True)

    new_connections = connections[mask_connections]
    # plot_loop(new_connections, points_unique_geometry)

new_connected_points = get_all_connected_points(new_connections, points_unique_geometry)
p2p = store_connected_points_per_point(new_connections)

# plot income

# f, ax = plt.subplots()
# new_connections.plot(ax=ax)
# points_unique_geometry.plot(ax=ax, column=points_unique_geometry['income'], cmap='YlOrRd', legend=True, vmin=0, vmax=10000)
# plt.show()

# start calculating which houses to drop

def plot_paths(paths: dict, connections, points, losing_points):
    f, ax = plt.subplots(1,2)
    points.plot(ax=ax[0], alpha=0.2, color ='grey')
    connections.plot(ax=ax[0], color='grey', alpha=0.5)
    for k, v in points.iterrows():
        plt.annotate(s=k, xy=v.geometry.coords[:][0], fontsize = 10)
    for k, path in paths.items():
        points.loc[path].plot(ax=ax[0])
    points.plot(ax=ax[1], alpha=0.2, color ='grey')
    connections.plot(ax=ax[1], color='grey', alpha=0.5)
    points.loc[losing_points].plot(ax=ax[1], color='r')
    plt.show()
#
# end_point_branches = {key: value for key, value in p2p.items() if len(value) == 1}
# end_point_branches.pop(index_bron)
# junctions_branched = {key: value for key, value in p2p.items() if len(value) > 2}
# junctions_branched_status = {key: [False for v in value] for key, value in junctions_branched.items()}
# junctions_branched_income = {key: [0 for v in value] for key, value in junctions_branched.items()}
# junctions_branched_cost = {key: [0 for v in value] for key, value in junctions_branched.items()}
# junctions_branched_profit = {key: [0 for v in value] for key, value in junctions_branched.items()}
# junctions_branched_points = {key: [[] for v in value] for key, value in junctions_branched.items()}
# junctions_merging_status = {key: False for key, value in junctions_branched.items()}
#
# paths = {}
# income = {}
# cost = {}
# profit = {}
# finished_points = []
# losing_points = []
#
# for x, key in enumerate(end_point_branches.keys()):
#     paths[x] = [key]
# x = len(paths)
#
# active_keys = list(paths.keys())
#
# new_connections_by_points = new_connections.set_index(['A', 'B'])
# new_connections_by_points_reverse = new_connections.set_index(['B', 'A'])
# new_conns_both_directions = new_connections_by_points.append(new_connections_by_points_reverse)
#
#
# while len(active_keys) > 0:
#     rounds = active_keys.copy()
#     plot_paths(paths=paths, connections=new_connections, points=points_unique_geometry, losing_points=losing_points)
#     for key in rounds:
#         if len(paths[key]) > 1:
#             income[key] += calculate_revenue(paths[key][-2], points_unique_geometry, points_with_house_and_source)
#             cost[key] += new_conns_both_directions.loc[paths[key][-1], paths[key][-2]]['costs']
#             profit[key] = income[key] - cost[key]
#         else:
#             profit[key] = 0
#             cost[key] = 0
#             income[key] = 0
#
#         if profit[key] < 0:
#             print('popping key because of profit:', key)
#             active_keys.remove(key)
#             finished_points.extend(paths[key][:-1])
#             losing_points.extend(paths[key][:-1])
#             if len(p2p[paths[key][-1]]) > 2:
#                 p_index = p2p[paths[key][-1]].index(paths[key][-2])
#                 junctions_branched_status[paths[key][-1]][p_index] = True
#
#             x += 1
#             paths[x] = [paths[key][-1]]
#             active_keys.append(x)
#
#         else:
#             if len(p2p[paths[key][-1]]) == 1:
#                 if not p2p[paths[key][-1]] in paths[key]:
#                     next_point = p2p[paths[key][-1]]
#                     paths[key].extend(next_point)
#                 else:
#                     print('BRON found')
#                     finished_points.extend(paths[key])
#                     active_keys.remove(key)
#                     print('profit:', profit[key])
#
#             elif len(p2p[paths[key][-1]]) == 2:
#                 next_point = None
#                 for point in p2p[paths[key][-1]]:
#                     if point not in finished_points:
#                         if len(paths[key]) == 1:
#                             next_point = point
#                         elif point != paths[key][-2]:
#                             next_point = point
#
#                 if next_point is not None:
#                     paths[key].append(next_point)
#                 else:
#                     raise ValueError(f'couldnt find next point for {paths[key][-1]}')
#
#             elif len(p2p[paths[key][-1]]) > 2:
#
#                 if (sum([p in paths[key] for p in p2p[paths[key][-1]]]) + sum([p in finished_points for p in p2p[paths[key][-1]]])) > 1\
#                         and junctions_merging_status[paths[key][-1]]:
#                     directions = p2p[paths[key][-1]]
#                     index = junctions_branched_status[paths[key][-1]].index(False)
#                     next_point = directions[index]
#                     paths[key].append(next_point)
#                 else:
#                     print('popping key because of junction:', key)
#                     active_keys.remove(key)
#                     if len(paths[key]) > 1:
#                         finished_points.extend(paths[key][:-1])
#                         p_index = p2p[paths[key][-1]].index(paths[key][-2])
#                         junctions_branched_status[paths[key][-1]][p_index] = True
#                         junctions_branched_income[paths[key][-1]][p_index] = income[key]
#                         junctions_branched_cost[paths[key][-1]][p_index] = cost[key]
#                         junctions_branched_profit[paths[key][-1]][p_index] = profit[key]
#                         junctions_branched_points[paths[key][-1]][p_index] = paths[key][:-1]
#
#                     if len(junctions_branched_status[paths[key][-1]]) - sum(junctions_branched_status[paths[key][-1]]) == 1:
#                         junctions_merging_status[paths[key][-1]] = True
#                         x += 1
#                         paths[x] = [item for sublist in junctions_branched_points[paths[key][-1]] for item in sublist] + [paths[key][-1]]
#                         income[x] = sum(junctions_branched_income[paths[key][-1]])
#                         cost[x] = sum(junctions_branched_cost[paths[key][-1]])
#                         profit[x] = sum(junctions_branched_profit[paths[key][-1]])
#                         active_keys.append(x)
#                         keys_to_remove = [k for k, path in paths.items() if path[-1] == paths[key][-1] and k != x and k in active_keys]
#                         print(keys_to_remove)
#                         for k in keys_to_remove:
#                             print('popping key because of junction, only one path can continue:', k, paths[k])
#                             active_keys.remove(k)
#
# f, ax = plt.subplots(1, 2)
# points_unique_geometry.plot(ax=ax[0])
# points_unique_geometry.plot(ax=ax[1])
# points_unique_geometry.loc[losing_points].plot(ax=ax[0], color='r')
# points_unique_geometry.loc[finished_points].plot(ax=ax[1], color='g')
# new_connections.plot(ax=ax[0])
# new_connections.plot(ax=ax[1])
# plt.show()
#
# print(wvs)
# print(prices)

