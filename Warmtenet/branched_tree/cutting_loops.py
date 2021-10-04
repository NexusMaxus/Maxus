import geopandas as gpd
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import random
from branched_tree.network_functions import merge_buildings, create_unique_points_and_merge_panden, \
    get_all_connections, get_all_connected_points, store_connected_points_per_point
from branched_tree.profit_functions import calculate_revenue
import seaborn as shs
import json
shs.set_theme('paper')
price = 55

points_with_house_and_source = gpd.read_file('/home/rogier/PycharmProjects/Maxus/Warmtenet/data/loops_district/Aansluitpunten.geojson')
all_points = gpd.read_file('/home/rogier/PycharmProjects/Maxus/Warmtenet/data/loops_district/Kruispunten.geojson')

kruispunten_mask = []
for cell in all_points.id:
    if isinstance(cell, str):
        kruispunten_mask.append('kruispunt' in cell)
    else:
        kruispunten_mask.append(False)

junctions = all_points[kruispunten_mask]

st_gids = []
for index, row in junctions['straten'].items():
    st_gids.append([street['streetid'] for street in json.loads(row)])
junctions['st_gid_list'] = st_gids


points_with_house = points_with_house_and_source[points_with_house_and_source['pandidentificatie'] != 'BRON']
points_with_house.loc[:, 'pandidentificatie'] = [str(p[1:]) for p in points_with_house['pandidentificatie']]
points_with_house_and_source['st_gid_list'] = [[st_gid] for st_gid in points_with_house_and_source.st_gid]

# generate price for each house
price_threshold = []
warmtevraag = []
# prices = np.fromfile('/home/rogier/PycharmProjects/Maxus/Warmtenet/branched_tree/prices.dat', dtype=int)
# wvs = np.fromfile('/home/rogier/PycharmProjects/Maxus/Warmtenet/branched_tree/wv.dat', dtype=int)
prices = np.random.randint(50, 60, len(points_with_house_and_source))
wvs = np.random.randint(10, 120, len(points_with_house_and_source))
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
points_with_house_and_source = points_with_house_and_source[points_with_house_and_source.threshold >= price]

# load junctions and put all selected points in dataframe
points = pd.concat([points_with_house_and_source, junctions], ignore_index=True, sort=True)

# load roads
roads = gpd.read_file('/home/rogier/PycharmProjects/Maxus/Warmtenet/data/loops_district/Wegen.geojson')

# see what point are so near to one another that they should be treated as one
points_unique_geometry = merge_buildings(points, price).reset_index()

# points_unique_geometry = gpd.read_file('./point_unique_geometry.geojson')
connections = get_all_connections(roads=roads, points=points_unique_geometry)

connected_points = get_all_connected_points(connections=connections, points=points_unique_geometry)

# assign costs to connections
connections['A'] = connected_points[:, 0]
connections['B'] = connected_points[:, 1]
connections['costs'] = connections.geometry.length * 100
connections = connections.loc[connections.groupby(['A', 'B']).costs.idxmin()]

# both directions
connections_indexed = connections.set_index(['A', 'B'])
connections_indexed_reverse = connections.set_index(['B', 'A'])
conns_both_directions = connections_indexed.append(connections_indexed_reverse)
connections_dict = conns_both_directions.to_dict()


# see what points can connect to other points
p2p = store_connected_points_per_point(connections)
# cut_loops
print(p2p)

index_bron = points_unique_geometry[points_unique_geometry['pandidentificatie'] == 'BRON'].index.values[0]


def make_tree(p2p, index_bron, connections, connections_dict, plot=False):
    paths = {}
    x = 0
    paths[x] = [index_bron]
    cuts = []
    loops = []

    while len(paths) > 0:
        keys_previous_round = list(paths.keys()).copy()
        print(f'cuts: {len(cuts)}, paths: {len(keys_previous_round)}, loops: {len(loops)}')
        if plot:
            plot_path_loop(paths={k: v for k, v in paths.items() if k in active_keys}, connections=connections, points=points_unique_geometry)
        for key in keys_previous_round:
            path_orig = paths[key].copy()
            p_conn = np.array(p2p[path_orig[-1]])

            # is it a dead end?
            if len(p_conn) == 1:
                # is it the source?
                if path_orig[-1] == index_bron:
                    paths[key].append(p_conn[0])
                else:
                    del paths[key]
            else:
                for i, p_to in enumerate(p_conn[p_conn != path_orig[-2]]):
                    if (p_to, path_orig[-1]) not in cuts:
                        if i == 0 and (p_to not in path_orig):
                            # extend path
                            paths[key].append(p_to)
                        elif i > 0 and (p_to not in path_orig):
                            # make new path and extend
                            x += 1
                            paths[x] = path_orig + [p_to]
                        else:
                            define_loop_and_cut(path_orig, p_to, loops, cuts, connections_dict)
                            if i == 0:
                                del paths[key]
                    else:
                        if i == 0:
                            del paths[key]
                # all connection to go by are already cut

    return cuts, len(loops)
        # for key in active_keys:


def find_cut(loop, connections_dict):

    costs_segments_in_loop = []
    for i in range(len(loop) - 1):
        costs_segments_in_loop.append(connections_dict['costs'][loop[i], loop[i + 1]])

    index_exp = np.argmax(costs_segments_in_loop)
    return (loop[index_exp], loop[index_exp + 1]), (loop[index_exp + 1], loop[index_exp])


def define_loop_and_cut(path_orig, p_index, loops, cuts, connections_dict):
    loop = path_orig[path_orig.index(p_index):] + [p_index]
    if (loop not in loops) and (loop.reverse() not in loops):
        loops.append(loop)
        cut_pair = find_cut(loop, connections_dict)
        for cut in cut_pair:
            if cut not in cuts:
                cuts.append(cut)



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


cuts, number_of_loops = make_tree(p2p, index_bron, connections, connections_dict, plot=False)

mask = []
for index in connections_indexed.index.values:
    if index in cuts:
        mask.append(False)
    else:
        mask.append(True)

new_connections = connections_indexed[mask]
plot_loop(new_connections, points_unique_geometry)
print(cuts)

p2p = store_connected_points_per_point(new_connections.reset_index())

# plot income

f, ax = plt.subplots()
new_connections.plot(ax=ax)
points_unique_geometry.plot(ax=ax, column=points_unique_geometry['income'], cmap='YlOrRd', legend=True, vmin=0, vmax=10000)
plt.show()

# start calculating which houses to drop

def plot_paths(paths: dict, connections, points, losing_points):
    f, ax = plt.subplots(1,2)
    points.plot(ax=ax[0], alpha=0.2, color ='grey')
    connections.plot(ax=ax[0], color='grey', alpha=0.5)
    # for k, v in points.iterrows():
    #     plt.annotate(s=k, xy=v.geometry.coords[:][0], fontsize = 10)
    for k, path in paths.items():
        points.loc[path].plot(ax=ax[0])
    points.plot(ax=ax[1], alpha=0.2, color ='grey')
    connections.plot(ax=ax[1], color='grey', alpha=0.5)
    points.loc[losing_points].plot(ax=ax[1], color='r')
    plt.show()

end_point_branches = {key: value for key, value in p2p.items() if len(value) == 1}
end_point_branches.pop(index_bron)
junctions_branched = {key: value for key, value in p2p.items() if len(value) > 2}
junctions_branched_status = {key: [False for v in value] for key, value in junctions_branched.items()}
junctions_branched_income = {key: [0 for v in value] for key, value in junctions_branched.items()}
junctions_branched_cost = {key: [0 for v in value] for key, value in junctions_branched.items()}
junctions_branched_profit = {key: [0 for v in value] for key, value in junctions_branched.items()}
junctions_branched_points = {key: [[] for v in value] for key, value in junctions_branched.items()}
junctions_merging_status = {key: False for key, value in junctions_branched.items()}

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

new_connections_reversed = new_connections.reset_index().set_index(['B', 'A'])
new_conns_both_directions = new_connections.append(new_connections_reversed)
new_conns_dict = new_conns_both_directions.to_dict()

while len(active_keys) > 0:
    previous_keys = active_keys.copy()
    # plot_paths(paths=paths, connections=new_connections, points=points_unique_geometry, losing_points=losing_points)
    for key in previous_keys:
        if len(paths[key]) > 1:
            income[key] += calculate_revenue(paths[key][-2], points_unique_geometry, points_with_house_and_source)
            cost[key] += new_conns_dict['costs'][(paths[key][-1], paths[key][-2])]
            profit[key] = income[key] - cost[key]
        else:
            profit[key] = 0
            cost[key] = 0
            income[key] = 0

        if profit[key] < 0:
            print('popping key because of profit:', key)
            active_keys.remove(key)
            finished_points.extend(paths[key][:-1])
            losing_points.extend(paths[key][:-1])
            if len(p2p[paths[key][-1]]) > 2:
                p_index = p2p[paths[key][-1]].index(paths[key][-2])
                junctions_branched_status[paths[key][-1]][p_index] = True

            x += 1
            paths[x] = [paths[key][-1]]
            active_keys.append(x)

        else:
            if len(p2p[paths[key][-1]]) == 1:
                if paths[key][-1] != index_bron:
                    next_point = p2p[paths[key][-1]][0]
                    paths[key].append(next_point)
                else:
                    print('BRON found')
                    finished_points.extend(paths[key])
                    active_keys.remove(key)
                    print('profit:', profit[key])

            elif len(p2p[paths[key][-1]]) == 2:
                next_point = None
                for point in p2p[paths[key][-1]]:
                    if point not in finished_points:
                        if len(paths[key]) == 1:
                            next_point = point
                        elif point != paths[key][-2]:
                            next_point = point

                if next_point is not None:
                    paths[key].append(next_point)
                else:
                    raise ValueError(f'couldnt find next point for {paths[key][-1]}')

            elif len(p2p[paths[key][-1]]) > 2:

                if (sum([p in paths[key] for p in p2p[paths[key][-1]]]) + sum([p in finished_points for p in p2p[paths[key][-1]]])) > 1\
                        and junctions_merging_status[paths[key][-1]]:
                    directions = p2p[paths[key][-1]]
                    index = junctions_branched_status[paths[key][-1]].index(False)
                    next_point = directions[index]
                    paths[key].append(next_point)
                else:
                    print('popping key because of junction:', key)
                    active_keys.remove(key)
                    if len(paths[key]) > 1:
                        finished_points.extend(paths[key][:-1])
                        p_index = p2p[paths[key][-1]].index(paths[key][-2])
                        junctions_branched_status[paths[key][-1]][p_index] = True
                        junctions_branched_income[paths[key][-1]][p_index] = income[key]
                        junctions_branched_cost[paths[key][-1]][p_index] = cost[key]
                        junctions_branched_profit[paths[key][-1]][p_index] = profit[key]
                        junctions_branched_points[paths[key][-1]][p_index] = paths[key][:-1]

                    if len(junctions_branched_status[paths[key][-1]]) - sum(junctions_branched_status[paths[key][-1]]) == 1:
                        junctions_merging_status[paths[key][-1]] = True
                        x += 1
                        paths[x] = [item for sublist in junctions_branched_points[paths[key][-1]] for item in sublist] + [paths[key][-1]]
                        income[x] = sum(junctions_branched_income[paths[key][-1]])
                        cost[x] = sum(junctions_branched_cost[paths[key][-1]])
                        profit[x] = sum(junctions_branched_profit[paths[key][-1]])
                        active_keys.append(x)
                        keys_to_remove = [k for k, path in paths.items() if path[-1] == paths[key][-1] and k != x and k in active_keys]
                        print(keys_to_remove)
                        for k in keys_to_remove:
                            print('popping key because of junction, only one path can continue:', k, paths[k])
                            active_keys.remove(k)

    if len(active_keys) == 0:
        plot_paths(paths=paths, connections=new_connections, points=points_unique_geometry, losing_points=losing_points)
        plt.show()

f, ax = plt.subplots(1, 2)
points_unique_geometry.plot(ax=ax[0])
points_unique_geometry.loc[losing_points].plot(ax=ax[0], color='r')
new_connections.plot(ax=ax[0])
connections.plot(ax=ax[1])
points_unique_geometry.plot(ax=ax[1], column=points_unique_geometry['income'], cmap='YlOrRd', legend=True, vmin=0, vmax=10000)
plt.show()

print(wvs)
print(prices)

