import geopandas as gpd
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from pandas.io.json import json_normalize
import json
import io

import random
from branched_tree.network_functions import create_unique_points_and_merge_panden, \
    get_all_connections, get_all_connected_points, store_connected_points_per_point

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
for index in points_with_house_and_source.index:
    if points_with_house_and_source.loc[index, 'pandidentificatie'] != 'BRON':
        price_threshold.append(random.randint(50, 60))
    else:
        price_threshold.append(999)

# select houses above price threshold
points_with_house_and_source['threshold'] = price_threshold
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

    while len(active_keys) > 0:
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


def plot_loop(roads, cuts):
    street_index = [streets[(cut[0], cut[1])] for cut in cuts]
    f, ax = plt.subplots()
    roads_to_plot = list(set(roads.index) - set(street_index))
    roads.loc[roads_to_plot].plot(ax=ax)
    # roads.loc[roads_to_plot].to_file('~/PycharmProjects/Maxus/Warmtenet/data/loops_district/output_cut_network.geojson', driver='GeoJSON')
    points_unique_geometry.plot(ax=ax, color='r')
    plt.show()

cuts = find_loops(p2p, index_bron)
plot_loop(connections, cuts)
print(cuts)




