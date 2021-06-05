import geopandas as gpd
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from pandas.io.json import json_normalize
import json
import io
from utils import split_line_with_points


crs = {'init': 'epsg:28992'}


points_with_house_and_source = gpd.read_file('../data/totaalgebied/Aansluitpunten.geojson')
points_with_house = points_with_house_and_source[points_with_house_and_source['pandidentificatie'] != 'BRON']
points_with_house.loc[:, 'pandidentificatie'] = [str(p[1:]) for p in points_with_house['pandidentificatie']]

junctions = gpd.read_file('../data/totaalgebied/Kruispunten.geojson')
points = pd.concat([points_with_house_and_source, junctions], ignore_index=True, sort=False)
roads = gpd.read_file('../data/totaalgebied/Wegen.geojson')

# points_unique_geometry = gpd.read_file('./deelgebied/Kruispunten_Ap_deel.geojson')
points_unique_geometry = points[~points.geometry.duplicated()].reset_index()
points_unique_geometry.geometry = [geom.centroid for geom in points_unique_geometry.geometry]

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

print(p2p)


