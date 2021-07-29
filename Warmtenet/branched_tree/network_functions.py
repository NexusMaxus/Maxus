import geopandas as gpd
from typing import Dict, List
import numpy as np
import pandas as pd
from shapely.geometry import MultiPoint, LineString
from utils import split_line_with_points

def create_unique_points_and_merge_panden(points: gpd.GeoDataFrame) -> gpd.GeoDataFrame:
    """

    Parameters
    ----------
    points

    Returns
    -------

    """
    same_points = _find_overlapping_points_in_network(points)
    points_unique_geometry = _merge_buildings_to_create_unique_points(points, same_points)

    # in case of MultiPoints type instead of Point type
    geoms = []
    for geom in points_unique_geometry.geometry:
        if isinstance(geom, MultiPoint):
            geoms.append(geom.centroid)
        else:
            geoms.append(geom)

    if not all(gpd.GeoSeries(geoms) == points_unique_geometry.geometry.reset_index(drop=True)):
        points_unique_geometry.geometry = geoms

    return points_unique_geometry


def _find_overlapping_points_in_network(points: gpd.GeoDataFrame) -> Dict[int, List[int]]:
    """
    seeks unique points in network

    Parameters
    ----------
    points:
        geodataframe with all points of the network

    Returns
    -------

    """
    unique_points = []
    same_points = {}
    for index, point in points.iterrows():
        point_list = []
        for index2, point2 in points.iterrows():
            if index2 not in unique_points:
                if point.geometry.distance(point2.geometry) < 1e-1 and index != index2:
                    point_list.append(index2)
        if len(point_list) == 0:
            unique_points.append(index)
        else:
            same_points[index] = point_list

    return same_points


def _merge_buildings_to_create_unique_points(points: gpd.GeoDataFrame, same_points: Dict[int, List[int]]) -> gpd.GeoDataFrame:
    # get all panden together on that single point
    panden = {}
    to_remove = []
    for index, point in points.iterrows():
        eigen_pand = points.loc[index, 'pandidentificatie']
        if index in same_points.keys():
            andere_panden = [i for i in points.loc[same_points[index], 'pandidentificatie'].values if i is not None]
            if eigen_pand is not None:
                andere_panden.append(eigen_pand)
            if not andere_panden:
                panden[index] = str([])
            else:
                panden[index] = str(andere_panden)
            to_remove.extend(same_points[index])
            for key in same_points[index]:
                same_points.pop(key)
        else:
            if eigen_pand is not None:
                panden[index] = str([eigen_pand])
            else:
                panden[index] = str([])

    alle_panden = pd.Series(panden, name='panden')
    points_unique_geometry = pd.concat([points, alle_panden], axis=1)
    points_unique_geometry = points_unique_geometry.drop(to_remove)
    print('prepared all unique point geometries')
    return points_unique_geometry


def find_long_roads(roads: gpd.GeoDataFrame, points: gpd.GeoDataFrame, print_roads_stats = False) -> Dict[int, List[int]]:
    """
    see what roads contain multiple points (roads that do not go only from one to another point

    Parameters
    ----------
    roads
    points
    print_roads_stats

    Returns
    -------

    """

    long_roads = {}
    road_point_count = np.zeros(roads.shape[0], dtype=int)

    for i, road in roads.geometry.items():
        x = 0
        p_array = []
        for j, point in points.geometry.items():
            if road.distance(point) < 1e-1:
                x += 1
                p_array.append(j)
        if x > 2:
            long_roads[i] = p_array
        road_point_count[i] = x

    if print_roads_stats:
        print(road_point_count)

    return long_roads


def split_long_roads(long_roads:  Dict[int, List[int]], roads: gpd.GeoDataFrame, points: gpd.GeoDataFrame) -> List[LineString]:
    """
    split roads that contain multiple points in smaller parts

    Parameters
    ----------
    long_roads
    roads
    points

    Returns
    -------

    """
    smaller_parts = []
    for road_number, segment in long_roads.items():
        points_in_long_road = []
        distance = []
        for j in segment:
            distance.append(roads.geometry[road_number].project(points.geometry[j]))
            points_in_long_road.append(points.geometry[j])

        df_points_in_road = pd.DataFrame({'dis': distance})
        points_gpd = gpd.GeoDataFrame(df_points_in_road, crs=roads.crs, geometry=points_in_long_road)
        points_gpd.sort_values('dis', inplace=True)
        points_gpd.reset_index(drop=True, inplace=True)
        smaller_parts.extend(split_line_with_points(roads.geometry[road_number], points_gpd.geometry[1:-1]))
    return smaller_parts


def get_all_connections(roads: gpd.GeoDataFrame, points: gpd.GeoDataFrame) -> gpd.GeoDataFrame:
    """
    find long roads and split them and merge them

    Parameters
    ----------
    roads
    points

    Returns
    -------

    """
    long_roads = find_long_roads(roads=roads, points=points)
    smaller_parts = split_long_roads(long_roads, roads=roads, points=points)

    small_roads = roads.drop(axis=0, index=long_roads.keys())
    gdf_long_roads_split = gpd.GeoDataFrame(crs=roads.crs, geometry=smaller_parts)
    gdf_small_roads = gpd.GeoDataFrame(crs=roads.crs, geometry=small_roads.geometry)
    connections = pd.concat([gdf_small_roads, gdf_long_roads_split], axis=0, ignore_index=True, sort=False)
    print('prepared all connections')
    return connections


def get_all_connected_points(connections, points):
    """
    makes a N-2 array with for each connection the connected points

    Parameters
    ----------
    connections
    points

    Returns
    -------

    """
    points_in_road_short = np.ones((len(connections.geometry), 2), dtype=np.int16) * -1

    for i, road in connections.geometry.items():
        x = 0
        for j, point in points.geometry.items():
            if road.distance(point) < 1e-1:
                points_in_road_short[i, x] = j
                x += 1

    return points_in_road_short


def get_unique_junctions(all_points, points_with_house_and_source):
    unique_point_index = []
    for index, point in all_points.iterrows():
        for index2, point2 in points_with_house_and_source.iterrows():
             if point.geometry.distance(point2.geometry) < 1e-1:
                 break
        if index2 == points_with_house_and_source.index[-1]:
            unique_point_index.append(index)

    junctions = all_points.loc[unique_point_index]
    return junctions


def store_connected_points_per_point(connected_points, connections):
    p2p = {}
    cost_streets = {}
    streets = {}

    for i in range(len(connected_points)):
        if np.logical_and(connected_points[i, 0] != -1, connected_points[i, 1] != -1):
            if connected_points[i, 0] not in p2p.keys():
                p2p[connected_points[i, 0]] = [connected_points[i, 1]]
                cost_streets[(connected_points[i, 0], connected_points[i, 1])] = connections.geometry[i].length * 100
                streets[(connected_points[i, 0], connected_points[i, 1])] = i
            else:
                p2p[connected_points[i, 0]].append(connected_points[i, 1])
                cost_streets[(connected_points[i, 0], connected_points[i, 1])] = connections.geometry[i].length * 100
                streets[(connected_points[i, 0], connected_points[i, 1])] = i

            if connected_points[i, 1] not in p2p.keys():
                p2p[connected_points[i, 1]] = [connected_points[i, 0]]
                cost_streets[(connected_points[i, 1], connected_points[i, 0])] = connections.geometry[i].length * 100
                streets[(connected_points[i, 1], connected_points[i, 0])] = i
            else:
                p2p[connected_points[i, 1]].append(connected_points[i, 0])
                cost_streets[(connected_points[i, 1], connected_points[i, 0])] = connections.geometry[i].length * 100
                streets[(connected_points[i, 1], connected_points[i, 0])] = i

    return p2p, streets, cost_streets