import geopandas as gpd
import pandas as pd
import json
import numpy as np
import matplotlib.pyplot as plt


points = gpd.read_file(r'C:\Users\Rogier\OneDrive\Warmtenet\All_Points_copy.shp')
roads = gpd.read_file(r'C:\Users\Rogier\OneDrive\Warmtenet\assen_test\wegen_wijk.shp')
buildings = gpd.read_file(r'C:\Users\Rogier\OneDrive\Warmtenet\assen_test\shape\buildings.shp')

X = len(points) - 1
H = len(buildings) - 1
w = ['LT', 'MT', 'HT']


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



# read file
with open('results.json', 'r') as myfile:
    data=myfile.read()

# parse file
obj = json.loads(data)


A = np.zeros([len(points), len(points)], dtype=np.float32)
for i in range(0,len(points)):
    for j in range(0,len(points)):
        try:
            a =  obj['Solution'][1]['Variable']['A['+ str(i) + ',' + str(j) + ']']['Value']
            if a>0.001:
                print ('A [' + str(i) + ',' + str(j) + ']: ', a)
                A[i,j]= a

        except:
            pass

isotype=[]
for i in range(0,len(buildings)):
    for isolationtype in w:
        try:
            a = obj['Solution'][1]['Variable']['Type['+isolationtype + ',' + str(i) + ']']['Value']
            if a>0.5:
                isotype.append(isolationtype)
                print ('Type['+isolationtype + ',' + str(i) + ']:', a)
        except:
            pass



AreaTube=[]
for row in PointsInRoadsShort:
    AreaTube.append(A[int(row[0]),int(row[1])] + A[int(row[1]),int(row[0])])
roads['width']=AreaTube
buildings['isotype']=isotype

f, ax = plt.subplots()

# roads.iloc[i,:].plot()
#
# df= gpd.GeoDataFrame(data = roads.iloc[i].values, columns = roads.iloc[i].index)


roads.plot(column='width', vmin=0, vmax=1.5, cmap='hot_r', ax=ax, legend=True)
# for i in range(0, len(roads)):
#
#     roads.iloc[i].to_frame().plot( column='width', ax=ax, linewidth= roads['width'][i])

points.plot(ax=ax)

roadPalette = {'HT': '#E85921',
               'MT': '#E8D715',
               'LT': '#BFE815'}

for ctype, data in buildings.groupby('isotype'):
    # Define the color for each group using the dictionary
    color = roadPalette[ctype]
    # Plot each group using the color defined above
    data.plot(color=color,
              ax=ax,
              label = ctype)


ax.set_axis_off()
plt.show()


# for i in range(1,16):
#     for j in range(1, 16):
#         try:
#             a =  obj['Solution'][1]['Variable']['Link_built['+ str(i) + ','+ str(j) +']']['Value']
#             if a>0.01:
#                 print('Link ['+ str(i)+','+str(j) +']: ',  obj['Solution'][1]['Variable']['Link_built['+ str(i) + ','+ str(j) +']']['Value'])
#
#         except:
#             pass
#
# #link built boolean
# for i in range(1,16):
#     for j in range(1, 16):
#         try:
#             a =  obj['Solution'][1]['Variable']['Link_Boolean['+ str(i) + ','+ str(j) +']']['Value']
#             if a>0.01:
#                 print('Link Boolean ['+ str(i)+','+str(j) +']: ',  round(a))
#
#         except:
#             pass