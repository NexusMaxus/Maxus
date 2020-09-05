import geopandas as gpd
import json
import numpy as np
import matplotlib.pyplot as plt


points = gpd.read_file(r'C:\Users\Rogier\OneDrive\Warmtenet\All_Points_copy.shp')
roads = gpd.read_file(r'C:\Users\Rogier\OneDrive\Warmtenet\assen_test\wegen_wijk.shp')
buildings = gpd.read_file(r'C:\Users\Rogier\OneDrive\Warmtenet\assen_test\shape\buildings.shp')

X = len(points) - 1
H = len(buildings) - 1



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
with open('results_9_1temperatuur_wijk.json', 'r') as myfile:
    data=myfile.read()

# parse file
obj = json.loads(data)


A = np.zeros([len(points), len(points)], dtype=np.float32)
for i in range(0,len(points)):
    for j in range(0,len(points)):
        try:
            a =  obj['Solution'][1]['Variable']['A_exist['+ str(i) + ',' + str(j) + ']']['Value']
            if a>0.001:
                print ('A_exist [' + str(i) + ',' + str(j) + ']: ', a)
                A[i,j]= a

        except:
            pass

connected=[]
for i in range(0,len(buildings)):
        try:
            a = obj['Solution'][1]['Variable']['Conn['+ str(i) + ']']['Value']
            if a>0.9:
                print('Conn['+ str(i) + ']:', a)
            connected.append(a)
        except:
            connected.append(0)



AreaTube=[]
for row in PointsInRoadsShort:
    AreaTube.append(A[int(row[0]),int(row[1])] + A[int(row[1]),int(row[0])])
roads['width']=AreaTube
buildings['Conn']=connected

f, ax = plt.subplots()

# roads.iloc[i,:].plot()
#
# df= gpd.GeoDataFrame(data = roads.iloc[i].values, columns = roads.iloc[i].index)


roads.plot(column='width', cmap='hot_r', ax=ax, legend=True)
# for i in range(0, len(roads)):
#
#     roads.iloc[i].to_frame().plot( column='width', ax=ax, linewidth= roads['width'][i])

points.plot(ax=ax)


buildings.plot(column='Conn', vmin=0, vmax=1.5, cmap='Greys', ax=ax, edgecolor='grey')

# for i in range(0, len(roads)):
#

ax.set_axis_off()
plt.show()

