'''
Code that reads in the coordinates of the rectangles and adds 1.5 km to each corner.
Stores the list of coordinates as a txt file

Code by Johanna Kauffert
'''

from pandas import *
#read csv with coordinates information
data = read_csv("/exports/csce/datastore/geos/groups/MSCGIS/s1937352/Uganda/Coordinates_T_PolygonsWKT.csv", sep= ";")
print(data.head())
#create empty list
wkt_list = []
#loop over all rows and add the overlab of 1.5 km
for i, row in data.iterrows(): 
    row0 = row[0]-0.015
    row1 = row[1]-0.015
    row2 = row[2]-0.015
    row3 = row[3]+0.015
    row4 = row[4]+0.015
    row5 = row[5]+0.015
    row6 = row[6]+0.015
    row7 = row[7]-0.015
    #create nested list and append to outer list
    wkt = [[row0, row1],[row2, row3], [row4, row5], [row6, row7],[row0, row1]]
    wkt_list.append(wkt)

#dumb list into a text file
with open("/exports/csce/datastore/geos/groups/MSCGIS/s1937352/Uganda/wkt", 'w') as n:
    n.write(str(wkt_list))