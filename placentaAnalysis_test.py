import numpy as np
import math
import pandas as pd
from placentaAnalysisFunctions import *
import placentagen as pg

######
# Tests all Placenta Analysis Functions with a test case
# Inputs: file test_tree.csv
# Outputs: Types to screen whether each function provides the correct output, by comparing it against hand calculated solutions
######

#Load file
with open('test_tree.csv') as csvDataFile:
    data_file = pd.read_csv(csvDataFile, usecols=['Skeleton ID', 'Branch length', 'V1 x', 'V1 y', 'V1 z', 'V2 x', 'V2 y', 'V2 z', 'Euclidean distance', 'average intensity (inner 3rd)'])
    data_file.columns = ['SkeletonID', 'Branchlength', 'V1x', 'V1y', 'V1z', 'V2x', 'V2y', 'V2z', 'Euclideandistance', 'averageintensityinner3rd']

#Get Skeleton Info
print('\nTest Sort Elems')
geom=sort_data(data_file)

nodes_true=np.array([[0.,0.,1.],[0., 0., 0.],[0.,1.,0.],[0.,-1.,0.],[0.,1.,-1.],[0.,-1.,-1.],[-3,-3,-3],[1.,2.,-1.],[-1.,2.,-1.],[1.,-2.,-1.],[-1.,-2.,-1.]])
if np.array_equal(geom['nodes'], nodes_true):
    print('Nodes correct')
else:
    print('Nodes INCORRECT')

elems_true=np.array([[0.,0.,1],[1, 1, 2],[2,1,3],[3,2,4],[4,3,5],[5,4,7],[6,4,8],[7,5,9],[8,5,10]])
if np.array_equal(geom['elems'], elems_true):
    print('Elems correct')
else:
    print('Elems INCORRECT')

#Get order
print('\nTest Strahler Order')
inlet_loc=np.array([0,0,1])
geom= get_strahler_order(geom, inlet_loc)

order_true=np.array([3.,2.,2.,2.,2.,1.,1.,1.,1.])
#if np.array_equal(geom['strahler_order'], order_true):
#    print('Order correct')
#else:
#    print('Order incorrect')

#Get branch angles
print('\nTest Branch Angles')
orders=pg.evaluate_orders(geom['nodes'],geom['elems'])
geom['branch_angles']=find_branch_angles(geom['nodes'], geom['elems'], orders['generation'])

angles_true=np.array([-1,np.pi/2,np.pi/2,-1,-1,np.pi/2,np.pi/2,np.pi/2,np.pi/2])
if np.array_equal(geom['branch_angles'], angles_true):
    print('Angles correct')
else:
    print('Angles incorrect')

#Output Skeleton Info
print('\nOutput Table')
table=summary_statistics(orders['strahler'],orders['generation'], geom['length'],geom['euclidean length'], geom['radii'],geom['branch_angles'],geom['branch_angles'],geom['branch_angles'])
#table_true=np.array([[1,4,4,1.15, math.sqrt(2),4.5,(1.2/20+1.1/16),1.15/math.sqrt(2), np.pi/2],[2,4,2, 2.3,2,9,((1.1/20+1.2/20+1.1/16+1.2/16)/4),1.15, np.pi/2],[3, 1,1, 1.1,1,20, 1.1/40, 1.1, np.nan]])

#table=np.float16(table) #to discount small errors associated with precision of floating point numbers
#table_true=np.float16(table_true)

#np.testing.assert_equal(table, table_true)
#print('Table correct')

#PG
#3d plots
plotVasculature3D(geom['nodes'], geom['elems'], geom['branch_angles'],geom['radii'])