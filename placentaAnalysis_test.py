import placentagen as pg
import numpy as np
import pandas as pd
from placentaAnalysisFunctions import *

######
# Uses Test Case to Test Placenta Analysis Functions & Utilities
# Inputs: uses files test_tree.csv and test_tree_answers.csv
# Outputs: uses assert statements to verify code outputs match those in the answers file (which where manually calculated)
######

#Load test answers:
with open('test_tree_answers.csv') as csvDataFile:
    data_file = pd.read_csv(csvDataFile, usecols=['nx','ny','nz','e0','e1','e2','e0_ordered','e1_ordered','e2_ordered','angle','diamR','lengthR','order','generation1','generation2','e0_ordered_2','e1_ordered_2','e2_ordered_2'])
    data_file.columns = ['nx','ny','nz','e0','e1','e2','e0_ordered','e1_ordered','e2_ordered','angle','diamR','lengthR','order','generation1','generation2','e0_ordered_2','e1_ordered_2','e2_ordered_2']

#Convert answer files to arrays:
generation1_true=data_file.generation1.values
generation2_true=data_file.generation2.values
diam_ratio_true=data_file.diamR.values
length_ratio_true=data_file.lengthR.values
order_true=data_file.order.values
angles_true=data_file.angle.values
order_true=order_true[0:11]
generation1_true=generation1_true[0:11]
generation2_true=generation2_true[0:10]
angles_true=angles_true[0:10]
diam_ratio_true=diam_ratio_true[0:10]
length_ratio_true=length_ratio_true[0:10]

data_file=data_file.drop(['angle','diamR','lengthR','order','generation1','generation2'], axis=1)
data_file=data_file.values
nodes_true=data_file[:,0:3]
elems_unordered_true=data_file[:, 3:6]
elems_unordered_true=elems_unordered_true[0:11,:]
elems_ordered_true=data_file[:, 6:9]
elems_ordered_true=elems_ordered_true[0:11,:]
elems_pruned_true=data_file[:, 9:12]
elems_pruned_true=elems_pruned_true[0:10,:]

#Load test file:
with open('test_tree.csv') as csvDataFile:
    data_file = pd.read_csv(csvDataFile, usecols=['Skeleton ID', 'Branch length', 'V1 x', 'V1 y', 'V1 z', 'V2 x', 'V2 y', 'V2 z', 'Euclidean distance', 'average intensity (inner 3rd)'])
    data_file.columns = ['SkeletonID', 'Branchlength', 'V1x', 'V1y', 'V1z', 'V2x', 'V2y', 'V2z', 'Euclideandistance', 'averageintensityinner3rd']

#Test "sort_data" function:
print('\nTest Sort Elems')
geom=sort_data(data_file)

np.testing.assert_equal(geom['nodes'], nodes_true)
np.testing.assert_equal(geom['elems'], elems_unordered_true)

#Test "arrange_by_strahler_order" function:
print('\nTest Arrange by Order')
inlet_loc=np.array([0,0,2])
geom = arrange_by_strahler_order(geom, inlet_loc)

np.testing.assert_equal(geom['elems'], elems_ordered_true)

#Test "evaluate_orders" function:
print('\nTest Ordering')
Nc=find_maximum_joins(geom['elems'])
elem_connect=element_connectivity_1D(geom['nodes'], geom['elems'], Nc)
orders=evaluate_orders(geom['elems'], elem_connect)

strahler=orders['strahler']
strahler[0]=strahler[1] #assumed
orders['strahler']=strahler

np.testing.assert_equal(orders['strahler'], order_true)
np.testing.assert_equal(orders['generation'], generation1_true)

#Test "prune_by_order" function:
print('\nTest Prune and Re-Ordering')
threshold_order=2
geom=prune_by_order(geom, orders['strahler'], threshold_order)

Nc=find_maximum_joins(geom['elems'])
elem_connect=element_connectivity_1D(geom['nodes'], geom['elems'], Nc)
order2=evaluate_orders(geom['elems'], elem_connect)

np.testing.assert_equal(order2['generation'], generation2_true)
np.testing.assert_equal(geom['elems'], elems_pruned_true)

#Test angles
print('\nTest Branch Angles')
(geom['branch_angles'],geom['diam_ratio'],geom['length_ratio']) = find_branch_angles(geom, order2, elem_connect)

np.testing.assert_equal(np.around(geom['branch_angles'],2), np.around(angles_true,2))
np.testing.assert_equal(np.around(geom['diam_ratio'],2), np.around(diam_ratio_true,2))
np.testing.assert_equal(np.around(geom['length_ratio'],2), np.around(length_ratio_true,2))


#Test "summary_statistics" function:
print('\nOutput Table')
table=summary_statistics(order2, geom)

with open('test_tree_table.csv') as csvDataFile:
    data_file = pd.read_csv(csvDataFile, usecols=['ord','N','N_joined','Len','Len_e','Diam','Len/Diam','Tort','Angle'])
    data_file.columns = ['ord','N','N_joined','Len','Len_e','Diam','Len/Diam','Tort','Angle']
    table_true=data_file.values[0:4,:]

np.testing.assert_equal(np.around(table,2), np.around(table_true,2))

#Output plot and files for test case:
plot_vasculature_3d(geom['nodes'], geom['elems'], geom['branch_angles'],geom['radii'])

output=1
if output:
    pg.export_ex_coords(geom['nodes'], 'placenta', 'test_tree', 'exnode')
    pg.export_exelem_1d(geom['elems'], 'placenta', 'test_tree')
    export_solution_2(orders['strahler'], 'solution', 'test_order','orders')
    export_solution_2(orders['generation'], 'generations', 'test_generations', 'gen')