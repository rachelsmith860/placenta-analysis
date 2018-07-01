import placentagen as pg
import numpy as np
import pandas as pd
from placentaAnalysisFunctions import *

######
# Give information about the branching tree
# Inputs: A csv file generated from the MySkeletonizationProcess ImageJ macro
# Outputs: Creates exelem and exnode files of tree, prints a table of results
######

#Define data to use
csvDataFile="branch info human 2.csv"
inlet_loc=np.array([281,4,215])
conversionFactor=22.11
voxelSize=1 #mm #####################################to fix

#Read in file
print("Reading in Data")
data_file = pd.read_csv(csvDataFile, usecols=['Skeleton ID', 'Branch length', 'V1 x', 'V1 y', 'V1 z', 'V2 x', 'V2 y', 'V2 z', 'Euclidean distance', 'average intensity (inner 3rd)'])
data_file.columns = ['SkeletonID', 'Branchlength', 'V1x', 'V1y', 'V1z', 'V2x', 'V2y', 'V2z', 'Euclideandistance', 'averageintensityinner3rd']

#Get Skeleton Info
print('Organizing Data')
geom=sort_data(data_file)

#Find orders for skeleton
print('Analysing Skeleton')
geom = arrange_by_strahler_order(geom, inlet_loc)

Nc=find_maximum_joins(geom['elems'])
print(Nc)
elem_connect=element_connectivity_1D(geom['nodes'], geom['elems'], Nc)
orders=evaluate_orders(geom['elems'], elem_connect)

strahler=orders['strahler']
strahler[0]=strahler[1] #assumed
orders['strahler']=strahler

#Prune elements by order and re-evaluate ordering
threshold_order=4
geom=prune_by_order(geom, orders['strahler'], threshold_order)

Nc=find_maximum_joins(geom['elems'])
elem_connect=element_connectivity_1D(geom['nodes'], geom['elems'], Nc)
orders=evaluate_orders(geom['elems'], elem_connect)

strahler=orders['strahler']
strahler[0]=strahler[1] #assumed
orders['strahler']=strahler

geom['radii']=geom['radii']/conversionFactor
radii_unscaled=np.copy(geom['radii'])
(geom['branch_angles'],geom['diam_ratio'],geom['length_ratio']) = find_branch_angles(geom, orders, elem_connect)

#scale results into mm and degrees
geom['radii']=geom['radii']*voxelSize
geom['length']=geom['length']*voxelSize
geom['euclidean length']=geom['euclidean length']*voxelSize
geom['branch_angles']=geom['branch_angles']*180/np.pi

#Output Skeleton Info
print('Output Data')

#table
table=summary_statistics(orders, geom)
#3d plots
print('Plotting')
plot_vasculature_3d(geom['nodes'], geom['elems'], orders['strahler'],geom['radii'])

output=1
if output:

    # csv files
    print('Writing files')
    elems=geom['elems']
    outPutData=np.column_stack([elems[:,1:3], geom['radii'], geom['branch_angles'], strahler])

    np.savetxt('ElementInfo.csv', outPutData, fmt='%.2f', delimiter=',', header=" ,elems,  radii(mm),  angles(degrees),  order")
    np.savetxt('NodeInfo.csv', geom['nodes'], fmt='%.4f', delimiter=',', header=" ,nodes(voxels)")

    #cmgui files
    pg.export_ex_coords(geom['nodes'],'vessels','human_tree','exnode')
    pg.export_exelem_1d(geom['elems'],'vessels','human_tree')
    export_solution_2(orders['strahler'], 'solution', 'human_solution','orders')
    export_solution_2(radii_unscaled, 'radii', 'human_radii','radius')
    export_solution_2(orders['generation'], 'generations', 'human_generations', 'gen')
