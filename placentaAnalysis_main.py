import placentagen as pg
import numpy as np
import pandas as pd
from placentaAnalysisFunctions import *

######
# Give information about the branching tree
# Inputs: A csv file generated from the MySkeletonizationProcess ImageJ macro
# Outputs: Creates exelem and exnode files of tree, prints a table of results
######

#Define data to use (this is the bit that needs to be edited each time)
csvDataFile="branch info.csv"
inlet_loc=np.array([312,51,628]) #you need to find this manually
conversionFactor=9.1955 #ImageJ log prints this out when you run "MySkeletonizationProcess"
voxelSize=0.01933 #mm

#Read in file
print("Reading in Data")
data_file = pd.read_csv(csvDataFile, usecols=['Skeleton ID', 'Branch length', 'V1 x', 'V1 y', 'V1 z', 'V2 x', 'V2 y', 'V2 z', 'Euclidean distance', 'average intensity (inner 3rd)'])
data_file.columns = ['SkeletonID', 'Branchlength', 'V1x', 'V1y', 'V1z', 'V2x', 'V2y', 'V2z', 'Euclideandistance', 'averageintensityinner3rd']

#Get Skeleton Info
print('Organizing Data')
geom=sort_data(data_file)

#Find statistics for skeleton
print('Analysing Skeleton')
geom = arrange_by_strahler_order(geom, inlet_loc)

orders=evaluate_orders(geom['nodes'],geom['elems'])
strahler=orders['strahler']
strahler[0]=strahler[1] #as pg doesn't find this order
orders['strahler']=strahler

#Prune elements by order and re-evaluate generation
threshold_order=4
geom=prune_by_order(geom, orders, threshold_order)
order2=evaluate_orders(geom['nodes'],geom['elems'])
geom['branch_angles'] = find_branch_angles(geom['nodes'], geom['elems'], order2['generation'])# need to  ################################

#scale results into mm and degrees
geom['radii']=geom['radii']*voxelSize/conversionFactor
geom['length']=geom['length']*voxelSize
geom['euclidean length']=geom['euclidean length']*voxelSize
geom['branch_angles']=geom['branch_angles']*180/np.pi

#Output Skeleton Info
print('Output Data')
#table
table=summary_statistics(orders['strahler'],order2['generation'], geom['length'],geom['euclidean length'], geom['radii'],geom['branch_angles'],geom['branch_angles'],geom['branch_angles']) #fix the last 2 inputs once they are done

#3d plots
print('Plotting')
plotVasculature3D(geom['nodes'], geom['elems'], order2['generation'],geom['radii'])

#csv files
output=0
if output:
    print('Writing files')
    elems=geom['elems']
    outPutData=np.column_stack([elems[:,1:3], geom['radii'], geom['branch_angles'], geom['strahler_order']])

    np.savetxt('ElementInfo.csv', outPutData, fmt='%.2f', delimiter=',', header=" ,elems,  radii(mm),  angles(degrees),  order")
    np.savetxt('NodeInfo.csv', geom['nodes'], fmt='%.4f', delimiter=',', header=" ,nodes(voxels)")

    #cmgui files
    pg.export_ex_coords(geom['nodes'],'placenta','full_tree','exnode')
    pg.export_exelem_1d(geom['elems'],'placenta','full_tree')
