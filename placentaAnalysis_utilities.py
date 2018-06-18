import numpy as np
import matplotlib.pyplot as plt
from matplotlib import cm
from mpl_toolkits.mplot3d import Axes3D

######
# This function removes the rows in all input arrays the correspond to where the first array has values less than zero
######
def removeRows(mainArray, Arrays):
    i = 0
    while (i < len(mainArray)):
        if (mainArray[i,0] < 0):

            # get rid of element, from elements and from other variables
            for j in range(0, len(Arrays)): #sort out list here
                array=Arrays[j]
                array=np.delete(array, (i), axis=0)
                Arrays[j]=array

            mainArray = np.delete(mainArray, (i), axis=0)
        else:
            i = i + 1
    return (mainArray, Arrays)

######
# This function swaps two rows of an array
######
def rowSwap2d(array, row1, row2):
    placeholder = np.copy(array[row1, :])
    array[row1, :] = array[row2, :]
    array[row2, :] = placeholder
    return array

######
# This function swaps two rows of an array
######
def rowSwap1d(array, row1, row2):
    placeholder = np.copy(array[row1])
    array[row1] = array[row2]
    array[row2] = placeholder
    return array

######
# This simple function finds the first occurence of row v in a given matrix
# Inputs: row v, and matrix
# Outputs: returns index of row or else -1 if row is not present in the matrix
######
def ismember(v, matrix):

    index = -1

    L =(np.shape(matrix))
    L=L[0]

    for i in range(0, L):
        if np.array_equal(v, matrix[i,:]):
            index = i
            return index
    return -1

######
# This function plots the branching tree in 3D
# Inputs: list of nodes, elements, colour of each element, radius of each element
# Outputs: creates a 3D plot
######
def plotVasculature3D(nodes, elems, colour, radii):

    Ne=len(elems)
    elems=elems[:,1:3]
    x=np.zeros([Ne,2])
    y = np.zeros([Ne,2])
    z = np.zeros([Ne,2])

    fig = plt.figure()
    ax = fig.gca(projection='3d')

    colour = (colour - min(colour)) / max(colour)*255
    radii = radii / max(radii) * 3
    for i in range(0,Ne):
        nN1=int(elems[i,0])
        nN2 = int(elems[i, 1])
        x[i][0]=nodes[nN1,0]
        y[i][0]=nodes[nN1,1]
        z[i][0]=nodes[nN1, 2]
        x[i][1]=nodes[nN2,0]
        y[i][1]=nodes[nN2,1]
        z[i][1]=nodes[nN2, 2]

    for i in range (0,Ne):
        x1=np.squeeze(x[i,0:2])
        y1=np.squeeze(y[i,0:2])
        z1=np.squeeze(z[i,0:2])

        colourValue=np.asarray(cm.jet(int(colour[i])))
        ax.plot(x1,y1,z1, c=colourValue[0:3],linewidth=radii[i])

    ax.set_aspect('equal')
    plt.show()

    return 0

############
# This function is taken from placentagen, but has some modifications, because the tree from image analysis does not follow all the rules of a generated tree
############
def evaluate_orders(node_loc, elems):
    # calculates generations, Horsfield orders, Strahler orders for a given tree
    # Works for diverging trees only
    # Inputs are:
    # node_loc = array with location of nodes
    # elems = array with location of elements

    num_elems = len(elems)

    # Calculate connectivity of elements
    elem_connect = element_connectivity_1D(node_loc, elems)
    elem_upstream = elem_connect['elem_up']
    elem_downstream = elem_connect['elem_down']

    # Initialise order definition arrays
    strahler = np.zeros(len(elems), dtype=int)
    horsfield = np.zeros(len(elems), dtype=int)
    generation = np.zeros(len(elems), dtype=int)

    # Calculate generation of each element
    maxgen = 1  # Maximum possible generation
    for ne in range(0, num_elems):
        ne0 = elem_upstream[ne][1]
        if ne0 != 0:
            # Calculate parent generation
            n_generation = generation[ne0]
            if elem_downstream[ne0][0] == 1:
                # Continuation of previous element
                generation[ne] = n_generation
            elif elem_downstream[ne0][0] >= 2:
                # Bifurcation (or morefurcation)
                generation[ne] = n_generation + 1
        else:
            generation[ne] = 1  # Inlet
        maxgen = np.maximum(maxgen, generation[ne])

    # Now need to loop backwards to do ordering systems
    for ne in range(num_elems - 1, -1, -1):
        n_horsfield = np.maximum(horsfield[ne], 1)
        n_children = elem_downstream[ne][0]
        if n_children == 1:
            if generation[elem_downstream[ne][1]] == 0:
                n_children = 0
        temp_strahler = 0
        strahler_add = 1
        if n_children >= 2:  # Bifurcation downstream
            temp_strahler = strahler[elem_downstream[ne][1]]  # first daughter
            for noelem in range(1, n_children + 1):
                ne2 = elem_downstream[ne][noelem]
                temp_horsfield = horsfield[ne2]
                if temp_horsfield > n_horsfield:
                    n_horsfield = temp_horsfield
                if strahler[ne2] < temp_strahler:
                    strahler_add = 0
                elif strahler[ne2] > temp_strahler:
                    strahler_add = 0
                    temp_strahler = strahler[ne2]  # strahler of highest daughter
            n_horsfield = n_horsfield + 1
        elif n_children == 1:
            ne2 = elem_downstream[ne][1]  # element no of daughter
            n_horsfield = horsfield[ne2]
            strahler_add = strahler[ne2]
        horsfield[ne] = n_horsfield
        strahler[ne] = temp_strahler + strahler_add

    return {'strahler': strahler, 'horsfield': horsfield, 'generation': generation}


############
# This function is taken from placentagen, but has some modifications, because the tree from image analysis does not follow all the rules of a generated tree
############
def element_connectivity_1D(node_loc, elems):

    # Initialise connectivity arrays, these need to be bigger to accomodate more branches
    num_elems = len(elems)
    elem_upstream = np.zeros((num_elems, 4), dtype=int)
    elem_downstream = np.zeros((num_elems, 4), dtype=int)

    num_nodes = len(node_loc)
    elems_at_node = np.zeros((num_nodes, 8), dtype=int)

    # determine elements that are associated with each node
    for ne in range(0, num_elems):
        for nn in range(1, 3):
            nnod = int(elems[ne][nn])
            elems_at_node[nnod][0] = elems_at_node[nnod][0] + 1
            elems_at_node[nnod][elems_at_node[nnod][0]] = ne

    # assign connectivity
    for ne in range(0, num_elems):
        nnod2 = int(elems[ne][2])

        for noelem in range(1, elems_at_node[nnod2][0] + 1):
            ne2 = elems_at_node[nnod2][noelem]

            if ne2 != ne:
                elem_upstream[ne2][0] = elem_upstream[ne2][0] + 1
                elem_upstream[ne2][elem_upstream[ne2][0]] = ne
                elem_downstream[ne][0] = elem_downstream[ne][0] + 1
                elem_downstream[ne][elem_downstream[ne][0]] = ne2

    return {'elem_up': elem_upstream, 'elem_down': elem_downstream}