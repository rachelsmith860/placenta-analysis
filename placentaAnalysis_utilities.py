import numpy as np
import matplotlib.pyplot as plt
from matplotlib import cm
from mpl_toolkits.mplot3d import Axes3D


######
# Function: Remove rows from both mainArray and Arrays at which main array has values less than zero
# Inputs: mainArray - an N x M array of values
#         Arrays - a list of arrays each with length N for their first axis
# Outputs: for each row of mainArray for which the first element is below zero; this row is removed from mainArray and from each array
######

def remove_rows(main_array, arrays):

    i = 0

    while i < len(main_array):
        if main_array[i, 0] < 0:    # then get rid of row from all arrays

            for j in range(0, len(arrays)):
                array = arrays[j]
                array = np.delete(array, (i), axis=0)
                arrays[j] = array
            main_array = np.delete(main_array, (i), axis=0)

        else:
            i = i + 1

    return main_array, arrays


######
# Function: Swaps 2 rows in an array
# Inputs: array - a N x M array
#         row1 & row2 - the indices of the two rows to be swapped
# Outputs: array, with row1 and row2 swapped
######

def row_swap_2d(array, row1, row2):
    placeholder = np.copy(array[row1, :])
    array[row1, :] = array[row2, :]
    array[row2, :] = placeholder
    return array


######
# Function: Swaps 2 rows in an array
# Inputs: array - a N x 1 array
#         row1 & row2 - the indices of the two rows to be swapped
# Outputs: array, with row1 and row2 swapped
######

def row_swap_1d(array, row1, row2):
    placeholder = np.copy(array[row1])
    array[row1] = array[row2]
    array[row2] = placeholder
    return array


######
# Function: Finds first occurrence of a specified row of values in an array or returns -1 if the given row is not present
#           Similar to Matlab isMember function
# Inputs: matrix - an N x M array
#         v - a 1 x M array
# Outputs: index at which v first occurs in matrix, or else -1
######

def is_member(v, matrix):

    L = (np.shape(matrix))
    L = L[0]

    for i in range(0, L):
        if np.array_equal(v, matrix[i, :]):
            index = i
            return index
    return -1


######
# Function: Creates a 3D plot of branching tree
# Inputs: nodes - an M x 3 array giving cartesian coordinates (x,y,z) for the node locations in the tree
#         elems - an N x 3 array, the first colum in the element number, the second two columns are the index of the start and end node
#         colour - an N x 1 array where value determines colour of corresponding element
#         Nc - the maximum number of elements connected at a single node
# Outputs: 3D plot of tree, with radius proportional to radii and colour depending on the input array
######

def plot_vasculature_3d(nodes, elems, colour, radii):

    # initialize arrays
    Ne = len(elems)
    elems = elems[:, 1:3]
    x = np.zeros([Ne,2])
    y = np.zeros([Ne,2])
    z = np.zeros([Ne,2])

    fig = plt.figure()
    ax = fig.gca(projection='3d')

    # scale colour and radii
    colour = (colour - min(colour)) / max(colour) * 255
    radii = radii / max(radii) * 3

    for i in range(0, Ne):

        #get start and end node
        nN1 = int(elems[i, 0])
        nN2 = int(elems[i, 1])

        #get coordinates of nodes
        x[i,0] = nodes[nN1, 0]
        y[i,0] = nodes[nN1, 1]
        z[i,0] = nodes[nN1, 2]
        x[i,1] = nodes[nN2, 0]
        y[i,1] = nodes[nN2, 1]
        z[i,1] = nodes[nN2, 2]

        colour_value = np.asarray(cm.jet(int(colour[i])))
        ax.plot(np.squeeze(x[i,:]), np.squeeze(y[i,:]), np.squeeze(z[i,:]), c=colour_value[0:3], linewidth=radii[i])

    plt.show()

    return 0


######
# Function: Finds the maximum number of elements that join at one node
# Inputs: elems - an N x 3 array containing element number in the first column and node indices in the second two columns
# Outputs: Nc - the maximum number of elements that join at one node
######

def find_maximum_joins(elems):

    elems = np.concatenate([np.squeeze(elems[:, 1]), np.squeeze(elems[:, 2])])
    elems = elems.astype(int)
    result = np.bincount(elems)
    Nc = (max(result)) + 1

    # Warning if detect an unusual value
    if Nc > 10:
        print('Warning, large number of elements at one node: ' + str(Nc))
        Nc = 10

    return Nc

######
# Modified from placentagen (https://github.com/VirtualPregnancy/placentagen)
# Note: only works for diverging trees
# Modifications ensure that function works for more than three elements joining at one node
# Inputs: elems - an N x 3 array, the first colum in the element number, the second two columns are the index of the start and end node
#         elem_connect - connectivity of elements, as created by element_connectivity_1D
# Outputs: orders, containing 3 N x 1 arrays which give the Strahler / Horsefield / Generation of each element
######

def evaluate_orders(elems, elem_connect):

    num_elems = len(elems)

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


######
# Modified from placentagen (https://github.com/VirtualPregnancy/placentagen)
# Modifications ensure that function works for more than three elements joining at one node
# Inputs: node_loc - an M x 3 array giving cartesian coordinates (x,y,z) for the node locations in the tree
#         elems - an N x 3 array, the first colum in the element number, the second two columns are the index of the start and end node
#         Nc - the maximum number of elements connected at a single node
# Outputs: elem_up: an N x Nc array containing indices of upstream elements (the first value is number of upstream elements)
#          elem_down: an N x Nc array containing indices of downstream elements (the first value is number of downstream elements)
######

def element_connectivity_1D(node_loc, elems, Nc):

    # Initialise connectivity arrays
    num_elems = len(elems)
    elem_upstream = np.zeros((num_elems, Nc), dtype=int)
    elem_downstream = np.zeros((num_elems, Nc), dtype=int)

    num_nodes = len(node_loc)
    elems_at_node = np.zeros((num_nodes, Nc), dtype=int)

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


######
# Modified from placentagen (https://github.com/VirtualPregnancy/placentagen)
# Writes values to a cmgui exelem file
# Inputs: data - an N x 1 array with a value for each element in the tree
#         groupname - group name that will appear in cmgui
#         filename - name that the file is saved as
#         name - name that values will be called in cmgui
# Outputs: an "exelem" file containing the data value for each element, named according to names specified
######

def export_solution_2(data, groupname, filename, name):

    # Write header
    type = "exelem"
    data_num = len(data)
    filename = filename + '.' + type
    f = open(filename, 'w')
    f.write(" Group name: %s\n" % groupname)
    f.write("Shape. Dimension=1\n")
    f.write("#Scale factor sets=0\n")
    f.write("#Nodes=0\n")
    f.write(" #Fields=1\n")
    f.write("1) " + name + ", field, rectangular cartesian, #Components=1\n")
    f.write(name + ".  l.Lagrange, no modify, grid based.\n")
    f.write(" #xi1=1\n")

    # Write element values
    for x in range(0, data_num):
        f.write(" Element:            %s 0 0\n" % int(x + 1))
        f.write("   Values:\n")
        f.write("          %s" % np.squeeze(data[x]))
        f.write("   %s \n" % np.squeeze(data[x]))
    f.close()

    return 0

