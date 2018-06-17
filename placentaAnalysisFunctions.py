import numpy as np
from tabulate import tabulate
from placentaAnalysis_utilities import *
import placentagen as pg

######
# This function takes data from the csv and converts it to arrays
# Inputs: data_file, generated from the panadas read_csv function
# Outputs: arrays containing nodes, elems and radii, length & euclidean length of each elem
# Nodes: A list of x/y/z coordinates of each node
# Elements: the first column is the element number, the following two columns are the index of the start and end node
######
def sort_data(data_file):

    # get rid of any skeletons other than the main one
    data_file=data_file[data_file.SkeletonID == 1]

    #get skeleton properties as arrays
    euclid_length=data_file.Euclideandistance.values
    length = data_file.Branchlength.values
    radii = data_file.averageintensityinner3rd.values

    #get elem and node data
    data_file=data_file.drop(['SkeletonID','Branchlength','averageintensityinner3rd','Euclideandistance'], axis=1)
    data_file=data_file.values
    (elems, nodes) = sort_elements(data_file[:, 0:3],data_file[:,3:6])

    #get rid of dud elements (which have negative values
    (elems, [length, euclid_length, radii])=removeRows(elems, [length, euclid_length, radii])

    return {'nodes': nodes, 'elems':elems, 'radii': radii, 'length': length, 'euclidean length': euclid_length}

######
# This function takes a list of node pairs (v1, v2) and creates a list of nodes and elements
# Inputs: start nodes: v1, end nodes: v2
# Outputs: nodes, containing a list of all unique nodes, and elems, containing the start and end node index
#          where an element starts and ends at the same node, it given a value of -1
######
def sort_elements(v1, v2):


    Nelem=len(v1)
    elems = np.zeros([Nelem, 3])
    nodes = np.zeros([Nelem*2, 3]) #max number of nodes possible

    iN=0 #node index

    for iE in range(0, Nelem): #go through first node list
        v=v1[iE,:]
        index = ismember(v, nodes[0:iN][:]) #see if the node is in the nodes list

        if (index == -1): #if not, create a new node
            nodes[iN, :]= v
            index=iN
            iN=iN+1

        # else just use index of existing node
        elems[iE,1] = int(index)
        elems[iE, 0] = int(iE) #first column of elements is just the element number

    for iE in range(0, Nelem ): #go through second node list and do the same thing
        v = v2[iE, :]
        index = ismember(v, nodes[0:iN, :])

        if (index == -1):
            nodes[iN, :] = v
            index = iN
            iN = iN + 1
        #else index is alread found
        elems[iE, 2] = int(index)

        if (elems[iE][1]==elems[iE][2]):
            elems[iE,0:2]=int(-2)

    nodes = nodes[0:iN:1][:] #truncate

    return (elems, nodes)


######
# This function rearranges elems (and other properties) according to their Strahler order, to be compatible with placentagen functions
# Inputs: geom (all info about the skeleton) and coordinates of root node
# Outputs: list of orders of each element, and all elements etc. are rearranged in order
######
def arrange_by_strahler_order(geom, inlet_loc):

    #set up arrays
    nodes=geom['nodes']
    radii=geom['radii']
    length=geom['length']
    euclid_length = geom['euclidean length']
    elems = np.copy(geom['elems'])  # as elems is altered in this function
    elems=elems[:,1:3] #get rid of first column which means nothing

    Ne = len(elems)
    Nn = len(nodes)

    elems_new = np.zeros([Ne, 2])
    radii_new = np.zeros([Ne, 1])
    len_new = np.zeros([Ne, 1])
    euclid_len_new = np.zeros([Ne, 1])
    strahler_order = np.zeros([Ne, 1])

    # find root node and element from its coordinates
    Nn_root=ismember(inlet_loc, nodes)
    if (Nn_root==-1):
        print("Warning, root node not located")

    #find root element
    Ne_place=np.where(elems==Nn_root)
    Ne_root = Ne_place[0]  # only need first index
    if len(Ne_root) > 1:
        print("Warning, root node is associated with multiple elements")
    if len(Ne_root) == 0:
        print("Warning, no root element located")
    Ne_root = Ne_root[0]
    # make root element the first element
    elems=rowSwap2d(elems, 0, Ne_root)
    radii = rowSwap1d(radii, 0, Ne_root)
    length = rowSwap1d(length, 0, Ne_root)
    euclid_length = rowSwap1d(euclid_length, 0, Ne_root)

    #get element pointing right way
    if (np.squeeze(Ne_place[1])!= 0):
        elems[0,:]=rowSwap1d(np.squeeze(elems[0,:]),1,0)

    #find orders
    counter=1
    counter_new=0
    while (counter<Ne):
        # find elements which are terminal
        terminal_elems = np.zeros([Ne, 1])

        #go through each node
        for i in range(0, Nn+1):

            # find number of occurences of the node
            places = np.where(elems == i)
            ind1=places[0]
            ind2 = places[1]

            if (len(ind1) == 1) and ((ind1[0]) != 0): #if occurs once, then element is terminal (avoids root element)

                ind1 = ind1[0]
                ind2 = ind2[0]

                # swap to ensure element points right way
                if ind2==0:
                    elems[ind1,:]=rowSwap1d(np.squeeze(elems[ind1,:]),1,0)

                #assign element under the new element ordering scheme
                elems_new[counter_new, :] = elems[ind1, :]
                radii_new[counter_new, :] = radii[ind1]
                len_new[counter_new, :] = length[ind1]
                euclid_len_new[counter_new, :] = euclid_length[ind1]
                counter_new=counter_new+1

                terminal_elems[ind1] = 1

                # join up element with upstream elements
                nodeNumNew = elems[ind1, 0] #this is node number at other end of element
                nodeNum=i
                places = np.where(elems == nodeNumNew)  # find where the new node occurs
                ind1 = places[0]
                ind2 = places[1]

                counter2 = 1

                while ((len(ind1) == 2) & (counter2 < Ne)):  # as can only be present twice if a joining node

                    # see if branch joins to yet another branch, that we haven't yet encountered (i.e. not nodeNum)
                    if (elems[ind1[0], ~ind2[0]] == nodeNum):
                        k = 1
                    else:
                        k = 0
                    terminal_elems[ind1[k]] = 1 # label terminal_elems as joining elements

                    # switch the way element points
                    if (ind2[k] == 0):
                        elems[ind1[k], :] = rowSwap1d(np.squeeze(elems[ind1[k], :]), 1, 0)

                    nodeNum = nodeNumNew
                    nodeNumNew = elems[ind1[k], 0]

                    #assign new order
                    elems_new[counter_new, :] = elems[ind1[k], :]
                    radii_new[counter_new, :] = radii[ind1[k]]
                    len_new[counter_new, :] = length[ind1[k]]
                    euclid_len_new[counter_new, :] = euclid_length[ind1[k]]
                    counter_new = counter_new + 1

                    # update loop criteria
                    places = np.where(elems == nodeNumNew)
                    ind1 = places[0]
                    ind2 = places[1]
                    counter2 = counter2 + 1

        #update elems to 'get rid of' terminal elements from the list
        terminal_elems[0]= 0 #the root node can never be terminal
        terminal_elems_pair=np.column_stack([terminal_elems, terminal_elems])
        elems[terminal_elems_pair == 1] = -1

        #loop exit criteria
        places=np.where(terminal_elems == 1)
        places=places[1]
        if len(places)==0:
            counter = Ne+1
        counter = counter + 1

    #assign root element in new order systems
    elems_new[Ne-1, :] = elems[0, :]
    radii_new[Ne-1, :] = radii[0]
    len_new[Ne-1, :] = length[0]
    euclid_len_new[Ne-1, :] = euclid_length[0]

    #reverse order
    elems_new=np.flip(elems_new,0)
    radii_new = np.flip(radii_new, 0)
    len_new = np.flip(len_new, 0)
    euclid_len_new = np.flip(euclid_len_new, 0)

    elems=geom['elems']
    elems[:,1:3]=elems_new

    return {'elems': elems, 'radii': radii_new,'length': len_new, 'euclidean length': euclid_len_new, 'nodes': nodes}



######
# This function removes terminal elements that connect to high order branches
# Inputs: geom, all the info about the skeleton and threshold order shich is the parent branch is equal to order higher than, the terminal branch is removed
# Outputs: removes elements that are unwanted and also removed corresponding rows from other data
######
def prune_by_order(geom, ordersAll, threshold_order):

     orders=ordersAll['strahler']
     elems=geom['elems']
     elems=elems[:,1:3]

     terminalList = np.where(orders == 1)
     terminalList=terminalList[0]

     #go through list of terminal elements
     for i in range(0, len(terminalList)):
         row = terminalList[i]

         #find parents at the non terminal end of the element, and their order
         ind=np.where(elems == elems[row,0])
         ind=ind[0]
         orderMax = np.max(orders[ind])

         #remove element if order exceeds threshold
         if (orderMax>threshold_order):
             elems[row,:]=-1

     # get rid of dud elements
     (elems, [geom['length'],geom['euclidean length'],geom['radii'],ordersAll['strahler'],ordersAll['generation'],geom['elems']])=removeRows(elems, [geom['length'],geom['euclidean length'],geom['radii'],ordersAll['strahler'],ordersAll['generation'],geom['elems']])
     return geom

######
# This function finds branch angles of nodes
# Inputs: list of nodes, elements and their generation
# Outputs: list of branch angles in radians, where the angle is the angle of a given element from its parents, and is zero if the element is a continuation of the parent
######
def find_branch_angles(nodes, elems, generations): #######################needs fixing + add in finding diameter and length ratios
    connectivity=pg.pg_utilities.element_connectivity_1D(nodes, elems)
    elem_up=connectivity['elem_up']

    num_elems = len(elems)
    elems = elems[:, 1:3] #get rid of useless first column
    branch_angles = -1. * np.ones(num_elems)

    #find angle for each element
    error=0
    for ne in range(0, num_elems):

        neUp=elem_up[ne,1]

        if elem_up[ne,0]!=1:
            error=error+1
        elif generations[neUp]<generations[ne]: #then there is a branch at this node

            #parent node
            endNode=int(elems[neUp, 0])
            startNode=int(elems[neUp, 1])
            v_parent = nodes[endNode, :] - nodes[startNode,:]
            v_parent = v_parent / np.linalg.norm(v_parent)

            #daughter
            endNode = int(elems[ne, 1])
            startNode = int(elems[ne, 0])
            v_daughter = nodes[startNode, :] - nodes[endNode, :]
            v_daughter=v_daughter/np.linalg.norm(v_daughter)

            branch_angles[ne] = np.arccos(np.dot(v_parent, v_daughter))

    print('Number of elements for which no angle could be found (no unqiue parent) = ' +str(error))
    return (branch_angles)

######
# This function finds statistics on branching tree
# Inputs: list of various tree attributes
# Outputs: table of information by order
######
def summary_statistics(orders, generation, length, euclid_length, radii, branch_angles, diam_ratio, length_ratio):

    #statisitcs by order
    num_orders = max(orders)
    num_orders=int(num_orders)

    #initialize arrays
    order_ord = np.zeros(num_orders)
    radii_ord = np.zeros(num_orders)
    tortuosity_ord = np.zeros(num_orders)
    number_ord = np.zeros(num_orders)
    number_ord_unjoined = np.zeros(num_orders)
    angle_ord = np.zeros(num_orders)
    euclid_length_ord = np.zeros(num_orders)
    length_ord = np.zeros(num_orders)
    length_diam_ord= np.zeros(num_orders)

    for n_ord in range(0, num_orders):

        elem_list = (orders==n_ord+1)

        # get stats for each order (_ord means "by order")
        number_ord_unjoined[n_ord]=len(np.extract(elem_list, elem_list))
        order_ord[n_ord]=n_ord+1
        radii_2=np.extract(elem_list, radii)
        radii_ord[n_ord]=np.mean(radii_2)
        tortuosity_ord[n_ord]=np.mean(np.extract(elem_list, length)/np.extract(elem_list, euclid_length))

        branch_list=np.extract(elem_list, branch_angles)
        branch_list = branch_list[(branch_list > -1)] #for actual distinct branches

        number_ord[n_ord]=len(branch_list)

        if (number_ord[n_ord]==0):
            number_ord[n_ord]=1
            angle_ord[n_ord]=np.nan
        else:
            angle_ord[n_ord]=np.mean(branch_list)

        euclid_length_ord[n_ord]=np.sum(np.extract(elem_list, euclid_length))/number_ord[n_ord]

        length_2=np.extract(elem_list, length)
        length_ord[n_ord] = np.sum(length_2) / number_ord[n_ord]

        #Some branches have zero radius and so we will ignore these, and assume the number of theze is small
        length_2 = length_2[(radii_2 > 0)]
        radii_2 = radii_2[(radii_2 > 0)]
        length_diam_ord[n_ord]=np.mean(length_2/(2*radii_2))

    #print table
    header = ['Order','Unjoined Number','Number','Length(mm)','Euclidean Length(mm)', 'Radius(mm)','Len/Diam', 'Tortuosity','Angle(degrees)']
    table=np.column_stack([order_ord,number_ord_unjoined,number_ord, length_ord,euclid_length_ord,radii_ord,length_diam_ord,tortuosity_ord, angle_ord])
    print('\n')
    print('Statistics By Order: ')
    print('..................')
    print(tabulate(table, headers=header))

    #statistics independent of order
    length_2 = length[(radii > 0)]
    radii_2 = radii[(radii > 0)]
    length_diam_overall = np.mean(length_2 / (2 * radii_2))

    elem_list = (orders > 0)
    branch_list = np.extract(elem_list, branch_angles)
    branch_list = branch_list[(branch_list > -1)]  # for actual distinct branches
    angle_overall = np.mean(branch_list)
    table = np.column_stack([-1, (len(np.extract(elem_list, elem_list))), len(branch_list), np.sum(length)/len(branch_list), np.sum(euclid_length)/len(branch_list), np.mean(radii), length_diam_overall,np.mean(length/euclid_length), angle_overall])
    header = ['     ', '               ', '      ', '          ', '                    ', '          ', '        ',
              '          ', '              ']
    print(tabulate(table, headers=header))
    print('\n')

    #Other statistics
    print('Other statistics: ')
    print('..................')
    print('Num generations = ' + str(max(generation))) #note that this value is calculated after pruning
    terminalGen = generation[(orders == 1)]
    print('Terminal generation = ' + str(np.mean(terminalGen)))
    diam_ratio = diam_ratio[(diam_ratio > -1)]
    length_ratio = length_ratio[(length_ratio > -1)]
    print('D/Dparent = ' + str(np.mean(diam_ratio)))
    print('L/Lparent = ' + str(np.mean(length_ratio)))
    print('\n')

    return table



