import numpy as np
from tabulate import tabulate
import matplotlib.pyplot as plt
from matplotlib import cm
from mpl_toolkits.mplot3d import Axes3D

######
# This function takes data from the csv and converts it to arrays
# Inputs: data_file, generated from the panadas read_csv function
# Outputs: arrays containing nodes, elems and radii, length & euclidean length of each elem
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

    #get rid of dud elements (which have negative values)
    i=0
    while (i<len(radii)):
        if (elems[i][0]<0):
            # get rid of element, from elements and from other variables
            length = np.delete(length, (i), axis=0)
            euclid_length = np.delete(euclid_length, (i), axis=0)
            radii = np.delete(radii, (i), axis=0)
            elems = np.delete(elems, (i), axis=0)
        i=i+1

    return {'nodes': nodes, 'elems':elems, 'radii': radii, 'length': length, 'euclidean length': euclid_length}

######
# This function takes a list of node pairs (v1, v2) and creates a list of nodes and elements
# Inputs: start nodes: v1, end nodes: v2
# Outputs: nodes, containing a list of all unique nodes, and elems, containing the start and end node number
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
# This simple function finds the first occurence of row v in a given matrix
# Inputs: row v, and maxtrix
# Outputs: returns index of row or else -1 if row is not present in the matrix
######
def ismember(v, matrix):

    index = -1

    L =(np.shape(matrix))
    for i in range(0, L[0]):
        if np.array_equal(v, matrix[i,:]):
            index = i
    return index

######
# This function find strahler ordering of skeleton
# Inputs: nodes, elements and coordinates of root node
# Outputs: list of orders of each element. The order of the parent node can't be found with this algorithm
#          and is assumed to be one higher than the last order assigned
######
def get_strahler_order(geom, inlet_loc):

    #set up arrays
    nodes=geom['nodes']
    radii = geom['radii']
    length = geom['length']
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

    # find root node and element from its physical location
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
    # make root element the first element
    Ne_root = Ne_root[0]
    placeholder = elems[0, :]
    elems[0, :] = elems[Ne_root, :]
    elems[Ne_root, :] = placeholder
    placeholder = radii[0]
    radii[0] = radii[Ne_root]
    radii[Ne_root] = placeholder
    placeholder = length[0]
    length[0] = length[Ne_root]
    length[Ne_root] = placeholder
    placeholder = euclid_length[0]
    euclid_length[0] = euclid_length[Ne_root]
    euclid_length[Ne_root] = placeholder
    #get element pointing right way
    if (np.squeeze(Ne_place[1])!= 0):
        elems[Ne_root,1]=elems[Ne_root,0]
        elems[Ne_root,0]=Nn_root

    #find orders
    counter=1
    counter_new_order=0
    while (counter<Ne):
        # find elements which are terminal
        terminal_elems = np.zeros([Ne, 1])

        for i in range(0, Nn+1):
            places = np.where(elems == i) #find number of occurences of the node
            ind1=places[0]
            ind2 = places[1]

            if (len(ind1) == 1) and ((ind1[0]) != 0): #if occurs once, then element is terminal

                ind1 = ind1[0]
                ind2 = ind2[0]

                if ind2==0: #swap to ensure element points right way
                    placeholder = np.copy(elems[ind1, 1])
                    elems[ind1, 1] = int(np.copy(elems[ind1, 0]))
                    elems[ind1, 0] = int(placeholder)

                elems_new[counter_new_order, :] = elems[ind1, :]
                radii_new[counter_new_order, :] = radii[ind1]
                len_new[counter_new_order, :] = length[ind1]
                euclid_len_new[counter_new_order, :] = euclid_length[ind1]
                counter_new_order=counter_new_order+1

                terminal_elems[ind1] = 1
                iNew = elems[ind1, 0] #this is node number at other end of element
                [terminal_elems, elems_new,len_new, euclid_len_new, radii_new, counter_new_order] = join_elements(terminal_elems, elems, elems_new,len_new, euclid_len_new, radii_new,length, euclid_length, radii, counter_new_order,i, iNew) #find elements that join this element without branching

        terminal_elems[0]= 0 #the root node can never be terminal

        strahler_order[terminal_elems == 1] = counter #assign order to terminal elems

        terminal_elems_pair=np.column_stack([terminal_elems, terminal_elems])
        elems[terminal_elems_pair == 1] = -1 #'get rid of' terminal elements from the list

        #loop exit criteria
        places=np.where(terminal_elems == 1)
        inds=places[1]
        if len(inds)==0:
            counter = Ne+1
        else:
            strahler_order[0] = counter +1  # assume the root is one higher than last order assigned

        counter = counter + 1

    strahler_order=np.squeeze(strahler_order)

    elems_new[Ne-1, :] = elems[0, :]
    radii_new[Ne-1, :] = radii[0]
    len_new[Ne-1, :] = length[0]
    euclid_len_new[Ne-1, :] = euclid_length[0]

    elems_new=np.flip(elems_new,0)
    radii_new = np.flip(radii_new, 0)
    len_new = np.flip(len_new, 0)
    euclid_len_new = np.flip(euclid_len_new, 0)
    elems=geom['elems']
    elems[:,1:3]=elems_new
    return {'strahler_order': strahler_order, 'elems': elems, 'radii': radii_new,'length': len_new, 'euclidean length': euclid_len_new, 'nodes': nodes}

######
# This function joins up elements which attach with no branching by making them also terminal
# Inputs: list of elements, terminal elements and the number number for the node, and the previous node
# Outputs: updates terminal elements so that joining elements are also marked as terminal
######
def join_elements(terminal_elems, elems, elems_new, len_new, euclid_len_new, radii_new,length, euclid_length, radii, counter_new_order, nodeNum, nodeNumNew):

    Ne=len(elems)

    counter = 1
    places = np.where(elems == nodeNumNew) #find where the new node occurs
    ind1=places[0]
    ind2=places[1]

    while ((len(ind1) == 2)&(counter < Ne)): #as can only be present twice if a joining node

        #label element terminal as is a joining elements
        terminal_elems[ind1[0]]=1
        terminal_elems[ind1[1]]=1

        #see if branch joins to yet another branch, that we haven't yet encountered (i.e. not nodeNum)
        if (elems[ind1[0], ~ind2[0]]==nodeNum):
            k=1
        else:
            k=0

        #switch the way element pointd
        if (ind2[k] == 0):
            placeholder = np.copy(elems[ind1[k], 1])
            elems[ind1[k], 1] = int(np.copy(elems[ind1[k], 0]))
            elems[ind1[k], 0] = int(placeholder)
        nodeNum = nodeNumNew
        nodeNumNew = elems[ind1[k], 0]

        #if ismember(np.squeeze(elems[ind1[k],:]),elems_new[0:counter_new_order])==-1:
        elems_new[counter_new_order, :] = elems[ind1[k], :]
        radii_new[counter_new_order, :] = radii[ind1[k]]
        len_new[counter_new_order, :] = length[ind1[k]]
        euclid_len_new[counter_new_order, :] = euclid_length[ind1[k]]
        counter_new_order = counter_new_order + 1

        #update loop criteria
        places = np.where(elems == nodeNumNew)
        ind1 = places[0]
        ind2 = places[1]
        counter = counter + 1
    return (terminal_elems, elems_new, len_new, euclid_len_new, radii_new, counter_new_order)

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

         #for each end of the element
         for j in range(0,2):
             places=np.where(elems == elems[row,j])
             ind1=places[0]
             nParent=len(ind1)
             if (nParent > 1): #then this end attaches to a parent

                orderParent=0
                for k in range(0, nParent): #find highest order parent
                    order=orders[ind1[k]]
                    if (order>orderParent):
                        orderParent=order

                if (orderParent>threshold_order):
                    elems[row,:]=int(-1)

     # get rid of elements
     i = 0
     while (i < len(geom['radii'])):
          if (elems[i][0] < 0):
              # get rid of element, from elements and from other variables
              geom['length'] = np.delete(geom['length'], (i), axis=0)
              geom['euclidean length'] = np.delete(geom['euclidean length'], (i), axis=0)
              geom['radii'] = np.delete(geom['radii'], (i), axis=0)
              ordersAll['strahler']= np.delete(ordersAll['strahler'], (i), axis=0)
              ordersAll['generation'] = np.delete(ordersAll['generation'], (i), axis=0)
              geom['elems'] = np.delete(geom['elems'], (i),axis=0)
              elems = np.delete(elems, (i), axis=0)
          else:
              i=i+1
     return geom

######
# This function finds branch angles of nodes
# Inputs: list of nodes, elements and their orders
# Outputs: list of branch angles in radians, where the angle is the angle of a given element from its parents, and is zero if the element is a continuation of the parent
######
def find_branch_angles(nodes, elems, orders):

    elems = elems[:, 1:3] #get rid of useless first column

    num_node = len(nodes)
    num_elems = len(elems)
    branch_angles = -1.*np.ones(num_elems)  # initialise radius array

    for nn in range(0, num_node+1):

        #find elements at this node
        places=np.where(elems == nn)
        ind1=places[0]
        ind2=places[1]

        num_branches=len(ind1)

        if num_branches>2:  # then it is a branching node

            # find lower generation branch
            order_list=orders[ind1]
            order_min=min(order_list)
            n_min=np.where(order_list==order_min)
            n_min=n_min[0]
            for i in range(0, len(n_min)): # as if has two higher order parents, will find both and take lower order angle
                if len(n_min)>1:
                    print(len(n_min))
                nm = n_min[i]
                #find parent
                endNode=int(elems[ind1[nm], ~ind2[nm]])
                startNode=int(elems[ind1[nm], ind2[nm]])
                v_parent = nodes[endNode, :] - nodes[startNode,:]
                v_parent = v_parent / np.linalg.norm(v_parent)

                for nb in range (0, num_branches):
                    if (order_list[nb]>order_min):

                        #find daughter
                        endNode = int(elems[ind1[nb], ~ind2[nb]])
                        startNode = int(elems[ind1[nb], ind2[nb]])
                        v_daughter = nodes[startNode, :] - nodes[endNode, :]
                        v_daughter=v_daughter/np.linalg.norm(v_daughter)

                        #find angle, # as if has two higher order parents, will find both and take lower order angle
                        current_angle=branch_angles[ind1[nb]]
                        new_angle=np.arccos(np.dot(v_parent, v_daughter))
                        if (current_angle==-1)or(current_angle>new_angle):
                            branch_angles[ind1[nb]] = new_angle
    return branch_angles



######
# This function finds statistics on branching tree
# Inputs: list of various tree attributes
# Outputs: table of information by order
######
def summary_statistics(orders, length, euclid_length, radii, branch_angles):

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
    print(tabulate(table, headers=header))

    return table

######
# This function plots the branching tree in 3D
# Inputs: list of nodes, elements and a colour code
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

    #ax.set_aspect('equal')
    plt.show()
    return 0