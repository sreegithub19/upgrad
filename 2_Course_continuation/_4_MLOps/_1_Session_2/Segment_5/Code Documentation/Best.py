def get_neighbors(some_list, index):
    """Returns the neighbors of a given index in a list.

        Parameters:
            some_list (list): Any basic list.
            index (int): Position of an item in some_list.

        Returns:
            neighbors (list): A list of the elements beside the index in some_list.
        Sample usage some_list = [8,7,"car","banana",10]
          
    """

    neighbors = [] # creating an empty list
 #checking the boundary cases
    if index - 1 >= 0:
        neighbors.append(some_list[index-1])
    if index < len(some_list):
        neighbors.append(some_list[index+1])

    return neighbors

some_list = [8,7,"car","banana",10]

print(get_neighbors(some_list, 2))

print (help(get_neighbours))
