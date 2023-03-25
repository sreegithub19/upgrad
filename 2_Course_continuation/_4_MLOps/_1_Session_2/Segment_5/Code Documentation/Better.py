def get_neighbors(some_list, index):
# Returns the neighbors of a given index in a list
    neighbors = []

    if index - 1 >= 0:
        neighbors.append(some_list[index-1])
    if index < len(some_list):
        neighbors.append(some_list[index+1])

    return neighbors

some_list = [8,7,"car","banana",10]

print(get_neighbors(some_list, 2))

