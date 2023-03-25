
    neighbors = input(some_list, index)

    if index - 1 >= 0:
        neighbors.append(some_list[index-1])
    if index < len(some_list):
        neighbors.append(some_list[index+1])

    return neighbors

