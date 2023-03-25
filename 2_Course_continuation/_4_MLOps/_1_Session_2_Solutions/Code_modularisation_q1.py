def get_greater_value(array, value):
    for var in array:
        if var > value:
            print(var)
            break
        else:
            continue
    else:
        print('No value greater than given {} found'.format(value))

# Post defining function we can resuse it for multiple arrays and value
get_greater_value([12, 22, 33, 44, 55, 46], 46) 
get_greater_value([12, 32, 43, 32, 64, 75, 32], 332)