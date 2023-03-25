def get_array_input(array, size):
    value = 0
    for i in range(size):
        value = int(input())
        array.append( value )
 
def check_if_even(array):
    is_even = []
    for values in array:
        if array %  2 == 0:
            is_even.append('True')
        else:
            is_even.append('False')