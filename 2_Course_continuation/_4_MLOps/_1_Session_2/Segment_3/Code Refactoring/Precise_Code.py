## Case 1: List Comprehensions in place of for loop
# Code to create a new squared_list of numbers based on conditon(if odd)
my_numbers = [1,2,3,4,5]
'''
sqrd_list = []
for item in my_numbers:
  if item % 2 == 1:
    sqrd_list.append(item ** 2)
'''
# The above four lines can be replaced by a single line
sqrd_list = [value ** 2 for value in my_numbers if value % 2 == 1]
print(sqrd_list)

## Case 2: Value Assignment Using tuple unpacking
'''
name = 'Steve'
age = 44
roll_no = 28
'''
# The above task can be easily done using tuple unpacking
# on a single line
name, age, roll_no = 'Steve', 44, 28

## Case 3: Using any and all functions
# Old code to check if any even number present
num_list = [12, 22, 33, 44, 55]
'''
for value in num_list:
    if value % 2 == 0:
        is_even_present = True
        break
'''
is_even_present = any(value % 2 == 0 for value in num_list)

# Case 4: Lambda Function
## A Simple function to calculate_average
'''
def get_greater_number(num_1, num_2):
    if num_1 > num_2:
        return num_1
    return num_2
'''
# You can use lambda(anonymous function) to write this in one line
 
get_greater = lambda num_1, num_2 : num_1 if num_1 > num_2 else num_2
print(get_greater(12, 22))