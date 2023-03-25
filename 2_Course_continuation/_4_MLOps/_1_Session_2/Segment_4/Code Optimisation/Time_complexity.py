## Code Optimization Technique
import time
# Functions that takes a value and performs increement and decreement
# certain number of times
# Each time the variables are initialized within the loop
def increement_1(value, times):
    for i in range(0, times):
        increase_by = 100
        decrease_by = 50
        value += increase_by
        value -= decrease_by
# Here the variables are initialized outside the loop which makes it a 
# constant time complexicity O(1)
def increement_2(value, times):
    increase_by = 1000
    decrease_by = 50
    for i in range(0, times):
        value += increase_by
        value -= decrease_by
        
        
# Case_1
start_time = time.time()
increement_1(1000, 100000000)
print("increement_1 : --- {} ---".format(time.time() - start_time))
# Case_2
start_time = time.time()
increement_2(1000, 100000000)
print("increement_1 : --- {} ---".format(time.time() - start_time))