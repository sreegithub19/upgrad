def get_squared_values(inp_list):
    sqrd_list = [] # Initialising Empty list
    dig_sum = 0
    
    for value in inp_list:
        try:
            # Converts value to integer if fails goes to the except block
            value = int(value)  
            while value > 0:
                dig_sum += value % 10 # Extracts the last digit
                value //= 10          # Removes the last digit
            sqrd_list.append(dig_sum) # Appends the result
            dig_sum = 0 # Again reinitialises dig_sum variable to 0
            
        except ValueError as v_e:
            sqrd_list.append('NAN') # Appends NAN for strings
            
    return sqrd_list

# Sample Input: [‘22’, ‘Sam’, ‘33’, ‘44’]
# Sample Output: [4, ‘NAN’, 6, 8]

print(get_squared_values(['22', 'Sam', '33', '44']))