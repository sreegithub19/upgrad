# Importing pytest module for writing unittests
# Importing main module which contains the various functions related to an Employee

import pytest
from main import *

# Creating Global Objects
employee_1 = create_employee_profile('Steve', 'Gates', date(1999, 1, 8), 50000, 'Technical')
employee_2 = create_employee_profile('Suyash', 'Singh', date(1995, 4, 22), 60000, 'Electrical')
display_employee_details(employee_1)
# Note: All the functions need to start with 'test_' or else it will be ignored

def test_increase_employee_salary():
    assert increase_employee_salary(employee_1, 30) == 65000, 'Incorrect Salary Calculation'

def test_get_age():
    assert get_age(employee_1) == 23, 'Incorrect age calculation'

# Checks if the user is in valid department
def test_get_department():
    assert get_department(employee_2) in departments, 'Invalid Department'

# This function will be ignored by pytest as it does not begin with 'test_'
def check_email_id():
    assert get_email(employee_2) == 'SuyashSingh@gmail.com', 'Email id is not correct'
    


