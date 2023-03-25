# Importing required libraries
from datetime import date

# Global objects
departments = ['Technical', 'Finance', 'Research', 'Sales']

# Function to create an Employee profile 
def create_employee_profile(f_name, l_name, dob, salary, department):
    return {
        'Name' : f_name + l_name,
        'Date_of_birth' : dob,
        'Salary' : salary,
        'Department' : department,
        'Email' : f_name + l_name + '@gmail.com'
    }


# Function to displays the details of an employee
def display_employee_details(e_object):
    for key,value in e_object.items():
        print('{} : {}'.format(key, value))


# Function to increase salary of an employee on passed percent        
def increase_employee_salary(e_object, percent):
    increased_amount = (e_object['Salary'] * percent) / 100
    e_object['Salary'] += increased_amount
    return e_object['Salary']
    
    
# Function to calculate the age of an employee on basis of his DOB
def get_age(e_object):
    today = date.today()
    is_birthday_passed = ((today.month, today.day) < (e_object['Date_of_birth'].month, e_object['Date_of_birth'].day))
    return today.year - e_object['Date_of_birth'].year - is_birthday_passed


# Function to return department of an Employee
def get_department(e_object):
    return e_object['Department']


# Function to return email of an Employee
def get_email(e_object):
    return e_object['Email']
