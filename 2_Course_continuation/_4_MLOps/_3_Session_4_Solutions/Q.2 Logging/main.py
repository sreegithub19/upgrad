import logging

# Function to iterate through the list and check ages 
def validate_age_data(ages):
    for value in ages:
        if value <= 0: 
            logger.error('Invalid age')
        elif value < 18:
            logger.debug('teenager')
        else: 
            logger.debug('adult')


if __name__ == '__main__':
    # Defining the structure of error message
    error_message_format = '%(asctime)s - %(name)s - %(levelname)s - %(message)s'
    
    # Setting up the basic configuration
    logging.basicConfig(filename = 'age_log.log', format = error_message_format, filemode = 'a')
    
    # Defining the logger object
    # Name can be given a logger object by passing as an argument, otherwise it takes the name as root
    logger = logging.getLogger('upGrad')
    
    # Set the level(Here we will set the level to DEBUG(10))
    # so that it can log itself and all other levels(INFO(20), WARNING(30) etc)
    logger.setLevel(logging.DEBUG)

    # Example_1:
    ages = [12, 22, 33, -6, 0, 15]
    validate_age_data(ages)
    
    # Example_2:
    ages = [12, 32, -55, 11, 61, 33]
    validate_age_data(ages)
