#Created a function to replace the above code

def bar_chart(feature):
    ''' Plots a graph on the basis of feature passed as an input

        Parameters:
            feature(string): Represents the particular column of the
                             Data set 'titanic.csv' on the basis of which
                             graph is plotted

        Returns:
            None
    '''
    survived = train[train['Survived']==1][feature].value_counts()
    dead = train[train['Survived']==0][feature].value_counts()
    df = pd.DataFrame([survived,dead])
    df.index = ['Survived','Dead']
    df.plot(kind='bar',stacked=True, figsize=(10,5))

'''
Multi Line Comment:
To access the docstring of a particular object, you can use the help 
function. For eg: 
help(bar_chart)
'''