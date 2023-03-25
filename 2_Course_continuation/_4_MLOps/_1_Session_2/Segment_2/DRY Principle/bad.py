#plotting the distribution of Survived/Dead in the column 'Sex'
survived = train[train['Survived']==1]['Sex'].value_counts()
dead = train[train['Survived']==0]['Sex'].value_counts()
df = pd.DataFrame([survived,dead])
df.index = ['Survived','Dead']
df.plot(kind='bar',stacked=True, figsize=(10,5))

#plotting the distribution of Survived/Dead in the column 'Plass'
survived = train[train['Survived']==1]['Pclass'].value_counts()
dead = train[train['Survived']==0]['Pclass'].value_counts()
df = pd.DataFrame([survived,dead])
df.index = ['Survived','Dead']
df.plot(kind='bar',stacked=True, figsize=(10,5))
