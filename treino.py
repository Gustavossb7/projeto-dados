import pandas as ps
import seaborn as sns
import matplotlib.pyplot as plt

base = ps.read_csv('titanic_train.csv')

median_age = base['Age'].median()
base['Age'] = base['Age'].fillna(median_age)
base_sorted = base.sort_values(by='Age')
print('A idade média é: ',median_age)
#print(base_sorted[['Name', 'Age']].head(10))

base_sorted_fare = base.sort_values(by='Fare',ascending=False)
print(base_sorted_fare[['Name', 'Fare']].head(50))