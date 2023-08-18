import inline as inline
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns

plt.style.use('ggplot')
from matplotlib.pyplot import figure

plt.show()
plt.rcParams['figure.figsize'] = (12, 8)

Data =pd.read_csv("D:\Data Analysis Practice\movies.csv")
print(Data)
print(Data.head())

for col in Data.columns:
    perc_missing = np.mean(Data[col].isnull())
    print('{} - {}%'.format(col, perc_missing))

print(Data.dtypes)

Data['budget'] = Data['budget'].fillna(0)
Data['gross'] = Data['gross'].fillna(0)
Data['budget'] = Data['budget'].astype("Int64")
Data['gross'] = Data['gross'].astype("Int64")

print(Data)

Data['month'] = Data['released'].astype(str).str[:4]
print(Data)

Data = Data.sort_values(by='gross', ascending=False)
print(Data)
pd.set_option('display.max_rows', None)
pd.set_option('display.max_columns', None)
print(Data)
Data['company'].drop_duplicates().sort_values(ascending=False)
print(Data.head())

Data['budget'] = Data['budget'].astype(float)
Data['gross'] = Data['gross'].astype(float)

plt.scatter(x=Data['budget'], y=Data['gross'])
plt.title('Budget vs Gross')
plt.xlabel('Gross Earnings')
plt.ylabel('Budget for Film')

plt.show()
print(Data.head())
sns.regplot(x='budget', y='gross', data=Data, scatter_kws=({"color":"red"}), line_kws=({"color":"blue"}))

plt.show()
numeric_data = Data.select_dtypes(include=['number'])
correlation_matrix = numeric_data.corr(method='pearson')
print(correlation_matrix)

plt.title('Correlation matrix for numeric features')
plt.xlabel('Movie Features')
plt.ylabel('Movie Features')

sns.heatmap(correlation_matrix, annot=True)

plt.show()

Data_numerized= Data

for col_name in Data_numerized.columns:
    if Data_numerized[col_name].dtype == 'object':
        Data_numerized[col_name] = Data_numerized[col_name].astype('category')
        Data_numerized[col_name] = Data_numerized[col_name].cat.codes

print(Data_numerized)

numeric_data = Data_numerized.select_dtypes(include=['number'])
correlation_matrix = numeric_data.corr(method='pearson')
print(correlation_matrix)

plt.title('Correlation matrix for numeric features')
plt.xlabel('Movie Features')
plt.ylabel('Movie Features')

sns.heatmap(correlation_matrix, annot=True)

plt.show()

correlation_matrix_numerized= Data_numerized.corr()
cor_pairs = correlation_matrix_numerized.unstack()
print(cor_pairs)

sorted_pairs = cor_pairs.sort_values()
high_corr= sorted_pairs[(sorted_pairs)>0.5]
print(high_corr)