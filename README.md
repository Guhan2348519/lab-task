# lab-task 
1.first  we import python packages
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
     
2. we mount drive and import the csv file 
churn_data=pd.read_csv('/content/drive/MyDrive/Churn_Modelling - Churn_Modelling.csv')
     
3.then we display first 10 columns
churn_data.head(10)
     
RowNumber	CustomerId	Surname	CreditScore	Geography	Gender	Age	Tenure	Balance	NumOfProducts	HasCrCard	IsActiveMember	EstimatedSalary	Exited
0	1	15634602	Hargrave	619	France	Female	42	2	0.00	1	1	1	101348.88	1
1	2	15647311	Hill	608	Spain	Female	41	1	83807.86	1	0	1	112542.58	0
2	3	15619304	Onio	502	France	Female	42	8	159660.80	3	1	0	113931.57	1
3	4	15701354	Boni	699	France	Female	39	1	0.00	2	0	0	93826.63	0
4	5	15737888	Mitchell	850	Spain	Female	43	2	125510.82	1	1	1	79084.10	0
5	6	15574012	Chu	645	Spain	Male	44	8	113755.78	2	1	0	149756.71	1
6	7	15592531	Bartlett	822	France	Male	50	7	0.00	2	1	1	10062.80	0
7	8	15656148	Obinna	376	Germany	Female	29	4	115046.74	4	1	0	119346.88	1
8	9	15792365	He	501	France	Male	44	4	142051.07	2	0	1	74940.50	0
9	10	15592389	H?	684	France	Male	27	2	134603.88	1	1	1	71725.73	0

4. it tells no of rows and columns of the specified dataset
churn_data.shape
(10000, 14)

5. it gives info of the dataset
churn_data.info()
     
<class 'pandas.core.frame.DataFrame'>
RangeIndex: 10000 entries, 0 to 9999
Data columns (total 14 columns):
 #   Column           Non-Null Count  Dtype  
---  ------           --------------  -----  
 0   RowNumber        10000 non-null  int64  
 1   CustomerId       10000 non-null  int64  
 2   Surname          10000 non-null  object 
 3   CreditScore      10000 non-null  int64  
 4   Geography        10000 non-null  object 
 5   Gender           10000 non-null  object 
 6   Age              10000 non-null  int64  
 7   Tenure           10000 non-null  int64  
 8   Balance          10000 non-null  float64
 9   NumOfProducts    10000 non-null  int64  
 10  HasCrCard        10000 non-null  int64  
 11  IsActiveMember   10000 non-null  int64  
 12  EstimatedSalary  10000 non-null  float64
 13  Exited           10000 non-null  int64  
dtypes: float64(2), int64(9), object(3)
memory usage: 1.1+ MB

6. to check the dataset has empty or null value in it  
churn_data.isnull().sum()
     
RowNumber          0
CustomerId         0
Surname            0
CreditScore        0
Geography          0
Gender             0
Age                0
Tenure             0
Balance            0
NumOfProducts      0
HasCrCard          0
IsActiveMember     0
EstimatedSalary    0
Exited             0
dtype: int64

7. to describe about dataset like mean std count 
churn_data.describe()

     
RowNumber	CustomerId	CreditScore	Age	Tenure	Balance	NumOfProducts	HasCrCard	IsActiveMember	EstimatedSalary	Exited
count	10000.00000	1.000000e+04	10000.000000	10000.000000	10000.000000	10000.000000	10000.000000	10000.00000	10000.000000	10000.000000	10000.000000
mean	5000.50000	1.569094e+07	650.528800	38.921800	5.012800	76485.889288	1.530200	0.70550	0.515100	100090.239881	0.203700
std	2886.89568	7.193619e+04	96.653299	10.487806	2.892174	62397.405202	0.581654	0.45584	0.499797	57510.492818	0.402769
min	1.00000	1.556570e+07	350.000000	18.000000	0.000000	0.000000	1.000000	0.00000	0.000000	11.580000	0.000000
25%	2500.75000	1.562853e+07	584.000000	32.000000	3.000000	0.000000	1.000000	0.00000	0.000000	51002.110000	0.000000
50%	5000.50000	1.569074e+07	652.000000	37.000000	5.000000	97198.540000	1.000000	1.00000	1.000000	100193.915000	0.000000
75%	7500.25000	1.575323e+07	718.000000	44.000000	7.000000	127644.240000	2.000000	1.00000	1.000000	149388.247500	0.000000
max	10000.00000	1.581569e+07	850.000000	92.000000	10.000000	250898.090000	4.000000	1.00000	1.000000	199992.480000	1.000000

8.to create a dataframe and display it
df =pd.DataFrame(churn_data)
print(df)
     
      RowNumber  CustomerId    Surname  CreditScore Geography  Gender  Age  \
0             1    15634602   Hargrave          619    France  Female   42   
1             2    15647311       Hill          608     Spain  Female   41   
2             3    15619304       Onio          502    France  Female   42   
3             4    15701354       Boni          699    France  Female   39   
4             5    15737888   Mitchell          850     Spain  Female   43   
...         ...         ...        ...          ...       ...     ...  ...   
9995       9996    15606229   Obijiaku          771    France    Male   39   
9996       9997    15569892  Johnstone          516    France    Male   35   
9997       9998    15584532        Liu          709    France  Female   36   
9998       9999    15682355  Sabbatini          772   Germany    Male   42   
9999      10000    15628319     Walker          792    France  Female   28   

      Tenure    Balance  NumOfProducts  HasCrCard  IsActiveMember  \
0          2       0.00              1          1               1   
1          1   83807.86              1          0               1   
2          8  159660.80              3          1               0   
3          1       0.00              2          0               0   
4          2  125510.82              1          1               1   
...      ...        ...            ...        ...             ...   
9995       5       0.00              2          1               0   
9996      10   57369.61              1          1               1   
9997       7       0.00              1          0               1   
9998       3   75075.31              2          1               0   
9999       4  130142.79              1          1               0   

      EstimatedSalary  Exited  
0           101348.88       1  
1           112542.58       0  
2           113931.57       1  
3            93826.63       0  
4            79084.10       0  
...               ...     ...  
9995         96270.64       0  
9996        101699.77       0  
9997         42085.58       1  
9998         92888.52       1  
9999         38190.78       0  

[10000 rows x 14 columns]

9. to perform label encoder for grnder and geography to convert categorial to numeric data
from sklearn.preprocessing import LabelEncoder

label_encoder = LabelEncoder()
churn_data['Gender'] = label_encoder.fit_transform(churn_data['Gender'])


     

from sklearn.preprocessing import LabelEncoder
label_encoder = LabelEncoder()
df['Geography'] = label_encoder.fit_transform(df['Geography'])
print(churn_data)

     
      RowNumber  CustomerId    Surname  CreditScore  Geography  Gender  Age  \
0             1    15634602   Hargrave          619          0       0   42   
1             2    15647311       Hill          608          2       0   41   
2             3    15619304       Onio          502          0       0   42   
3             4    15701354       Boni          699          0       0   39   
4             5    15737888   Mitchell          850          2       0   43   
...         ...         ...        ...          ...        ...     ...  ...   
9995       9996    15606229   Obijiaku          771          0       1   39   
9996       9997    15569892  Johnstone          516          0       1   35   
9997       9998    15584532        Liu          709          0       0   36   
9998       9999    15682355  Sabbatini          772          1       1   42   
9999      10000    15628319     Walker          792          0       0   28   

      Tenure    Balance  NumOfProducts  HasCrCard  IsActiveMember  \
0          2       0.00              1          1               1   
1          1   83807.86              1          0               1   
2          8  159660.80              3          1               0   
3          1       0.00              2          0               0   
4          2  125510.82              1          1               1   
...      ...        ...            ...        ...             ...   
9995       5       0.00              2          1               0   
9996      10   57369.61              1          1               1   
9997       7       0.00              1          0               1   
9998       3   75075.31              2          1               0   
9999       4  130142.79              1          1               0   

      EstimatedSalary  Exited  
0           101348.88       1  
1           112542.58       0  
2           113931.57       1  
3            93826.63       0  
4            79084.10       0  
...               ...     ...  
9995         96270.64       0  
9996        101699.77       0  
9997         42085.58       1  
9998         92888.52       1  
9999         38190.78       0  

[10000 rows x 14 columns]

10. clean the dataset by removing unwanted dataset
df =  df.drop(['RowNumber', 'CustomerId', 'Surname'], axis=1)
     

df.head(10)
     
CreditScore	Geography	Gender	Age	Tenure	Balance	NumOfProducts	HasCrCard	IsActiveMember	EstimatedSalary	Exited
0	619	0	0	42	2	0.00	1	1	1	101348.88	1
1	608	2	0	41	1	83807.86	1	0	1	112542.58	0
2	502	0	0	42	8	159660.80	3	1	0	113931.57	1
3	699	0	0	39	1	0.00	2	0	0	93826.63	0
4	850	2	0	43	2	125510.82	1	1	1	79084.10	0
5	645	2	1	44	8	113755.78	2	1	0	149756.71	1
6	822	0	1	50	7	0.00	2	1	1	10062.80	0
7	376	1	0	29	4	115046.74	4	1	0	119346.88	1
8	501	0	1	44	4	142051.07	2	0	1	74940.50	0
9	684	0	1	27	2	134603.88	1	1	1	71725.73	0

11. plotting graph tenure against exited 

plt.figure(figsize=(10, 6))
sns.countplot(x='Tenure', hue='Exited', data=churn_data)
plt.title('Tenure vs Churn')
plt.xlabel('Tenure')
plt.ylabel('Count')
plt.legend(title='Exited', labels=['Not Churned', 'Churned'])
plt.show()
     
test=pd.crosstab(churn_data['Tenure'],churn_data['Exited'])
print(test)
     
Exited    0    1
Tenure          
0       318   95
1       803  232
2       847  201
3       796  213
4       786  203
5       803  209
6       771  196
7       851  177
8       828  197
9       771  213
10      389  101


12. perform binning to age group
age_bins = [0, 18, 25, 35, 50, float('inf')]
age_labels = ['<18', '18-25', '26-35', '36-50', '50+']
churn_data['AgeGroup'] = pd.cut(churn_data['Age'], bins=age_bins, labels=age_labels, right=False)
print(churn_data[['Age', 'AgeGroup']])
     
      Age AgeGroup
0      42    36-50
1      41    36-50
2      42    36-50
3      39    36-50
4      43    36-50
...   ...      ...
9995   39    36-50
9996   35    36-50
9997   36    36-50
9998   42    36-50
9999   28    26-35

[10000 rows x 2 columns]


13. to perform chisquare
from sklearn.feature_selection import chi2
X=df.drop(columns=['Exited'])
Y=df['Exited']

     

chi_scores=chi2(X,Y)
     

chi_scores
     
(array([1.05403468e+02, 1.18532506e+01, 5.15399263e+01, 2.30041748e+03,
        3.27053797e+00, 7.15130278e+06, 5.05539429e+00, 1.50040970e-01,
        1.18199414e+02, 4.83508818e+04]),
 array([9.96353608e-25, 5.75607838e-04, 7.01557451e-13, 0.00000000e+00,
        7.05344899e-02, 0.00000000e+00, 2.45493956e-02, 6.98496209e-01,
        1.56803624e-27, 0.00000000e+00]))

chi_values=pd.Series(chi_scores[0],index=X.columns)
chi_values.sort_values(ascending=False,inplace=True)
chi_values.plot.bar()
     
<Axes: >

14. to perform minmax scaling
churn_numeric = df
from sklearn.preprocessing import MinMaxScaler

scaler = MinMaxScaler()
scaled_data = scaler.fit_transform(df)
     

scaled_df = pd.DataFrame(scaled_data)
     

display (scaled_df)
     
0	1	2	3	4	5	6	7	8	9	10
0	0.538	0.0	0.0	0.324324	0.2	0.000000	0.000000	1.0	1.0	0.506735	1.0
1	0.516	1.0	0.0	0.310811	0.1	0.334031	0.000000	0.0	1.0	0.562709	0.0
2	0.304	0.0	0.0	0.324324	0.8	0.636357	0.666667	1.0	0.0	0.569654	1.0
3	0.698	0.0	0.0	0.283784	0.1	0.000000	0.333333	0.0	0.0	0.469120	0.0
4	1.000	1.0	0.0	0.337838	0.2	0.500246	0.000000	1.0	1.0	0.395400	0.0
...	...	...	...	...	...	...	...	...	...	...	...
9995	0.842	0.0	1.0	0.283784	0.5	0.000000	0.333333	1.0	0.0	0.481341	0.0
9996	0.332	0.0	1.0	0.229730	1.0	0.228657	0.000000	1.0	1.0	0.508490	0.0
9997	0.718	0.0	0.0	0.243243	0.7	0.000000	0.000000	0.0	1.0	0.210390	1.0
9998	0.844	0.5	1.0	0.324324	0.3	0.299226	0.333333	1.0	0.0	0.464429	1.0
9999	0.884	0.0	0.0	0.135135	0.4	0.518708	0.000000	1.0	0.0	0.190914	0.0
10000 rows × 11 columns

15. to perform standard scaler
from sklearn.preprocessing import StandardScaler

scaler = StandardScaler()
scaled_data = scaler.fit_transform(df)

scaled_df = pd.DataFrame(scaled_data)
display(scaled_df)
     
0	1	2	3	4	5	6	7	8	9	10
0	-0.326221	-0.901886	-1.095988	0.293517	-1.041760	-1.225848	-0.911583	0.646092	0.970243	0.021886	1.977165
1	-0.440036	1.515067	-1.095988	0.198164	-1.387538	0.117350	-0.911583	-1.547768	0.970243	0.216534	-0.505775
2	-1.536794	-0.901886	-1.095988	0.293517	1.032908	1.333053	2.527057	0.646092	-1.030670	0.240687	1.977165
3	0.501521	-0.901886	-1.095988	0.007457	-1.387538	-1.225848	0.807737	-1.547768	-1.030670	-0.108918	-0.505775
4	2.063884	1.515067	-1.095988	0.388871	-1.041760	0.785728	-0.911583	0.646092	0.970243	-0.365276	-0.505775
...	...	...	...	...	...	...	...	...	...	...	...
9995	1.246488	-0.901886	0.912419	0.007457	-0.004426	-1.225848	0.807737	0.646092	-1.030670	-0.066419	-0.505775
9996	-1.391939	-0.901886	0.912419	-0.373958	1.724464	-0.306379	-0.911583	0.646092	0.970243	0.027988	-0.505775
9997	0.604988	-0.901886	-1.095988	-0.278604	0.687130	-1.225848	-0.911583	-1.547768	0.970243	-1.008643	1.977165
9998	1.256835	0.306591	0.912419	0.293517	-0.695982	-0.022608	0.807737	0.646092	-1.030670	-0.125231	1.977165
9999	1.463771	-0.901886	-1.095988	-1.041433	-0.350204	0.859965	-0.911583	0.646092	-1.030670	-1.076370	-0.505775
10000 rows × 11 columns


16. to perform pca
from sklearn.decomposition import PCA
pca = PCA(n_components=0.95)
reduced_data = pca.fit_transform(scaled_data)
display (reduced_data)
     
array([[ 9.26618826e-01,  1.35284070e+00,  5.52103724e-01, ...,
        -3.29434608e-01, -7.06310801e-01,  1.88090366e+00],
       [ 6.45097306e-01, -4.87404190e-01,  1.48535586e+00, ...,
        -1.12558091e+00, -8.68052083e-01, -3.79681424e-02],
       [ 9.50532532e-01,  2.36588329e+00, -1.80474261e+00, ...,
        -6.57994952e-01,  2.79791760e+00, -4.37556348e-01],
       ...,
       [ 6.35398907e-01,  1.03380886e+00,  6.25839634e-01, ...,
        -9.99943398e-01, -5.39624853e-01,  2.27410290e+00],
       [ 8.69185156e-01,  1.14446268e+00, -2.96066321e-01, ...,
         2.16225569e+00,  6.71849041e-01,  6.79273323e-01],
       [ 1.92434978e-01, -1.07041560e+00, -1.02009394e+00, ...,
        -2.18754749e-01, -2.03351047e-03, -3.40978626e-01]])


17.to perform crosstab tenure against exited
test=pd.crosstab(churn_data['Tenure'],churn_data['Exited'])
print(test)
     
Exited    0    1
Tenure          
0       318   95
1       803  232
2       847  201
3       796  213
4       786  203
5       803  209
6       771  196
7       851  177
8       828  197
9       771  213
10      389  101


18. performing heatmap for visualization of dataframe
plt.figure(figsize=(11, 10))
sns.heatmap(df.corr(), annot=True,  center=0)
plt.title('Correlation Heatmap')
plt.show()
