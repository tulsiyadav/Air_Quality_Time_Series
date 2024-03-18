#!/usr/bin/env python
# coding: utf-8

# In[7]:


import numpy as np # linear algebra
import pandas as pd # data processing, CSV file I/O (e.g. pd.read_csv)
import seaborn as sns
from matplotlib import pyplot as plt
from scipy.stats import chi2_contingency
from scipy import stats
from sklearn.model_selection import train_test_split
from sklearn.metrics import classification_report,accuracy_score
from sklearn.neighbors import KNeighborsClassifier
from sklearn.tree import DecisionTreeClassifier
from sklearn.cluster import KMeans
from sklearn.linear_model import Perceptron
from sklearn.metrics import accuracy_score, precision_score, recall_score, f1_score, roc_auc_score, roc_curve
import matplotlib.pyplot as plt
from sklearn.model_selection import train_test_split
from sklearn.metrics import roc_curve, auc
from sklearn.neural_network import MLPClassifier

from sklearn.model_selection import GridSearchCV

from sklearn.model_selection import train_test_split
from sklearn.linear_model import LinearRegression
from sklearn.metrics import mean_squared_error, mean_absolute_error, r2_score



# In[8]:


df = pd.read_csv("C:/Users/tulsi/OneDrive/Documents/AirQuality.csv", delimiter=';')
df
sub = pd.read_csv("C:/Users/tulsi/OneDrive/Documents/AirQuality.csv", delimiter=';')
df.head(400)


# In[9]:


df['NMHC(GT)'].fillna(df['NMHC(GT)'].mode()[0], inplace=True)
df


# In[10]:


df.describe()


# In[11]:


df.dtypes


# In[12]:


dff = df.iloc[:9001]


# In[13]:


df['NMHC(GT)'].fillna(df['NMHC(GT)'].mean(), inplace=True)


# In[14]:


df['NMHC(GT)'].fillna(df['NMHC(GT)'].mean(), inplace=True)


# In[15]:


df.isna().sum()


# In[16]:


sns.histplot(data=df.loc[:10,'CO(GT)'])


# In[17]:


sub


# In[18]:


plt.pie(sub['NMHC(GT)'].value_counts(), labels = sub['NMHC(GT)'].value_counts().index.tolist(), autopct='%.0f%%')
plt.show()


# In[19]:


sub.describe()


# In[20]:


df


# In[21]:


print(sub['NMHC(GT)'])


# In[23]:


selected_columns = ['PT08.S1(CO)', 'NMHC(GT)', 'PT08.S2(NMHC)', 'NOx(GT)', 'PT08.S3(NOx)', 'PT08.S5(O3)']
df_selected = sub[selected_columns]

# Вычисление матрицы корреляции
correlation_matrix = df_selected.corr()

# Визуализация матрицы корреляции с использованием seaborn
plt.figure(figsize=(8, 6))
sns.heatmap(correlation_matrix, annot=True, cmap='coolwarm', fmt=".2f")
plt.title("Матрица корреляции")
plt.show()


# In[24]:


sub['RH'] = sub['RH'].str.replace(',', '.').astype(float)
sub['CO(GT)'] = sub['CO(GT)'].str.replace(',', '.').astype(float)
df_22 =  sub[sub['CO(GT)'] == 2.2].loc[:,'RH']
df_2 =   sub[sub['CO(GT)'] == 2.0].loc[:,'RH']
df_22
sub.drop(columns=['Unnamed: 15', 'Unnamed: 16'])
sub


# In[25]:


sns.histplot(data = df_22, kde = True)
plt.show()


# In[26]:


sns.histplot(data = df_2, kde =True)
plt.show()


# In[27]:


print(len(df_22.axes[0]))


# In[28]:


print(len(df_2.axes[0]))


# In[29]:


df_22


# In[30]:


sub=sub[['CO(GT)','PT08.S1(CO)', 'NMHC(GT)','PT08.S2(NMHC)','NOx(GT)','PT08.S3(NOx)','NO2(GT)','PT08.S4(NO2)','PT08.S5(O3)','RH']]
sub.dropna(inplace=True) 
sub


# In[31]:


corr = sub.corr()
corr.style.background_gradient(cmap='coolwarm')


# In[33]:


def map_to_category(value):
    if 0.0 <= value < 1.0:
        return 'Low'
    elif 1.0 <= value < 2.0:
        return 'Medium'
    elif 2.5 < value < 4:
        return 'High'
    
    
def to_category(value):
    if 100.0 <= value < 119.0:
        return '100 den ko'
    elif 120.0 <= value < 140.0:
        return '120dan kop'
    elif 141.0 < value < 200.0:
        return '141kop'
    
# Apply the custom mapping function to create a new column
sub['CO(GT) Category'] = sub['CO(GT)'].apply(map_to_category)
sub['NOx(GT) Category'] = sub['NOx(GT)'].apply(to_category)


contingency_table = pd.crosstab(sub['CO(GT) Category'].values, sub['NOx(GT) Category'].values)
print(contingency_table)
print()
chi2, p, dof, expected = stats.chi2_contingency(contingency_table)
print(f"Статистика хи-квадрат: {chi2}")
print(f"P-значение: {p}")
print(f"Степень свободы: {dof}")



N= 10

df_co =  sub[sub['CO(GT) Category'] == 'High'].loc[:,'RH']
df_nm =   sub[sub['CO(GT) Category'] == 'Low'].loc[:,'RH']

# Calculating the Standard Deviation 
# Calculating the variance to get the standard deviation 
var_x = df_co.var(ddof = 1) 
var_y = df_nm.var(ddof = 1) 
# Standard Deviation 
SD = np.sqrt((var_x + var_y) / 2) 
print("Standard Deviation =", SD) 
# Calculating the T-Statistics 
tval =(df_22.mean() - df_2.mean()) /(SD * np.sqrt(2 / N)) 
# Comparing with the critical T-Value 
# Degrees of freedom 
dof = 2 * N - 2 
# p-value after comparison with the T-Statistics 
pval = 1 - stats.t.cdf( tval, df = dof) 
print("t = " + str(tval)) 
print("p = " + str(2 * pval)) 
## Cross Checking using the internal function from SciPy Package 
tval2, pval2 = stats.ttest_ind(df_22, df_2) 
print("t = " + str(tval2)) 
print("p = " + str(pval2))


# In[34]:


sub.dropna(inplace=True)  # This will remove rows with NaN values from the DataFrame

X = sub['NOx(GT)'].values.reshape(-1, 1)
y = sub['NO2(GT)'].values


# In[35]:


sub
plt.pie(sub['CO(GT) Category'].value_counts(), labels = sub['CO(GT) Category'].value_counts().index.tolist(), autopct='%.0f%%')
plt.show()


# In[36]:


X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)
# Создайте модель линейной регрессии
model = LinearRegression()
# Обучите модель на тренировочных данных
model.fit(X_train, y_train)

коэффициент_наклона = model.coef_[0]
коэффициент_пересечения = model.intercept_
print(f"Коэффициент наклона: {коэффициент_наклона}")
print(f"Коэффициент пересечения: {коэффициент_пересечения}")
предсказания = model.predict(X_test)

plt.scatter(X_test, y_test, color='blue', label='Реальные данные')
plt.plot(X_test, предсказания, color='red', linewidth=2, label='Предсказанные данные')
y_pred = model.predict(X_test)

mse = mean_squared_error(y_test, y_pred)
 # болжаған мән мен шыққан мәндер арасында айырмашылық   Төмен MSE мәндері үлгінің жақсырақ сәйкестігін көрсетеді.
mae = mean_absolute_error(y_test, y_pred) 
#MSE-мен салыстырғанда шектен тыс көрсеткіштерге азырақ сезімтал.
r2 = r2_score(y_test, y_pred)
#R2 детерминация коэффициенті ретінде белгілі, модельдегі тәуелсіз айнымалылармен түсіндірілетін тәуелді айнымалыдағы (y) дисперсияның үлесін білдіреді.
# Ол 0-ден 1-ге дейін ауытқиды, мұнда 0 модель ешқандай дисперсияны түсіндірмейтінін көрсетеді, ал 1 тамаша сәйкестікті көрсетеді.

print("Mean Squared Error (MSE):", mse)  
print("Mean Absolute Error (MAE):", mae)  
print("R-squared (R2):", r2)     

# Добавим столбец с "1" в X_train и X_test (смещение/intercept)
X_train_with_bias = np.c_[np.ones((X_train.shape[0], 1)), X_train]
X_test_with_bias = np.c_[np.ones((X_test.shape[0], 1)), X_test]

# Решение уравнения линейной регрессии вручную
theta = np.linalg.inv(X_train_with_bias.T.dot(X_train_with_bias)).dot(X_train_with_bias.T).dot(y_train)
# матрицаның инверсия  есептейді
#матрицаның жолдары мен бағандарын аударады.X_train_with_bias.T


# Сделайте предсказания на тестовых данных
y_pred_manual = X_test_with_bias.dot(theta)





plt.legend()
plt.xlabel('Признак')
plt.ylabel('Результат')
plt.title('Предсказание модели линейной регрессии')
plt.show()


# In[37]:


sub = sub.drop(columns=['NOx(GT) Category'])
sub


# In[38]:


from sklearn.preprocessing import StandardScaler
from sklearn.preprocessing import LabelEncoder

le = LabelEncoder()
sub['CO(GT) Category'] = le.fit_transform(sub['CO(GT) Category'])
#бұл мәндерді сандық белгілерге түрлендіру

scaler = StandardScaler()
scaler.fit(sub.drop('CO(GT) Category', axis=1))
scaled_features = scaler.transform(sub.drop('CO(GT) Category', axis=1))
#орташа және стандартты ауытқуды есептеңіз

plt.pie(sub['CO(GT) Category'].value_counts(), labels = sub['CO(GT) Category'].value_counts().index.tolist(), autopct='%.0f%%')
plt.show()


# In[39]:


scaled_data = pd.DataFrame(scaled_features, columns = sub.drop('CO(GT) Category', axis=1).columns)
#"CO(GT)" бағанынан басқа бастапқы DataFrame ішкі ішінен барлық бағандарды таңданады.
from sklearn.model_selection import train_test_split


# In[40]:


x = scaled_data
y = sub['CO(GT) Category']

x_training_data, x_test_data, y_training_data, y_test_data = train_test_split(x, y, test_size = 0.3)


# In[41]:


from sklearn.neighbors import KNeighborsClassifier


# In[42]:


model = KNeighborsClassifier(n_neighbors = 1)
model.fit(x_training_data, y_training_data)


# In[43]:


predictions = model.predict(x_test_data)
#сынақ деректеріне болжам жасау
predictions


# In[44]:


#Gaussian Naive Bayes
from sklearn.naive_bayes import GaussianNB
NBclassifier = GaussianNB()
NBmodel = NBclassifier.fit(x_training_data, y_training_data)


# In[45]:


NBpreds = NBmodel.predict(x_test_data)
NBpreds


# In[46]:


print(classification_report(y_test_data, predictions))


# In[47]:


print(classification_report(y_test_data, NBpreds))


# In[48]:


from sklearn.tree import DecisionTreeClassifier
dt = DecisionTreeClassifier()
dt_model=dt.fit(x_training_data, y_training_data)
dt_predict=dt_model.predict(x_test_data)
dt_predict


# In[49]:


print(classification_report(y_test_data, dt_predict))


# In[50]:


from sklearn.ensemble import RandomForestClassifier
rf = RandomForestClassifier(n_estimators=200)
rf_model=rf.fit(x_training_data, y_training_data)
rf_predict=rf_model.predict(x_test_data)
rf_predict


# In[51]:


print(classification_report(y_test_data, rf_predict))


# In[52]:


print(sub.dtypes)


# In[53]:


selected_columns = ['CO(GT)', 'PT08.S1(CO)', 'NMHC(GT)', 'PT08.S2(NMHC)', 'NOx(GT)', 'PT08.S3(NOx)', 'NO2(GT)', 'PT08.S4(NO2)', 'PT08.S5(O3)', 'RH']
df_selected = sub[selected_columns]

scal = StandardScaler()
df_scaled = scal.fit_transform(df_selected)

inertia = []

# Try a range of k values (e.g., from 1 to 10)
k_range = range(1, 11)

for k in k_range:
    kmeans = KMeans(n_clusters=k, random_state=0, n_init=10)
    kmeans.fit(df_scaled)
    inertia.append(kmeans.inertia_)

# Plot the Elbow Method graph
plt.figure(figsize=(8, 6))
plt.plot(k_range, inertia, marker='o', linestyle='--')
plt.xlabel('Number of Clusters (k)')
plt.ylabel('Inertia')
plt.title('Elbow Method')
plt.grid(True)
plt.show()


# In[54]:


# Выбор признаков для кластеризации (например, температура и влажность)
X = sub[['CO(GT)', 'NOx(GT)']]

# Определение количества кластеров (k)
k = 3  # Здесь можно указать желаемое количество кластеров

# Создание объекта KMeans с указанным количеством кластеров
kmeans = KMeans(n_clusters=k, random_state=0)

# Процесс кластеризации
kmeans.fit(X)

# Присвоение меток кластеров каждому объекту данных
labels = kmeans.labels_

# Визуализация результатов кластеризации
plt.scatter(X['CO(GT)'], X['NOx(GT)'], c=labels, cmap='viridis')
plt.scatter(kmeans.cluster_centers_[:, 0], kmeans.cluster_centers_[:, 1], s=300, c='red', label='Centroids')
plt.xlabel('CO(GT)')
plt.ylabel('NOx(GT)')
plt.legend()
plt.show()


# In[56]:


k =  6 # Choose the appropriate value for k
kmeans = KMeans(n_clusters=k, random_state=0)
sub['cluster'] = kmeans.fit_predict(df_scaled)


# In[58]:


sub



X = sub.drop(['CO(GT) Category'], axis=1)
y = sub['CO(GT) Category']

# Split the data into training and testing sets
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2)

# Standardize the data
scaler = StandardScaler()
X_train_scaled = scaler.fit_transform(X_train)
X_test_scaled = scaler.transform(X_test)

mlp = MLPClassifier(hidden_layer_sizes=(30,10,50),
                        max_iter = 300,activation = 'relu',
                        solver = 'adam')

mlp.fit(X_train_scaled, y_train)

y_pred = mlp.predict(X_test_scaled)

print('Accuracy: {:.2f}'.format(accuracy_score(y_test, y_pred)))


# Generate a classification report
class_report = classification_report(y_test, y_pred)
print("Classification Report:\n", class_report)


# Predict probabilities
y_prob = mlp.predict_proba(X_test_scaled)

# Compute ROC curve and ROC area for each class
fpr = dict()
tpr = dict()
roc_auc = dict()
for i in range(len(mlp.classes_)):
    fpr[i], tpr[i], _ = roc_curve(y_test, y_prob[:, i], pos_label=mlp.classes_[i])
    roc_auc[i] = auc(fpr[i], tpr[i])

# Plot ROC curve for each class
plt.figure(figsize=(8, 8))
for i in range(len(mlp.classes_)):
    plt.plot(fpr[i], tpr[i], lw=2, label=f'Class {mlp.classes_[i]} (AUC = {roc_auc[i]:.2f})')

plt.plot([0, 1], [0, 1], color='navy', lw=2, linestyle='--')
plt.xlabel('False Positive Rate')
plt.ylabel('True Positive Rate')
plt.title('Receiver Operating Characteristic (ROC) Curve')
plt.legend(loc='lower right')
plt.show()


# In[ ]:




