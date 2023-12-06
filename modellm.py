#!/usr/bin/env python
# coding: utf-8

# In[1]:


import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.preprocessing import StandardScaler
from sklearn.model_selection import train_test_split
from sklearn.linear_model import LinearRegression



import warnings
warnings.simplefilter("ignore")


# In[2]:


data=pd.read_csv(r"C:\Users\Dell123\Desktop\data science project\temperature_data.csv")


# In[3]:


data.head()


# In[4]:


data.shape


# In[5]:


data.info()


# In[6]:


data.describe()


# ### Understanding the Features
# 
#      ambient--->Ambient Temp(The basic ambient temperature rating point of nearly all electric motors is 40° C.) Can Say Room Temp
#      coolant---->Antifreeze ensures that your engine temperature remains stable to perform well in all climates.The coolant temperature for the battery must always be kept between 15 °C and 30 °C.
#      u_d---->Direct Axis --Flux linkage component of current is aligned along the d axis(Voltage Component)
#      u_q----?Quardartic axis --- torque component of current is aligned along the q axis((Voltage Component)
#      Motor_Speed-->Speed of the motor
#      torque  ---->the measure of the force that can cause an object to rotate about an axis. in other words amount of force
#      i_d,i_q ----> Current quardintes
#      pm---->Permanent Magnet surface temperature representing the rotor temperature. This was measured with an infrared thermography unit.
#      stator_yoke--->The outer frame of a dc machine -- temperature is measured with a thermal sensor.
#      Stator tooth --->temperature is measured with a thermal sensor.
#      Stator winding ---->temperature measured with a thermal sensor.
#      profile_id--->Each measurement session has a unique ID. Make sure not to try to estimate from one session onto the other as they are strongly independent.
# 
#      

# ### Exploratory Data Analysis

# In[7]:


data.isnull().sum()


# In[8]:


data.duplicated().sum()


# In[9]:


# Histogram

# Create subplots
fig, axes = plt.subplots(nrows=7, ncols=2, figsize=(12, 24))
axes = axes.flatten()

# Plot KDE for each feature
for i, col in enumerate(data.columns):
    sns.histplot(data[col], ax=axes[i], color='green',kde=True)
    axes[i].set_title(col)
    axes[i].set_ylabel('Density')
    
# Adjust the layout
plt.tight_layout()

# Display the plot
plt.show()


# I observe that zero value appears most frequently in columns representing coordinates for current and voltage
# 
# Coolant and ambient are measured in the same units. The highest peak of coolant is at the value 20 and the highest peak of of ambient is at the value 26. Values 20, 23 and 25 of ambient and 50 of coolant also have peaks

# In[10]:


# Plot the KDE plot

# Create subplots
fig, axes = plt.subplots(nrows=7, ncols=2, figsize=(12, 24))
axes = axes.flatten()

# Plot KDE for each feature
for i, col in enumerate(data.columns):
    sns.kdeplot(data[col], ax=axes[i], color='purple')
    axes[i].set_title(col)
    axes[i].set_ylabel('Density')
    

# Adjust the layout
plt.tight_layout()

# Display the plot
plt.show()

sns.pairplot(data)
# In[11]:



# Create subplots
fig, axes = plt.subplots(nrows=7, ncols=2, figsize=(12, 24))
axes = axes.flatten()

# Plot KDE for each feature
for i, col in enumerate(data.columns):
    sns.boxplot(data[col], ax=axes[i], color='red')
    axes[i].set_title(col)
    axes[i].set_ylabel('Density')
    

# Adjust the layout
plt.tight_layout()

# Display the plot
plt.show()


# we can see pm,i_q,torque,u_d,ambient has oulier which will deal later

# In[12]:


plt.figure(figsize=(10,7))
sns.heatmap(data.corr(),annot=True,cmap='viridis')


#      
# ### Observations:
# 
#      stator_tooth, stator_winding and stator_yoke are correlated between themselfs
#      
#      torque has the maximum correlation coeff  1 with i_q, and strong negative correlation with u_d
#      
#      motor_speed has strong positive correlation with u_q and strong negative correlation with i_d
#      
#      stator_yoke is significantly correlated with coolant and less significantly with ambient
#      
#      stator_tooth and stator_winding have positive correlation with coolant and negative correlation with i_d
#      
#      i_q and u_d have strong negative correlation

# In[13]:


# Visualizing the highly coreleated features

fig, axes = plt.subplots(nrows=1, ncols=2)

# Plot the first scatterplot
axes[0].scatter(data['torque'],data['i_q'],c='Yellow')
axes[0].set_xlabel("Torque")
axes[0].set_ylabel("Current Component")

# Plot the second scatterplot
axes[1].scatter(data['motor_speed'], data['u_q'],c='Cyan')
axes[1].set_xlabel("Motor Speed")
axes[1].set_ylabel("Voltage Component")

# Adjust the layout of subplots
plt.tight_layout()

# Display the plot
plt.show()


# In[14]:


def observe_3d_relationships(variable_x, variable_y, variable_z, ax):

    ax.set_xlabel('variable_x')
    ax.set_ylabel('variable_y')
    ax.set_zlabel('variable_z')

    x = data[variable_x]
    y = data[variable_y]
    z = data[variable_z]

    ax.scatter(x, y, z)


# In[15]:


fig = plt.figure(figsize=(20,20))
fig.subplots_adjust(hspace=0.2, wspace=0.2)

# =============
# First subplot
# =============
ax = fig.add_subplot(3, 2, 1, projection='3d')
observe_3d_relationships('u_d', 'torque', 'i_q', ax)

# =============
# Second subplot
# =============
ax = fig.add_subplot(3, 2, 2, projection='3d')
observe_3d_relationships('i_d', 'motor_speed', 'u_q', ax)

# =============
# Third subplot
# =============
ax = fig.add_subplot(3, 2, 3, projection='3d')
observe_3d_relationships('i_d', 'stator_tooth', 'coolant', ax)

# =============
# Forth subplot
# =============
ax = fig.add_subplot(3, 2, 4, projection='3d')
observe_3d_relationships('i_d', 'stator_winding', 'coolant', ax)

# =============
# Fifth subplot
# =============
ax = fig.add_subplot(3, 2, 5, projection='3d')
observe_3d_relationships('ambient', 'stator_yoke', 'coolant', ax)

# =============
# Sixth subplot
# =============
ax = fig.add_subplot(3, 2, 6, projection='3d')
observe_3d_relationships('stator_tooth', 'stator_winding', 'stator_yoke', ax)

plt.show()


# In[16]:


plt.figure(figsize=(10,8))
data['profile_id'].value_counts().sort_values().plot(kind='bar')


# profilie_id 66,65,6,20 have most number of measurement recorded
for i in data.columns:
    sns.distplot(data[i],color='r')
    sns.boxplot(data[i],color='g')
    plt.vlines(data[i].mean(),ymin = -1,ymax = 1,color = 'b')
    plt.show()

# As we can see from the the above plots, the mean and median for most of the plots are very close to each other. So the data seems to have low skewness for almost all variables.

# ## Checking Skewness and kurtosis numerically

# In[17]:


import scipy.stats as stats

for i in data.columns:
    print(i,':\n Skewness:',data[i].skew(),':\n Kurtosis:',data[i].kurt(),'\n')


# In[18]:


#!pip install autoviz


# In[19]:


#!pip install xlrd # Autoviz class - Dependency 

%matplotlib inline
#Generating AutoViz Report #this is the default command when using a file for the dataset
from autoviz.AutoViz_Class import AutoViz_Class
AV = AutoViz_Class()
filename= ""
sep = ","
dft = AV.AutoViz(
    filename,
    sep=",",
    depVar="motor_speed",
    dfte=data,
    header=0,
    verbose=2,
    lowess=False,
    chart_format="svg",
    max_rows_analyzed=1000000,
    max_cols_analyzed=13,
    
)
# In[ ]:




%matplotlib inline
from autoviz.AutoViz_Class import AutoViz_Class
import os

# Create an instance of AutoViz_Class
AV = AutoViz_Class()

# Generate visualizations without saving the plots to the specified directory
dft = AV.AutoViz(
    '',
    sep=",",
    depVar="motor_speed",
    dfte=data,
    header=0,
    verbose=0,
    lowess=False,
    chart_format="png",
    max_rows_analyzed=1000000,
    max_cols_analyzed=13
)



# In[20]:


# OUTLIER TREATMENT
df = pd.read_csv(r"C:\Users\Dell123\Desktop\data science project\temperature_data.csv")
df1=df.copy()

# Function to detect outliers using Z-score
def detect_outliers_zscore(data):
    threshold = 3
    z_scores = np.abs((data - data.mean()) / data.std())
    return z_scores > threshold

# Loop through each column (except 'profile_id') to detect outliers and impute with mean
for col in df1.columns:
    if col != 'profile_id':
        outliers = detect_outliers_zscore(df1[col])
        if outliers.any():
            df1[col] = np.where(outliers, df1[col].mean(), df1[col])

# Save the modified DataFrame to a new CSV file (if needed)
# Replace 'imputed_dataset.csv' with the desired filename
df.to_csv('imputed_dataset.csv', index=False)


# In[21]:


# Calculate the difference between the original DataFrame and the imputed DataFrame
difference_df = df - df1

# Check if the difference is non-zero to identify columns where outliers were imputed
outliers_imputed_mask = difference_df.abs() > 0

# Calculate the percentage of outliers imputed for each column
percentage_outliers_imputed = (outliers_imputed_mask.sum() / len(df)) * 100

# Create a DataFrame to display the results
results_df = pd.DataFrame({
    'Column Name': percentage_outliers_imputed.index,
    'Percentage of Outliers Imputed': percentage_outliers_imputed.values
})

print(results_df)


# Based on the results of the percentage of outliers imputed for each column,had a very low percentage of outliers imputed. The majority of the columns had a percentage of 0%, indicating that no outliers were imputed for those features.
# 
# The columns 'ambient', 'torque', 'i_d', and 'i_q' had small percentages of outliers imputed, but still relatively low, which suggests that only a small portion of their data was modified due to the imputation process.
# 
# columns 'coolant', 'u_d', 'u_q', 'motor_speed', 'pm', 'stator_yoke', 'stator_tooth', 'stator_winding', and 'profile_id' had no outliers imputed, meaning that their data remained unchanged during the imputation process.
# 
# Overall, it seems that the outlier imputation process was conservative and had a minimal impact on the dataset, which is a positive outcome. However, it's essential to interpret these results in the context of your specific analysis and understand the implications of the imputation on your data and any subsequent analyses.

# In[22]:



# List of features for visualization
features = df.columns.tolist()

# Calculate the number of features
num_features = len(features)

# Create sub=bplots for all features before and after imputation
fig, axes = plt.subplots(num_features, 2, figsize=(16, 4*num_features))

for i, feature in enumerate(features):
    # Plot original data (before imputation)
    sns.kdeplot(df[feature], label=feature, ax=axes[i, 0])
    axes[i, 0].set_title(f'Before Imputation - {feature}')
    axes[i, 0].legend()

    # Plot imputed data (after imputation)
    sns.kdeplot(df1[feature], label=feature, ax=axes[i, 1],color='orange')
    axes[i, 1].set_title(f'After Imputation - {feature}')
    axes[i, 1].legend()

plt.tight_layout()
plt.show()


# In[23]:


df1.info()


# In[24]:


x=df1.drop(['motor_speed','stator_winding','profile_id'],axis=1)
y=df1['motor_speed']


# In[25]:


from sklearn.preprocessing import StandardScaler
from sklearn.model_selection import train_test_split
from sklearn.linear_model import LinearRegression
from sklearn.metrics import mean_squared_error,mean_absolute_error,r2_score
get_ipython().system('pip install xgboost')
from xgboost import XGBRegressor

from sklearn.tree import DecisionTreeRegressor
from sklearn.ensemble import RandomForestRegressor


# In[26]:


sd=StandardScaler()
x=sd.fit_transform(x)


# In[27]:


x_train,x_test,y_train,y_test=train_test_split(x,y,test_size=0.3,random_state=42)


# In[28]:


lm=LinearRegression()
lm.fit(x_train,y_train)
lm_pred=lm.predict(x_test)


# In[29]:


print("MSE:",mean_squared_error(y_test,lm_pred))
print("MAE:",mean_absolute_error(y_test,lm_pred))
print("R2_Score:",r2_score(y_test,lm_pred))


# In[30]:


XG=XGBRegressor()
XG.fit(x_train,y_train)
XG_pred=XG.predict(x_test)

print("MSE:",mean_squared_error(y_test,XG_pred))
print("MAE:",mean_absolute_error(y_test,XG_pred))
print("R2_Score:",r2_score(y_test,XG_pred))


# In[31]:


dt=DecisionTreeRegressor()
dt.fit(x_train,y_train)
dt_pred=dt.predict(x_test)

print("MSE:",mean_squared_error(y_test,dt_pred))
print("MAE:",mean_absolute_error(y_test,dt_pred))
print("R2_Score:",r2_score(y_test,dt_pred))


# In[32]:


#import pickle
#pickle_out=open("XG.pkl","wb")
#pickle.dump(XG,pickle_out)
#pickle_out.close()


# In[33]:


#model=['Linear','DecisiionTree','RandomForest','XGBOOST']
#accuracy=[r2_score(y_test,lm_pred),r2_score(y_test,dt_pred),r2_score(y_test,rf_pred),r2_score(y_test,XG_pred)]
#acc=pd.DataFrame({'MLModel':model,
                 #'Accuracy':accuracy})


# In[34]:


#sorted_df=acc.sort_values(by='Accuracy',ascending=False)
#print(sorted_df)


# In[35]:


import joblib
joblib.dump(lm,r'c:\users\Dell123\lm.pkl')


# In[ ]:





# In[ ]:




