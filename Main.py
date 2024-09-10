#!/usr/bin/env python
# coding: utf-8

# In[1]:


import pandas as pd
import numpy as np
import matplotlib.pyplot as plt


# In[2]:


data = pd.read_excel("C:\\Users\\nlatifi\\Downloads\\archive (3)\\E Commerce Dataset.xlsx")


# In[3]:


data.head()


# In[4]:


data.info()


# In[5]:


data1 = data.drop(columns = 'CustomerID')


# In[6]:


data1['CityTier'] = data1.CityTier.astype('object')
data1.dtypes


# In[7]:


data1['SatisfactionScore'] = data1.SatisfactionScore.astype('object')
data1.dtypes


# In[8]:


data1.columns


# In[9]:


data1.describe()


# In[10]:


data1.isna().sum()


# In[11]:


m_1 = data1['DaySinceLastOrder'].mean()
m_2 = data1['OrderAmountHikeFromlastYear'].mean()
m_3 = data1['Tenure'].mean()
m_4 = data1['OrderCount'].mean()
m_5 = data1['CouponUsed'].mean()
m_6 = data1['WarehouseToHome'].mean()
m_7 = data1['HourSpendOnApp'].mean()

data1['DaySinceLastOrder'].fillna(m_1, inplace=True)
data1['OrderAmountHikeFromlastYear'].fillna(m_2, inplace=True)
data1['Tenure'].fillna(m_3, inplace=True)
data1['OrderCount'].fillna(m_4, inplace=True)
data1['CouponUsed'].fillna(m_5, inplace=True)
data1['WarehouseToHome'].fillna(m_6, inplace=True)
data1['HourSpendOnApp'].fillna(m_7, inplace=True)


# In[12]:


data1.isna().sum()


# In[13]:


x=data1[[ 'Tenure', 'PreferredLoginDevice', 'CityTier',
       'WarehouseToHome', 'PreferredPaymentMode', 'Gender', 'HourSpendOnApp',
       'NumberOfDeviceRegistered', 'PreferedOrderCat', 'SatisfactionScore',
       'MaritalStatus', 'NumberOfAddress', 'Complain',
       'OrderAmountHikeFromlastYear', 'CouponUsed', 'OrderCount',
       'DaySinceLastOrder', 'CashbackAmount']]


# In[14]:


x


# In[15]:


y = data1[['Churn']]
y


# In[16]:


x_dum = pd.get_dummies(x)


# In[17]:


x_dum


# In[18]:


x_dum.columns


# In[19]:


x_dum.drop(columns=['PreferredLoginDevice_Phone','PreferredPaymentMode_UPI', 'Gender_Male','PreferedOrderCat_Others','MaritalStatus_Single','CityTier_3','SatisfactionScore_5'], axis = 1, inplace=True)


# In[20]:


x_dum.columns


# ### Assumptions 

# In[21]:


import seaborn as sns
sns.pairplot(data,x_vars=['OrderAmountHikeFromlastYear', 'OrderCount',
       'DaySinceLastOrder', 'CashbackAmount']
            ,y_vars=['Churn'],kind='scatter')


# In[22]:


#multi-collinearity test
corr_X1 = x_dum.corr().round(2)


# In[23]:


corr_X1.head()


# In[24]:


plt.figure(figsize=(20, 15))
sns.heatmap(corr_X1, annot=True)
plt.show()


# In[25]:


y.value_counts()


# ## Split data
# 

# In[26]:


from sklearn.model_selection import train_test_split


# In[27]:


x_train,x_test,y_train,y_test=train_test_split(x_dum,y,test_size=0.2,random_state=0)


# In[28]:


x_train.head()


# In[29]:


y_train


# In[30]:


y_train.Churn.value_counts()


# In[31]:


from sklearn.preprocessing import StandardScaler


# In[32]:


stdscaler=StandardScaler()


# In[33]:


stdscaler.fit(x_train)


# In[34]:


x_train_scaled=stdscaler.transform(x_train)


# In[35]:


pd.DataFrame(x_train_scaled).head()


# In[36]:


x_test_scaled=stdscaler.transform(x_test)


# In[37]:


y_train.Churn.value_counts()


# ## Sampling

# In[38]:


from imblearn.over_sampling import SMOTE


# In[39]:


x_resample,y_resample=SMOTE().fit_resample(x_train_scaled,y_train)


# In[40]:


y_resample.Churn.value_counts()


# ## Logistics Regression

# In[41]:


from sklearn.linear_model import LogisticRegression


# In[42]:


logClassifier=LogisticRegression()


# In[43]:


#Training
logClassifier.fit(x_resample,y_resample)


# In[44]:


logClassifier.score(x_resample,y_resample)


# In[45]:


#Predict
y_pred=logClassifier.predict(x_test_scaled)


# In[46]:


y_pred


# In[47]:


y_pred_prob = logClassifier.predict_proba(x_test_scaled)


# In[48]:


y_pred_prob


# In[49]:


y_pred_prob=pd.DataFrame(y_pred_prob)


# In[50]:


#Evalution
from sklearn.metrics import confusion_matrix


# In[51]:


cm=confusion_matrix(y_test,y_pred)


# In[52]:


print(cm)


# In[53]:


from sklearn.metrics import classification_report


# In[54]:


print(classification_report(y_test,y_pred))


# In[55]:


from sklearn.metrics import roc_curve

# roc curve for models
fpr1, tpr1, thresh1 = roc_curve(y_test, y_pred_prob.iloc[:,1], pos_label=1)


# In[56]:


random_probs = [0 for i in range(len(y_test))]
p_fpr, p_tpr, _ = roc_curve(y_test, random_probs, pos_label=1)


# In[57]:


from sklearn.metrics import roc_auc_score

# auc scores
auc_score1 = roc_auc_score(y_test, y_pred_prob.iloc[:,1])
print(auc_score1)


# In[58]:


plt.style.use('seaborn')

# plot roc curves
plt.plot(fpr1, tpr1, linestyle='--',color='orange', label='Logistic Regression')
plt.plot(p_fpr, p_tpr, linestyle='--', color='blue')
# title
plt.title('ROC curve')
# x label
plt.xlabel('False Positive Rate')
# y label
plt.ylabel('True Positive rate')
plt.legend(loc='best')
plt.savefig('ROC',dpi=300)
plt.show();


# In[59]:


residuals = y_train.values-y_pred
mean_residuals = np.mean(residuals)
print("Mean of Residuals {}".format(mean_residuals))


# ### KNN

# In[60]:


from sklearn.neighbors import KNeighborsClassifier
classifier = KNeighborsClassifier(n_neighbors = 2, metric = 'minkowski', p = 2)
classifier.fit(x_resample, y_resample)


# In[61]:


y_pred_knn= classifier.predict(x_test_scaled)


# In[62]:


pd.DataFrame(y_pred)


# In[63]:


classifier.score(x_resample, y_resample)


# In[64]:


#Evalution
from sklearn.metrics import confusion_matrix


# In[65]:


cm=confusion_matrix(y_test,y_pred_knn)


# In[66]:


cm


# In[67]:


from sklearn.metrics import classification_report


# In[68]:


print(classification_report(y_test,y_pred))


# In[69]:


from sklearn.datasets import make_blobs


# In[70]:


X, y = make_blobs(n_samples=300, centers=4, cluster_std=0.60, random_state=0)
k_values = list(range(1, 11))
accuracies = []


# In[71]:


for k in k_values:
    knn = KNeighborsClassifier(n_neighbors=k)
    knn.fit(x_resample, y_resample)
    accuracy = knn.score(x_test_scaled, y_test)
    accuracies.append(accuracy)


# In[72]:


plt.plot(k_values, accuracies, marker='o', linestyle='-')
plt.title('Elbow Method for Optimal K')
plt.xlabel('Number of Neighbors (K)')
plt.ylabel('Accuracy')
plt.xticks(k_values)
plt.grid(True)
plt.show()


# ## Decision Tree

# In[73]:


from sklearn.tree import DecisionTreeClassifier


# In[74]:


classifier = DecisionTreeClassifier(criterion='gini')


# In[75]:


classifier.fit(x_resample, y_resample)


# In[76]:


y_pred3 = classifier.predict(x_test_scaled)


# In[77]:


y_pred3


# In[78]:


print('Accuracy on Training Data: ')
classifier.score(x_resample, y_resample)


# In[79]:


print('Accuracy on Test Data: ')
classifier.score(x_test_scaled, y_test)


# In[80]:


print('Confusion Matrix: ')
cm3 = confusion_matrix(y_test, y_pred)
cm3


# In[81]:


print("Classification Report: ")
print(classification_report(y_test, y_pred3))


# ### Random Forest 

# In[82]:


from sklearn.ensemble import RandomForestClassifier


# In[83]:


rclassifier = RandomForestClassifier(n_estimators=10, criterion ='entropy')


# In[84]:


rclassifier.fit(x_resample, y_resample)


# In[85]:


y_predi = rclassifier.predict(x_test_scaled)


# In[86]:


print('Accuracy on Training data : ')
rclassifier.score(x_resample, y_resample)


# In[87]:


rclassifier.feature_importances_


# In[88]:


data.columns


# In[89]:


print('Accuracy on Test data: ')
rclassifier.score(x_test_scaled, y_test)


# In[90]:


print('Confusion Matrix: ')
cm3 = confusion_matrix(y_test, y_predi)
cm3


# In[102]:


from sklearn.metrics import precision_score,accuracy_score
precision_score_list=[precision_score(y_test,y_pred),precision_score(y_test,y_pred_knn),precision_score(y_test,y_pred3),precision_score(y_test,y_predi)]
model_name_list=['LogisticRegression','KNN','DecisionTree(Gini Index)','RandomForest']
 
sns.barplot(x=model_name_list,y=precision_score_list)
plt.xlabel('Model')
plt.ylabel('Precision Score')
plt.title('Precision Scores of Different Models')
plt.ylim(0,1)
plt.xticks(rotation=45)
plt.show()


# In[ ]:




