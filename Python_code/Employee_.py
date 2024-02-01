#!/usr/bin/env python
# coding: utf-8

# # Employee Data :
# 
# objective : This dataset can be used for various HR and workforce-related analyses, including employee retention, salary structure assessments, diversity and inclusion studies, and leave pattern analyses

# ### 1. Importing Library And Data_Set :

# In[1]:


import pandas as pd
import numpy as np
import seaborn as sns
import matplotlib.pyplot as plt
from sklearn.metrics import accuracy_score,recall_score,f1_score,precision_score


# In[2]:


df_new = pd.read_csv(r"C:\Users\DELL\Downloads\ML_Project DTSET\Employee.csv")
df_new.sample(5)


# In[3]:


df_new['LeaveOrNot']=np.where(df_new['LeaveOrNot']==0,"Leave","Not_Leave")
df_new.sample()


# ### 2. Data Preprocessing :

# In[4]:


df_new["Years in Company"] = 2023 - df_new['JoiningYear']
df_new.drop('JoiningYear',axis=1,inplace=True)
df_new.head()


# In[5]:


df_new.info()


# In[6]:


df_new.isna().sum()   ## No Null values


# In[7]:


df_new.head()


# In[8]:


# df_new['PaymentTier'] = str(df_new['PaymentTier'])


# In[9]:


plt.figure(figsize=(10,2))
sns.boxplot(df_new)


# ### Check Data Is normally Distributed or Not ?

# In[10]:


# for i in df_new.select_dtypes(include=np.number):
#     print(i)
#     plt.figure(figsize=(2,3))
#     sns.histplot(df_new[i])
#     plt.show()


# In[11]:


len(list(sorted(df_new['Age'].unique())))


# In[12]:


def Age(df_new):
    if (df_new['Age'] > 35) :
        return "C"
    elif (df_new['Age'] > 27) & (df_new['Age'] <= 35):
        return "B"
    else:
        return "A"

df_new['Age_Category'] = df_new.apply(lambda x: Age(x),axis=1)


# In[13]:


df_new.drop("Age",axis=1,inplace=True)
df_new.head()


# #### Pie Chart :

# In[14]:


import plotly.express as px

fig = px.pie(df_new,names="LeaveOrNot",color='LeaveOrNot',
             color_discrete_map={'Leave':'red',
                                 'Not_Leave':'green'})
fig.show()           ## Data is imbalanced.


# In[15]:


list(df_new.select_dtypes(include=object))


# In[16]:


# fig = px.sunburst(df_new,path=['LeaveOrNot','Education', 'City', 'Gender', 'EverBenched','Age_Category'])
# # fig = px.sunburst(df_new,path=['LeaveOrNot','Education', 'City', 'Gender', 'EverBenched'])
# fig.show()


# #### Bar_Chart:

# In[ ]:





# In[78]:


for i in df_new:
    plt.figure(figsize=(5,3))
    sns.countplot(x= df_new[i], hue = 'LeaveOrNot',data=df_new)
    plt.title("Bar_Chart")


# #### Splitting Of Data:

# In[18]:


num_df = df_new.select_dtypes(include=np.number)
num_df.head()


# In[19]:


cat_df = df_new.select_dtypes(include= object)
cat_df.head()


# #### Use LabelEncoder:

# In[20]:


from sklearn.preprocessing import LabelEncoder
cat_df = cat_df.apply(LabelEncoder().fit_transform)
cat_df.head()


# In[21]:


df_new = pd.concat([num_df,cat_df],axis=1)
df_new.sample(5)


# In[22]:


x = df_new.drop("LeaveOrNot",axis=1)
y = df_new["LeaveOrNot"]


# In[23]:


from imblearn.combine import SMOTETomek

smt = SMOTETomek(random_state=555)
x_rev,y_rev =smt.fit_resample(x,y)


# In[24]:


df_new = pd.concat([x_rev,y_rev],axis=1)
df_new.sample(5)


# In[25]:


import plotly.express as px

fig = px.pie(df_new,names="LeaveOrNot",color='LeaveOrNot',
             color_discrete_map={0:'red',
                                 1:'green'})
fig.show()   ## So, Data is proper Balance.


# #### Data_Partition :

# In[26]:


from sklearn.model_selection import train_test_split

X_train,X_test,y_train,y_test = train_test_split(x_rev,y_rev,random_state=555,test_size=0.3)


# # Model:

# ### 1. Logistic Regression:

# In[27]:


from sklearn.feature_selection import SequentialFeatureSelector as sfs
from sklearn.linear_model import LogisticRegression

logreg = LogisticRegression()
model1 = sfs(logreg,n_features_to_select=4,direction='backward',scoring='accuracy',cv=5)
model1.fit(X_train,y_train)


# In[28]:


model1.feature_names_in_


# In[29]:


model1.get_feature_names_out()


# In[30]:


x_train = X_train.loc[:,['PaymentTier','Years in Company','Education','Gender','EverBenched']]
x_test = X_test.loc[:,['PaymentTier','Years in Company','Education','Gender','EverBenched']]
x_train.head()


# In[31]:


x_train.head(1)


# In[32]:


output = logreg.fit(x_train,y_train)


# In[33]:


output.intercept_


# In[34]:


output.coef_


# In[35]:


output.feature_names_in_


# ### Model Equation: 
# Y = 3.7898277 + ( -0.57633914 X PaymentTier - 0.23661371 X Years in Company  +  0.13515309 X Education -  0.87516551 X Gender - 0.23760026 X EverBenched )

# In[36]:


train = pd.concat([x_train,y_train],axis=1)
train.head()


# In[37]:


train["prob"] = output.predict_proba(x_train)[:,1]


# In[38]:


train['predict'] = np.where(train['prob']>=0.5,1,0)
train.head()


# In[39]:


from sklearn.metrics import classification_report,confusion_matrix
print('Classification Report of Train')
print(classification_report(train['LeaveOrNot'],train['predict']))


# In[40]:


test = pd.concat([x_test,y_test],axis=1)
test["prob"] = output.predict_proba(x_test)[:,1]
test['predict'] = np.where(test['prob']>=0.5,1,0)


# In[41]:


from sklearn.metrics import classification_report,confusion_matrix
print('Classification Report of Train')
print(classification_report(test['LeaveOrNot'],test['predict']))


# In[42]:


accurancy_LR = accuracy_score(test['LeaveOrNot'],test['predict'])
recall_LR = recall_score(test['LeaveOrNot'],test['predict'])
precision_LR = precision_score(test['LeaveOrNot'],test['predict'])
f1_LR = f1_score(test['LeaveOrNot'],test['predict'])
metrix_LR = confusion_matrix(test['LeaveOrNot'],test['predict'])


# In[43]:


from sklearn.metrics import confusion_matrix
print(confusion_matrix(test['LeaveOrNot'],test['predict']))


# ### 2. Decisiom Tree

# In[44]:


from sklearn.tree import DecisionTreeClassifier

dt = DecisionTreeClassifier(criterion="gini")
dt.fit(X_train,y_train)


# In[45]:


train = pd.concat([X_train,y_train],axis=1)
test = pd.concat([X_test,y_test],axis=1)
train.head()


# In[46]:


independent_variable = list(train.columns[:8])
independent_variable


# In[47]:


from sklearn import tree
import matplotlib.pyplot as plt

subscribe = ['Leave','Not_Leave']
fig, axes =plt.subplots(nrows = 1,ncols=1,figsize = (5,4), dpi=300)
tree.plot_tree(dt,
               feature_names=independent_variable,
               class_names=subscribe,
               filled=True,
               node_ids=True,
               fontsize=2);  


# In[48]:


train['predict'] = dt.predict(X_train)
test['predict'] = dt.predict(X_test)
train.head()


# In[49]:


from sklearn.metrics import classification_report
print(classification_report(train['LeaveOrNot'],train['predict']))


# #### Grid Search Method:

# In[50]:


params = {'min_samples_split':[110,100,90,80,70],
          'min_samples_leaf':[10,20,30,48,55,67],
          'max_depth':[3,4]}


# In[51]:


from sklearn.model_selection import GridSearchCV

gsv = GridSearchCV(DecisionTreeClassifier(random_state=123),
                   params,
                  verbose=1,
                  cv =10)


# In[52]:


gsv.fit(X_train,y_train)


# In[53]:


gsv.best_estimator_


# ### Prunning Method :

# In[54]:


from sklearn import tree

model2 = tree.DecisionTreeClassifier(criterion='gini',
                                min_samples_split=110,
                                min_samples_leaf=10,
                                max_depth =4)

model2.fit(X_train,y_train)


# In[55]:


X_train.head()


# In[56]:


from sklearn import tree
import matplotlib.pyplot as plt

subscribe = ['Leave','Not_Leave']
fig, axes =plt.subplots(nrows = 1,ncols=1,figsize = (5,4), dpi=300)
tree.plot_tree(model2,
               feature_names=independent_variable,
               class_names=subscribe,
               filled=True,
               node_ids=True,
               fontsize=2); 


# In[57]:


train['predict'] = model2.predict(X_train)
test['predict'] = model2.predict(X_test)
train.head()


# In[58]:


from sklearn.metrics import classification_report
print(classification_report(train['LeaveOrNot'],train['predict']))


# In[59]:


from sklearn.metrics import classification_report
print(classification_report(test['LeaveOrNot'],test['predict']))


# In[60]:


accurancy_DM= accuracy_score(test['LeaveOrNot'],test['predict'])
recall_DM = recall_score(test['LeaveOrNot'],test['predict'])
precision_DM = precision_score(test['LeaveOrNot'],test['predict'])
f1_DM = f1_score(test['LeaveOrNot'],test['predict'])
metrix_DM = confusion_matrix(test['LeaveOrNot'],test['predict'])


# In[61]:


from sklearn.metrics import confusion_matrix
print(confusion_matrix(test['LeaveOrNot'],test['predict']))


# ### 3. Random Forest

# In[62]:


from sklearn.ensemble import RandomForestClassifier

model3 = RandomForestClassifier(n_estimators=25,
                               criterion='gini',
                               max_depth=5,
                               min_samples_split=110,
                               min_samples_leaf=30,
                               max_features='sqrt',random_state=124)
model3.fit(X_train,y_train)


# In[63]:


### Que.3 (imp. feature)

imp = pd.Series(data=model3.feature_importances_,index=model3.feature_names_in_).sort_values(ascending=False)
plt.figure(figsize=(10,15))
plt.title("Feature Importance / Selection")
ax = sns.barplot(y=imp.index, x=imp.values, palette='BrBG', orient='h')


# In[64]:


from sklearn.tree import export_graphviz
import pydot


# In[65]:


list(x.columns)   


# In[66]:


feature_list = list(x.columns)
sub = ['Leave','Not_Leave']

tree = model3.estimators_[11]

export_graphviz(tree, out_file='abc.dot',
               feature_names=feature_list,
               class_names=sub,
               rounded=True,
               filled=True)

(graph,) = pydot.graph_from_dot_file('abc.dot')
graph.write_png('tree.png')

from IPython.display import Image
Image(filename='tree.png')


# In[67]:


train = pd.concat([X_train,y_train],axis=1)
train['predict'] =model3.predict(X_train)
train.head()


# In[68]:


print(classification_report(train['LeaveOrNot'],train['predict']))


# In[69]:


test = pd.concat([X_test,y_test],axis=1)
test['predict'] = model3.predict(X_test)
test.head()


# In[70]:


print(classification_report(test['LeaveOrNot'],test['predict']))


# In[71]:


accurancy_RF = accuracy_score(test['LeaveOrNot'],test['predict'])
recall_RF = recall_score(test['LeaveOrNot'],test['predict'])
precision_RF = precision_score(test['LeaveOrNot'],test['predict'])
f1_RF = f1_score(test['LeaveOrNot'],test['predict'])
metrix_RF = confusion_matrix(test['LeaveOrNot'],test['predict'])


# In[72]:


from sklearn.metrics import confusion_matrix
print(confusion_matrix(test['LeaveOrNot'],test['predict']))


# In[73]:


model = ["Logistic Regration",'Decision Tree','Random Forest']
accurancy =[accurancy_LR,accurancy_DM,accurancy_RF]
precision = [precision_LR,precision_DM,precision_RF]
recall = [recall_LR,recall_DM,recall_RF]
f1 = [f1_LR,f1_DM,f1_RF]


# ## Overall Performance :

# In[74]:


plt.figure(figsize=(10,10))
plt.subplot(2,2,1)
plots = sns.barplot(x = model,y=accurancy,linewidth = 1.5)
for bar in plots.patches:
    plots.annotate(format(bar.get_height(), '.2f'),
                   (bar.get_x() + bar.get_width() / 2,
                    bar.get_height()), ha='center', va='center',
                   size=10, xytext=(0, 5),
                   textcoords='offset points')
plt.xlabel("Accurancy")

plt.subplot(2,2,2)
plots=sns.barplot(x = model,y=precision,linewidth = 1.5)
for bar in plots.patches:
    plots.annotate(format(bar.get_height(), '.2f'),
                   (bar.get_x() + bar.get_width() / 2,
                    bar.get_height()), ha='center', va='center',
                   size=10, xytext=(0, 5),
                   textcoords='offset points')
plt.xlabel("precision")

plt.subplot(2,2,3)
plots=sns.barplot(x = model,y=recall,linewidth = 1.5)
for bar in plots.patches:
    plots.annotate(format(bar.get_height(), '.2f'),
                   (bar.get_x() + bar.get_width() / 2,
                    bar.get_height()), ha='center', va='center',
                   size=10, xytext=(0, 5),
                   textcoords='offset points')
plt.xlabel("recall")

plt.subplot(2,2,4)
plots=sns.barplot(x = model,y=f1,linewidth = 1.5)
for bar in plots.patches:
    plots.annotate(format(bar.get_height(), '.2f'),
                   (bar.get_x() + bar.get_width() / 2,
                    bar.get_height()), ha='center', va='center',
                   size=10, xytext=(0, 5),
                   textcoords='offset points')
plt.xlabel("f1")


# In[75]:


import os
os.chdir(r"C:\Users\DELL\OneDrive\Desktop")


# In[76]:


import pickle

pickle.dump(model3, open(r"C:\Users\DELL\OneDrive\Desktop\build.pkl",'wb'))  #model = dt # Exporting model from python to laptop


# In[ ]:




