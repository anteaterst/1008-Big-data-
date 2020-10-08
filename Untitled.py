#!/usr/bin/env python
# coding: utf-8

# In[1]:


import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns


# In[2]:


#디렉토리 확인할것
df_train = pd.read_csv("data/titanic_train.csv")
df_test = pd.read_csv("data/titanic_test.csv")


# In[3]:


#확인용
df_train.head(5)


# In[4]:


df_test.info()


# In[5]:


df_train = df_train.drop(['name','ticket','body','cabin','home.dest'],axis = 1)


# In[6]:


df_train.head()


# In[7]:


df_test = df_test.drop(['name','ticket','body','cabin','home.dest'],axis = 1)


# In[8]:


df_test.head()


# In[9]:


print(df_train['survived'].value_counts().plot.bar())


# In[10]:


#survived 피처를 기준으로 그룹을 나누어 그룹별 pclass 피처의 분포를 살펴보자
#pclass 승객 등급

print(df_train['pclass'].value_counts())


# In[11]:


ax = sns.countplot(x = 'pclass', hue = 'survived', data = df_train)


# In[27]:


# 함수 실행 age, sibsp

valid_features(df_train[df_train['age']>0],'age',distribution_check=True)


# In[20]:


valid_features(df_train,'sibsp',distribution_check=False)


# In[25]:


# parch : 동승한 부모 또는 자녀수
# sex : 탑승자의 성별
# sibsp 동승한 형제 또는 배우자 수
valid_features(df_train,'sex',distribution_check = True)


# In[22]:


valid_features(df_train,'pclass',distribution_check=True)


# In[23]:


valid_features(df_train,'parch',distribution_check=True)


# In[17]:


#pclass(고객 등급) : 생존에 영향을 미친다.
#age(나이) : ?
#sibsp, parch(동승자) : ?
#sex(성) : 생존에 영향을 미친다.


# In[30]:


#로지스틱 회귀 모델
#사건의 발생 가능성을 예측하는 데 사용되는 통계기법
#기존의 회귀 분석의 예측값 Y를 0~1사이의 값으로 제한
#0.5보다 크면 ? 1이다(중요*****)
#0.5보다 작으면 ? 0이다(중요*****) 로 분류하는 방법이다.
# 계수 분석을 통한 피처의 영항력 해석이 용이하다는 장점!


#전처리 1) 결측이 존재하는 데이터를 삭제
#       => 처리가 쉽고 주관성이 개입 될 여지가 없다.
#       2) 평균값, 또는 중앙값, 또는 최빈값 등의 임의의 수치로 결측치를 채워넣기
#       =>데이터를 모두 분석에 활용할 수 있다. 수치 왜곡의 가능성이 있다.

#2번 방법을 사용하여 전처리
# age의 결측값을 평균값으로 대체하자
replace_mean = df_train[df_train['age'] > 0]['age'].mean()
df_train['age'] = df_train['age'].fillna(replace_mean)
df_test['age'] = df_test['age'].fillna(replace_mean)

#embark의 결측값을 최빈값 대체하자
embarked_mode = df_train['embarked'].value_counts().index[0]
df_train['embarked'] = df_train['embarked'].fillna(embarked_mode)
df_test['embarked'] = df_test['embarked'].fillna(embarked_mode)

# 원-핫 인코딩, 
# 통합 데이터 프레임 (whwole_df) 생성
whole_df = df_train.append(df_test)
train_idx_num = len(df_train)

# pandas 패키지를 이용해서 원-핫 인코딩 수행
whole_df_encoded = pd.get_dummies(whole_df)
df_train = whole_df_encoded[:train_idx_num]
df_test = whole_df_encoded[:train_idx_num]

df_train.head()


# In[35]:


from sklearn.linear_model import LogisticRegression
from sklearn.metrics import accuracy_score, precision_score, recall_score, f1_score


# In[51]:


#데이터를 학습 데이터와 테스트 데이터로 분리
x_train, y_train = df_train.loc[:, df_train.columns != 'survived'].values, df_train['survived'].values


# In[52]:


x_test, y_test = df_train.loc[:, df_train.columns != 'survived'].values, df_train['survived'].values


# In[66]:


# 로지스틱 회귀 모델 학습

lr = LogisticRegression(random_state=0)
lr.fit(x_train, y_train)


# In[54]:


y_pred = lr.predict(x_test)
print(y_pred)


# In[64]:


y_pred_p = lr.predict_proba(x_test)[:,1]
print(y_pred_p)


# In[72]:


# 로지스틱 회귀 모델과 더불어 분류 분석의 가장 대표적인 방법인 
# 의사결정나무(Decision Tree) 모델을 적용해 보자

# 의사결정나무 모델은 피처 단위로 조건을 분기하여 정답의 집합을 좁혀나가는 방법
# 마치 스무고개 놀이에서 정답을 찾아 나가는 과정과 유사하다. 

#                         남자?
#                     예        노
#                 나이 > 10?        생존  0.73
#              예            노
#         사망                  가족
#     0.17%                  사망    생존
    
    
from sklearn.tree import DecisionTreeClassifier    

# 의사 결정 나무를 학습하고, 학습한 모델로 테스트 데이터셋에 대한 예측값을 반환한다.
dtc = DecisionTreeClassifier()
dtc.fit(x_train, y_train)
y_pred = dtc.predict(x_test)

print(y_pred)

y_pred_p = dtc.predict_proba(x_test)[:,1]
print(y_pred_p)


# In[71]:


##########################################
# 모델 개선
##########################################

# 분류 모델의 성능을 더욱 끌어올리기 위해서...
# 1) 좋은 분류 기법을 사용해야 한다. 
# 2) 더 많은 데이터를 사용한다. 

# 3) 피처 엔지니어링 Feature Engineering
#    피처 엔지니어링은 모델에 사용할 피처를 가공하는 분석 작업을 말한다. 



df_test = pd.read_csv("data/titanic_test.csv")
df_train = pd.read_csv("data/titanic_train.csv")


# In[73]:


# 전처리

# age의 결측값을 평균값으로 대체하자
replace_mean = df_train[df_train['age'] > 0]['age'].mean()
df_train['age'] = df_train['age'].fillna(replace_mean)
df_test['age'] = df_test['age'].fillna(replace_mean)

#embark의 결측값을 최빈값 대체하자
embarked_mode = df_train['embarked'].value_counts().index[0]
df_train['embarked'] = df_train['embarked'].fillna(embarked_mode)
df_test['embarked'] = df_test['embarked'].fillna(embarked_mode)

# 원-핫 인코딩, (카테고리 정규화)
# 통합 데이터 프레임 (whwole_df) 생성
whole_df = df_train.append(df_test)
train_idx_num = len(df_train)


# In[63]:


#피처 엔지니어링 & 전처리 

# cabin 피처 활용

# 결측 데이터를 'X' 대체
whole_df['cabin'] = whole_df['cabin'].fillna('X')

# cabin 피처의 첫 번째 알파벳을 추출하다.
whole_df['cabin'] = whole_df['cabin'].apply(lambda x:x[0])

# 추출한 알파벳 중에 G T 수가 너무작기 때문에 X 대체
whole_df['cabin'] = whole_df['cabin'].replace({"G":"X", "T":"X"})

ax = sns.countplot(x='cabin', hue = 'survived', data = whole_df)
plt.show()


# In[74]:


#피처 엔지니어링 & 전처리 

# cabin 피처 활용

# 결측 데이터를 'X' 대체
whole_df['cabin'] = whole_df['cabin'].fillna('X')

# cabin 피처의 첫 번째 알파벳을 추출하다.
whole_df['cabin'] = whole_df['cabin'].apply(lambda x:x[0])

# 추출한 알파벳 중에 G T 수가 너무작기 때문에 X 대체
whole_df['cabin'] = whole_df['cabin'].replace({"G":"X", "T":"X"})

ax = sns.countplot(x='cabin', hue = 'survived', data = whole_df)
plt.show()


# In[75]:


# name 피처 성 호칭 이름
whole_df.head()

name_grade = whole_df['name'].apply(lambda x : x.split(",",1)[1].split(".")[0])
name_grade = name_grade.unique().tolist()
print(name_grade)


# In[77]:


# 호칭에 따라서 사회적 지위를 정의 (1910 기준)
grade_dict = {'A' : ['Rev', 'Col', 'Major', 'Dr', 'Capt','Sir'],
                 'B' : ['Ms','Mme','Mrs','Dona'],
                 'C' : ['Jonkheer','the Countess'],
                 'D' : ['Mr','Don'],
                 'E' : ['Master'],
                 'F' : ['Miss','Mile','Lady']}


# In[78]:


def five_grade(x):
    grade = x.split(",",1)[1].split(".")[0]
    for key, value in grade_dict.times():
        for title in value:
            if grade == title:
                return key
        return 'G'


# In[79]:


print(whole_df)


# In[82]:


whole_df_encoded = pd.get_dummies(whole_df)
df_train = whole_df_encoded[:train_idx_num]
df_test = whole_df_encoded[:train_idx_num:]
df_train.head()


# In[86]:


who|le_df['body'] = whole_df['body'].fillna('0')


# In[85]:


df_train.head(10)


# In[ ]:




