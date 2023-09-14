# Implementation-of-Logistic-Regression-Model-to-Predict-the-Placement-Status-of-Student

## AIM:
To write a program to implement the the Logistic Regression Model to Predict the Placement Status of Student.

## Equipments Required:
1. Hardware – PCs
2. Anaconda – Python 3.7 Installation / Jupyter notebook

## Algorithm
1.Import the required libraries.

2.Load the dataset and check for null data values and duplicate data values in the dataframe.

3.Import label encoder from sklearn.preprocessing to encode the dataset.

4.Apply Logistic Regression on to the model.

5.Predict the y values.

6.Calculate the Accuracy,Confusion and Classsification report.

## Program:
```
/*
Program to implement the the Logistic Regression Model to Predict the Placement Status of Student.
Developed by: Barath S
RegisterNumber:  212222230018
*/
```
```python
import pandas as pd
df=pd.read_csv('Placement_Data (1).csv')
df.head()

df1=df.copy()
df1=df1.drop(["sl_no","salary"],axis=1)
df1.head()

df1.isnull().sum()

df1.duplicated().sum()

from sklearn.preprocessing import LabelEncoder
le=LabelEncoder()
df1["gender"]=le.fit_transform(df1["gender"])
df1["ssc_b"]=le.fit_transform(df1["ssc_b"])
df1["hsc_b"]=le.fit_transform(df1["hsc_b"])
df1["hsc_s"]=le.fit_transform(df1["hsc_s"])
df1["degree_t"]=le.fit_transform(df1["degree_t"])
df1["workex"]=le.fit_transform(df1["workex"])
df1["specialisation"]=le.fit_transform(df1["specialisation"])
df1["status"]=le.fit_transform(df1["status"])
df1

x=df1.iloc[:,:-1]
x

y=df1["status"]
y

from sklearn.model_selection import train_test_split
x_train,x_test,y_train,y_test=train_test_split(x,y,test_size=0.2,random_state=0)

from sklearn.linear_model import LogisticRegression
lr=LogisticRegression(solver="liblinear")
lr.fit(x_train,y_train)
y_pred=lr.predict(x_test)
y_pred

from sklearn.metrics import accuracy_score
accuracy=accuracy_score(y_test,y_pred)
accuracy

from sklearn.metrics import confusion_matrix
confusion=confusion_matrix(y_test,y_pred)
confusion

from sklearn.metrics import classification_report
classification_report1=classification_report(y_test,y_pred)
print(classification_report1)

lr.predict([[1,80,1,90,1,1,90,1,0,85,1,85]])
```

## Output:
## PLacement data
![model](https://github.com/barathsubramani/Implementation-of-Logistic-Regression-Model-to-Predict-the-Placement-Status-of-Student/blob/main/1st.png)

## Salary data
![model](https://github.com/barathsubramani/Implementation-of-Logistic-Regression-Model-to-Predict-the-Placement-Status-of-Student/blob/main/2nd.png)

## null()
![model](https://github.com/barathsubramani/Implementation-of-Logistic-Regression-Model-to-Predict-the-Placement-Status-of-Student/blob/main/3rd.png)

## duplicate
![model](https://github.com/barathsubramani/Implementation-of-Logistic-Regression-Model-to-Predict-the-Placement-Status-of-Student/blob/main/4th.png)
![model](https://github.com/barathsubramani/Implementation-of-Logistic-Regression-Model-to-Predict-the-Placement-Status-of-Student/blob/main/5th.png)

## status
![model](https://github.com/barathsubramani/Implementation-of-Logistic-Regression-Model-to-Predict-the-Placement-Status-of-Student/blob/main/6th.png)

## y_pred
![model](https://github.com/barathsubramani/Implementation-of-Logistic-Regression-Model-to-Predict-the-Placement-Status-of-Student/blob/main/7th.png)

## accuracy
![model](https://github.com/barathsubramani/Implementation-of-Logistic-Regression-Model-to-Predict-the-Placement-Status-of-Student/blob/main/8th.png)

## Confusion array
![model](https://github.com/barathsubramani/Implementation-of-Logistic-Regression-Model-to-Predict-the-Placement-Status-of-Student/blob/main/9th.png)

## Classification report
![model](https://github.com/barathsubramani/Implementation-of-Logistic-Regression-Model-to-Predict-the-Placement-Status-of-Student/blob/main/10th.png)

## Prediction of LR:
![model](https://github.com/barathsubramani/Implementation-of-Logistic-Regression-Model-to-Predict-the-Placement-Status-of-Student/blob/main/11th.png)


## Result:
Thus the program to implement the the Logistic Regression Model to Predict the Placement Status of Student is written and verified using python programming.
