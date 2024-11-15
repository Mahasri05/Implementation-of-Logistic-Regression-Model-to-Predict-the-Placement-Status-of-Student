# Implementation-of-Logistic-Regression-Model-to-Predict-the-Placement-Status-of-Student

## AIM:
To write a program to implement the the Logistic Regression Model to Predict the Placement Status of Student.

## Equipments Required:
1. Hardware – PCs
2. Anaconda – Python 3.7 Installation / Jupyter notebook

## Algorithm 

1. Import the required packages and print the present data

2. Print the placement data and salary data.

3. Find the null and duplicate values.

4. Using logistic regression find the predicted values of accuracy, confusion matrices

5. Display the results

## Program:
```
/*
Program to implement the the Logistic Regression Model to Predict the Placement Status of Student.
Developed by:Mahasri D 
RegisterNumber:24901210 
*/
```
import pandas as pd
data=pd.read_csv("Placement_Data.csv")
data.head()
data1=data.copy()
data1=data1.drop(["sl_no","salary"],axis=1)
data1.head()
data1.isnull().sum()
data1.duplicated().sum()
from sklearn.preprocessing import LabelEncoder
le=LabelEncoder()
data1["gender"]=le.fit_transform(data1["gender"])
data1["ssc_b"]=le.fit_transform(data1["ssc_b"])
data1["hsc_b"]=le.fit_transform(data1["hsc_b"])
data1["hsc_s"]=le.fit_transform(data1["hsc_s"])
data1["degree_t"]=le.fit_transform(data1["degree_t"])
data1["workex"]=le.fit_transform(data1["workex"])
data1["specialisation"]=le.fit_transform(data1["specialisation"])
data1["status"]=le.fit_transform(data1["status"])

x=data1.iloc[:,:-1]
x
print(x)
y=data1["status"]
y
print(y)
print()
from sklearn.model_selection import train_test_split
x_train,x_test,y_train,y_test=train_test_split(x,y,test_size=0.2,random_state=0)
from sklearn.linear_model import LogisticRegression
lr=LogisticRegression(solver="liblinear")
lr.fit(x_train,y_train)
y_pred=lr.predict(x_test)
print(y_pred)
print()
from sklearn.metrics import accuracy_score
accuracy=accuracy_score(y_test,y_pred)
print(accuracy)
print()
from sklearn.metrics import confusion_matrix
confusion=confusion_matrix(y_test,y_pred)
print(confusion)
print()
from sklearn.metrics import classification_report
classification_report1=classification_report(y_test,y_pred)
print(classification_report1)
lr.predict([[1,80,1,90,1,1,90,1,0,85,1,85]])

## Output:
![the Logistic Regression Model to Predict the Placement Status of Student](sam.png)


HEAD:
![Screenshot 2024-11-15 205155](https://github.com/user-attachments/assets/385b0873-5790-4418-80af-6dc1c8b3dc21)

COPY:
![Screenshot 2024-11-15 205219](https://github.com/user-attachments/assets/8a402dc7-c4c9-4662-b44a-45e1b4876883)


FIT TRANSFORM:
from sklearn.preprocessing import LabelEncoder
le=LabelEncoder()
data1["gender"]=le.fit_transform(data1["gender"])
data1["ssc_b"]=le.fit_transform(data1["ssc_b"])
data1["hsc_b"]=le.fit_transform(data1["hsc_b"])
data1["hsc_s"]=le.fit_transform(data1["hsc_s"])
data1["degree_t"]=le.fit_transform(data1["degree_t"])
data1["workex"]=le.fit_transform(data1["workex"])
data1["specialisation"]=le.fit_transform(data1["specialisation"])
data1["status"]=le.fit_transform(data1["status"])
![Screenshot 2024-11-15 175914](https://github.com/user-attachments/assets/11878aa2-c474-4586-9628-14b5ef16a2e9)

LOGISTIC REGRESSION:
from sklearn.model_selection import train_test_split
x_train,x_test,y_train,y_test=train_test_split(x,y,test_size=0.2,random_state=0)
from sklearn.linear_model import LogisticRegression
lr=LogisticRegression(solver="liblinear")
lr.fit(x_train,y_train)
y_pred=lr.predict(x_test)
print(y_pred)
array(![Screenshot 2024-11-15 180318](https://github.com/user-attachments/assets/25cf9f6c-4c5c-473d-b917-b862ed96c4b5),dtype=int64)

ACCURACY SCORE:
from sklearn.metrics import accuracy_score
accuracy=accuracy_score(y_test,y_pred)
print(accuracy)
![Screenshot 2024-11-15 180344](https://github.com/user-attachments/assets/8ccf5b57-ea4a-4fb4-9a54-7165531cc400)

CONFUSSION MATRIX:
from sklearn.metrics import confusion_matrix
confusion=confusion_matrix(y_test,y_pred)
print(confusion)
array(![Screenshot 2024-11-15 180357]),(https://github.com/user-attachments/assets/e6a4d970-92b9-4733-9d70-c92144c7b14c), dtype=int64


CLASSIFICATION REPORT:
from sklearn.metrics import classification_report
classification_report1=classification_report(y_test,y_pred)
print(classification_report1)
![Screenshot 2024-11-15 180420](https://github.com/user-attachments/assets/2e004eca-6ffe-4bbb-9d0d-eb7e2cd6815f)

PREDICTION:
lr.predict([[1,80,1,90,1,1,90,1,0,85,1,85]])
![Screenshot 2024-11-15 180704](https://github.com/user-attachments/assets/92f1447d-ba76-41ad-8060-70c50e44c45b)




## Result:
Thus the program to implement the the Logistic Regression Model to Predict the Placement Status of Student is written and verified using python programming.
