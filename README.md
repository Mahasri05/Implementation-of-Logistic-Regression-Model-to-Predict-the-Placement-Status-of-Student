# Implementation-of-Logistic-Regression-Model-to-Predict-the-Placement-Status-of-Student

## AIM:
To write a program to implement the the Logistic Regression Model to Predict the Placement Status of Student.

## Equipments Required:
1. Hardware – PCs
2. Anaconda – Python 3.7 Installation / Jupyter notebook

## Algorithm
1. 1. Import the required packages and print the present data

2. Print the placement data and salary data.

3. Find the null and duplicate values.

4. Using logistic regression find the predicted values of accuracy, confusion matrices

5. Display the results.


## Program:
```
/*
Program to implement the the Logistic Regression Model to Predict the Placement Status of Student.
Developed by:Ramya G 
RegisterNumber: 24003270
*/
'''
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
```


## Output:
![the Logistic Regression Model to Predict the Placement Status of Student](sam.png)

HEAD
import pandas as pd
data=pd.read_csv("Placement_Data.csv")
data.head()
![Screenshot 2024-11-21 204653](https://github.com/user-attachments/assets/4e797a18-7e1d-480e-a3ef-714867bdca17)

COPY
data1=data.copy()
data1=data1.drop(["sl_no","salary"],axis=1)
data1.head()
![Screenshot 2024-11-21 204822](https://github.com/user-attachments/assets/3d86d7c9-ab9f-4f87-9ad6-92a53dd5edfb)

FIT TRANSFORM
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
![Screenshot 2024-11-21 204948](https://github.com/user-attachments/assets/be242661-6787-4438-ba17-0b8c974a5d02)

LOGISTIC REGRESSION
from sklearn.model_selection import train_test_split
x_train,x_test,y_train,y_test=train_test_split(x,y,test_size=0.2,random_state=0)
from sklearn.linear_model import LogisticRegression
lr=LogisticRegression(solver="liblinear")
lr.fit(x_train,y_train)
y_pred=lr.predict(x_test)
print(y_pred)
print()
![Screenshot 2024-11-21 205012](https://github.com/user-attachments/assets/fb8e8185-446a-4db6-a3f3-7d05f256f3a3)

ACCURACY SCORE
from sklearn.metrics import accuracy_score
accuracy=accuracy_score(y_test,y_pred)
print(accuracy)
print()
![Screenshot 2024-11-21 205033](https://github.com/user-attachments/assets/daad4a2a-6062-40fb-808c-50f0fd5e9890)

CONFUSSION MATRIX
from sklearn.metrics import confusion_matrix
confusion=confusion_matrix(y_test,y_pred)
print(confusion)
print()
![Screenshot 2024-11-21 205042](https://github.com/user-attachments/assets/8dff4e09-04d9-4f34-8fa3-aa8d20406dd9)

CLASSIFICATION REPORT
from sklearn.metrics import classification_report
classification_report1=classification_report(y_test,y_pred)
print(classification_report1)
![Screenshot 2024-11-21 205054](https://github.com/user-attachments/assets/9326a61c-1c1b-45e5-9eb3-d2f248390028)

PREDICTION
lr.predict([[1,80,1,90,1,1,90,1,0,85,1,85]])

![Screenshot 2024-11-21 205119](https://github.com/user-attachments/assets/9dd9c1bc-1fbd-4400-8d65-aed3f7d43d9e)



## Result:
Thus the program to implement the the Logistic Regression Model to Predict the Placement Status of Student is written and verified using python programming.
