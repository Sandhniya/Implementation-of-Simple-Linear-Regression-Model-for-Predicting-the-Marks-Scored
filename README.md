# Implementation-of-Simple-Linear-Regression-Model-for-Predicting-the-Marks-Scored

## AIM:
To write a program to predict the marks scored by a student using the simple linear regression model.

## Equipments Required:
1. Hardware – PCs
2. Anaconda – Python 3.7 Installation / Jupyter notebook

## Algorithm
```
1.Gather data consisting of two variables. Input- a factor that affects the marks and Output - the marks scored by students
2.Plot the data points on a graph where x-axis represents the input variable and y-axis represents the marks scored
3.Define and initialize the parameters for regression model: slope controls the steepness and intercept represents where the line crsses the y-axis
4.Use the linear equation to predict marks based on the input Predicted Marks = m.(hours studied) + b for each data point calculate the difference between the actual and predicted marks
5.Adjust the values of m and b to reduce the overall error. The gradient descent algorithm helps update these parameters based on the calculated error
6.Once the model parameters are optimized, use the final equation to predict marks for any new input data
```
## Program:
```
/*
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
#import libraries to find mae, mse
from sklearn.metrics import mean_absolute_error
from sklearn.metrics import mean_squared_error

#read csv file
df= pd.read_csv('data.csv')

#displaying the content in datafile
df.head()
df.tail()

# Segregating data to variables
X=df.iloc[:,:-1].values
X
y=df.iloc[:,-1].values
y

#splitting train and test data
from sklearn.model_selection import train_test_split
X_train,X_test,y_train,y_test=train_test_split(X,y,test_size=1/2,random_state=0)

#import linear regression model and fit the model with the data
from sklearn.linear_model import LinearRegression
regressor=LinearRegression()
regressor.fit(X_train,y_train)

#displaying predicted values
y_pred=regressor.predict(X_test)
y_pred

#displaying actual values
y_test

#graph plot for training data
import matplotlib.pyplot as plt
plt.scatter(X_train,y_train,color='red')
plt.plot(X_train,regressor.predict(X_train),color='blue')
plt.title("Hours vs Scores (Training Set)")
plt.xlabel("Hours")
plt.ylabel("Scores")

#graph plot for test data
plt.scatter(X_test,y_test,color='red')
plt.plot(X_test,regressor.predict(X_test),color='blue')
plt.title("Hours vs Scores (Testing Set)")
plt.xlabel("Hours")
plt.ylabel("Scores")

#find mae,mse,rmse
mse=mean_squared_error(y_test,y_pred)
print('MSE = ',mse)
mae=mean_absolute_error(y_test,y_pred)
print('MAE = ',mae)
rmse=np.sqrt(mse)
print('RMSE = ',rmse)
*/
```

## Output:
HEAD VALUES:
<img width="438" height="622" alt="image" src="https://github.com/user-attachments/assets/f32c4ffa-2029-4de0-a5c6-95fa65dba56d" />
TAIL  VALUES:
<img width="322" height="418" alt="image" src="https://github.com/user-attachments/assets/0f2020ec-8a9a-4679-bf0a-5fd11066fcb7" />
X VALUES:
<img width="349" height="808" alt="image" src="https://github.com/user-attachments/assets/475678bb-24e1-42c7-b149-a2baaa480026" />
Y VALUES:
<img width="1070" height="153" alt="image" src="https://github.com/user-attachments/assets/13d5172c-db5b-457a-ba72-d64c109ba55e" />
predicted values:
<img width="1042" height="98" alt="image" src="https://github.com/user-attachments/assets/f6368ee3-622c-4335-9417-8adcdc1069da" />
actual values:
<img width="855" height="47" alt="image" src="https://github.com/user-attachments/assets/956324dc-279c-4621-bd51-2e2a27585ac0" />
Training set:
<img width="831" height="667" alt="image" src="https://github.com/user-attachments/assets/5e6ba1a4-0b94-4946-b16b-aed305d567c6" />
Testing set:
<img width="838" height="666" alt="image" src="https://github.com/user-attachments/assets/bc909a76-c1fc-4efb-a3ea-469570465eee" />






## Result:
Thus the program to implement the simple linear regression model for predicting the marks scored is written and verified using python programming.
