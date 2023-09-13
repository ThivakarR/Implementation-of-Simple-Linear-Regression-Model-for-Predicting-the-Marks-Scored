# Implementation-of-Simple-Linear-Regression-Model-for-Predicting-the-Marks-Scored

## AIM:
To write a program to predict the marks scored by a student using the simple linear regression model.

## Equipments Required:
1. Hardware – PCs
2. Anaconda – Python 3.7 Installation / Jupyter notebook

## Algorithm
1. import the required libraries and read the dataframe.
2. Assign hours to X and scores to Y.
3. Implement training set and test set of the dataframe.
4. Plot the required graph both for test data and training data.
5. Find the values of MSE , MAE and RMSE.

## Program:
```
Program to implement the simple linear regression model for predicting the marks scored.
Developed by: G.Chethan Kumar
RegisterNumber: 212222240022 
```
```python
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from sklearn.metrics import mean_absolute_error, mean_squared_error
df = pd.read_csv('dataset/student_scores.csv')
df.head()

#segregating data to variables
x = df.iloc[:, :-1].values
x

#splitting train and test data
y = df.iloc[:, -1].values
y

from sklearn.model_selection import train_test_split
x_train, x_test, y_train, y_test = train_test_split(x, y, test_size = 1/3, random_state = 0)

from sklearn.linear_model import LinearRegression 
regressor=LinearRegression()
regressor.fit(x_train,y_train)
y_pred=regressor.predict(x_test)

#displaying predicted values
y_pred

#displaying actual values
y_test

#graph plot for training data
plt.scatter(x_train,y_train,color="orange")
plt.plot(x_train,regressor.predict(x_train),color="red")
plt.title("Hours vs Scores (Training Set)")
plt.xlabel("Hours")
plt.ylabel("Scores")
plt.show()

#graph plot for test data
plt.scatter(x_test,y_test,color="purple")
plt.plot(x_train,regressor.predict(x_train),color="green")
plt.title("Hours vs Scores (Training Set)")
plt.xlabel("Hours")
plt.ylabel("Scores")
plt.show()

mse=mean_squared_error(y_test,y_pred)
print("MSE= ",mse)

mae=mean_absolute_error(y_test,y_pred)
print("MAE = ",mae)

rmse=np.sqrt(mse)
print("RMSE = ",rmse)
```
## Output:
### df.head()

![Screenshot from 2023-09-01 07-21-27](https://github.com/Gchethankumar/Implementation-of-Simple-Linear-Regression-Model-for-Predicting-the-Marks-Scored/assets/118348224/657d1d92-38a9-4094-8ed9-1f34525fc339)

### df.tail()

![Screenshot from 2023-09-13 19-19-23](https://github.com/Gchethankumar/Implementation-of-Simple-Linear-Regression-Model-for-Predicting-the-Marks-Scored/assets/118348224/6baf3d89-3554-4a80-ba75-b029b2731d7e)

### Array value of X

![Screenshot from 2023-09-01 07-21-38](https://github.com/Gchethankumar/Implementation-of-Simple-Linear-Regression-Model-for-Predicting-the-Marks-Scored/assets/118348224/572364a4-56b9-4a22-8d24-8546d76438df)

### Array value of Y

![Screenshot from 2023-09-01 07-21-53](https://github.com/Gchethankumar/Implementation-of-Simple-Linear-Regression-Model-for-Predicting-the-Marks-Scored/assets/118348224/c07d1859-8927-4097-9343-f8cba764cc8f)

### Values of Y prediction and Values of Y prediction

![Screenshot from 2023-09-01 07-22-04](https://github.com/Gchethankumar/Implementation-of-Simple-Linear-Regression-Model-for-Predicting-the-Marks-Scored/assets/118348224/c5f7ac09-c0f8-48bc-944c-3256eb75213d)

![Screenshot from 2023-09-01 07-22-17](https://github.com/Gchethankumar/Implementation-of-Simple-Linear-Regression-Model-for-Predicting-the-Marks-Scored/assets/118348224/a28462a2-1ae2-41be-9965-4d8897088ca9)

### Training Set Graph

![Screenshot from 2023-09-01 07-22-26](https://github.com/Gchethankumar/Implementation-of-Simple-Linear-Regression-Model-for-Predicting-the-Marks-Scored/assets/118348224/edbb21da-2800-4d93-9db2-c922c6def611)

### Test Set Graph

![Screenshot from 2023-09-01 07-22-33](https://github.com/Gchethankumar/Implementation-of-Simple-Linear-Regression-Model-for-Predicting-the-Marks-Scored/assets/118348224/b61ed986-b31d-46c4-978d-3e5a037e5272)

### Values of MSE, MAE and RMSE

![Screenshot from 2023-09-01 07-22-46](https://github.com/Gchethankumar/Implementation-of-Simple-Linear-Regression-Model-for-Predicting-the-Marks-Scored/assets/118348224/82744637-d103-4e83-848e-8580a48c6060)


## Result:
Thus the program to implement the simple linear regression model for predicting the marks scored is written and verified using python programming.
