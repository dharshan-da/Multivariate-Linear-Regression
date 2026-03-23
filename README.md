# Implementation of Multivariate Linear Regression
## Aim
To write a python program to implement multivariate linear regression and predict the output.
## Equipment’s required:
1.	Hardware – PCs
2.	Anaconda – Python 3.7 Installation / Moodle-Code Runner
## Algorithm:
Step1

Import the required Python libraries such as NumPy, Pandas, and Scikit-learn

Step2

Load the dataset and separate the independent variables (features) and the dependent variable (target).

Step3

Split the dataset into training data and testing data.

Step4

Create a Linear Regression model and train it using the training dataset.

Step5

Use the trained model to predict the output for the test data and display the predicted results.

## Program:
```import matplotlib.pyplot as plt
import numpy as np
from sklearn import datasets, linear_model, metrics
from sklearn.model_selection import train_test_split

housing = datasets.fetch_california_housing()
X = housing.data
y = housing.target

X_train, X_test, y_train, y_test = train_test_split(
    X, y, test_size=0.4, random_state=1
)

reg = linear_model.LinearRegression()
reg.fit(X_train, y_train)

print('Coefficients: ', reg.coef_)
print('Variance score: {}'.format(reg.score(X_test, y_test)))

plt.style.use('fivethirtyeight')
plt.scatter(reg.predict(X_train),
            reg.predict(X_train) - y_train,
            color="green", s=10, label='Train data')

plt.scatter(reg.predict(X_test),
            reg.predict(X_test) - y_test,
            color="blue", s=10, label='Test data')

plt.hlines(y=0, xmin=0, xmax=5, linewidth=2)
plt.legend(loc='upper right')
plt.title('Residual Errors')
plt.show()






```
## Output:

![WhatsApp Image 2026-03-23 at 10 18 21 AM](https://github.com/user-attachments/assets/3aa96d9b-71bf-4203-b8d4-76f44b7e9158)


## Result
Thus the multivariate linear regression is implemented and predicted the output using python program.
