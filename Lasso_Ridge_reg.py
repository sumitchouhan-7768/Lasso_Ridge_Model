import pandas as pd 
import numpy as np 
import seaborn as sns  # type: ignore
import matplotlib.pyplot as plt  
from sklearn import preprocessing 
from sklearn.preprocessing import PolynomialFeatures 
from sklearn.model_selection import train_test_split 
from sklearn.linear_model import LinearRegression
from sklearn.linear_model import LinearRegression, Ridge, Lasso
from sklearn.metrics import r2_score

df = pd.read_csv(r"car_mpg.csv")

# data cleaning 
df = df.drop(['car_name'], axis = 1) # type: ignore
df['origin'] = df['origin'].replace({1:'America',2:'Europe',3:'Asia'})
df = pd.get_dummies(df,columns = ['origin'])
df = df.replace('?',np.nan)
df = df.apply(pd.to_numeric, errors='coerce')  # Convert all applicable columns to numeric
df = df.apply(lambda x: x.fillna(x.median()),axis = 0)

#Model Building
X = df.drop(['mpg'],axis = 1) #independent variable
Y = df[['mpg']] #dependent variable

#Scaling Data
X_s = preprocessing.scale(X)
x_s = pd.DataFrame(X_s,columns = X.columns)

Y_s = preprocessing.scale(Y)
Y_s = pd.DataFrame(Y_s, columns = Y.columns)

#Splitting Data into Train , Test sets

X_train,X_test,Y_train,Y_test = train_test_split(X_s,Y_s, test_size=0.30, random_state = 1)
X_train.shape

#Fitting of SLR and finding coeff
regression_model = LinearRegression()
regression_model.fit(X_train,Y_train)

X_train = pd.DataFrame(X_train, columns=X.columns) #Converting X_train back to a DataFrame

for idx, col_name in enumerate(X_train.columns):  
    print('The coefficient for {} is {}'.format(col_name, regression_model.coef_[0][idx]))


intercept = regression_model.intercept_[0]
print('The intercept is {}'.format(intercept))

# To reduce magnitude of coeff  using Ridge

ridge_model = Ridge(alpha = 0.3)
ridge_model.fit(X_train,Y_train)

print('Ridge model coef: {}'.format(ridge_model.coef_))

# To reduce magnitude of coef using Lasso
lasso_model = Lasso(alpha =0.1)
lasso_model.fit(X_train,Y_train)

print('Lasso model coef: {}'.format(lasso_model.coef_))

# Score Comparison
#SLR
print(regression_model.score(X_train,Y_train))
print(regression_model.score(X_test,Y_test))

print("*****************************")
#Ridge
print(ridge_model.score(X_train,Y_train))
print(ridge_model.score(X_test,Y_test))

print("******************************")
#Lasso
print(lasso_model.score(X_train,Y_train))
print(lasso_model.score(X_test,Y_test))

#Model Paremeter Tuning
data_train_test = pd.concat([X_train,Y_train],axis = 1)
data_train_test.head()

import statsmodels.formula.api as smf
ols1 = smf.ols(formula = 'mpg ~ cyl+disp+hp+wt+acc+yr+car_type+origin_America+origin_Europe+origin_Asia', 
               data=data_train_test).fit()

ols1.params
print(ols1.summary())
#Lets check Sum of Squared Errors (SSE) by predicting value of y for test cases and subtracting from the actual y for the test cases
mse  = np.mean((regression_model.predict(X_test)-Y_test)**2)
import math
rmse = math.sqrt(mse)
print('Root Mean Squared Error: {}'.format(rmse))

#Lets check the residuals for some of these predictor.
fig = plt.figure(figsize=(10,8))
sns.residplot(x= X_test['hp'], y= Y_test['mpg'], color='green', lowess=True )


fig = plt.figure(figsize=(10,8))
sns.residplot(x= X_test['acc'], y= Y_test['mpg'], color='green', lowess=True )
# predict mileage (mpg) for a set of attributes not in the training or test set
y_pred = regression_model.predict(X_test)
plt.scatter(Y_test['mpg'], y_pred)




