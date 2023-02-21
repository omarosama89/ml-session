#!/opt/homebrew/bin/python3
import matplotlib.pyplot as plt
import numpy as np
from sklearn.linear_model import LinearRegression
from sklearn.model_selection import train_test_split

Total = 15
Number_features = 7
Images_namespace = 'first-trial'

mean = [10, 10] # here you can control the location of data
cov = [[10, 9.5],[9.5, 10]] # here you can control the dense and correlation for data

# ###############
# Here you need to generate random data using multivariate_normal method in numpy library
# Mean vector is already stored in `mean` variable, covariance matrix is stored `cov` and total number of
# data is stored in `Total`
# You should store data in `data` variable name
# ###############

features, response = data.T

years, years_test, y, y_test = train_test_split(features, response, test_size=0.33, random_state=1)
N = y.shape[0]
sort_indicies = np.argsort(years)

years = years[sort_indicies]
y = y[sort_indicies]

years = years.reshape(-1,1)
y = y.reshape(-1,1)
years_test = years_test.reshape(-1,1)
y_test = y_test.reshape(-1,1)


X = np.array([]).reshape(N, 0)
X_test = np.array([]).reshape(Total - N, 0)

plt.scatter(years, y)
plt.xlabel('Years of Experience')
plt.ylabel('Salary')
plt.savefig("data-{}.png".format(Images_namespace))
plt.clf()

number_of_features = np.array([])
train_error = np.array([])
test_error = np.array([])
Number_features = N
for i in range(Number_features):
    regressor = LinearRegression()

    x_powered = np.power(years, i)
    X = np.append(X, x_powered, axis=1)

    # ###############
    # Here you need to train the model, a LinearRegression instance is already defined under `regressor` variable name
    # Training data are already stored in a variable `X` and responses are stored in `y` variable
    # ###############

    # ###############
    # Here you need to predict responses for training data
    # Training data are already stored in a variable `X`
    # you need to store the predection in a variabe named `y_hat`
    # ###############

    plt.scatter(years, y)
    plt.xlabel('Years of Experience')
    plt.ylabel('Salary')

    plt.plot(years, y_hat, color='red')

    plt.savefig("linear_regresson_{}-{}.png".format(i,Images_namespace))
    plt.clf()
    number_of_features = np.append(number_of_features, i)

    # ###############
    # Here you need to calculate the training error
    # The actual response is stored in `y` and the prediction of training data is stored in `y_hat`
    # store the error in a variable named `mse_train`
    # ###############
    train_error = np.append(train_error, mse_train)

    years_test_powered = np.power(years_test, i)
    X_test = np.append(X_test, years_test_powered, axis=1)
    y_test_hat = regressor.predict(X_test)

    mse_test = np.square(y_test - y_test_hat).mean()
    test_error = np.append(test_error, mse_test)

plt.plot(number_of_features, train_error, color='green', label='Train error')
plt.plot(number_of_features, test_error, color='red',  label='Test error')
plt.legend()
plt.savefig("error-train-vs-test-{}.png".format(Images_namespace))
plt.clf()
