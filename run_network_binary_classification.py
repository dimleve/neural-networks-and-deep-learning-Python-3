import network
import pandas as pd
from sklearn.model_selection import train_test_split
import numpy as np

data = pd.read_csv('./data_banknote_authentication.txt', header=None)
# data = pd.read_csv('./sonar.all-data', header=None)
# data = pd.read_csv('./ionosphere.data', header=None)

y = data.iloc[:, data.shape[1]-1]
X = data.drop(data.shape[1]-1, axis=1)

X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.33, random_state=42)

training_inputs = [np.reshape(x, (data.shape[1]-1, 1)) for x in X_train.values]
training_inputs_y = [np.reshape(x, (1, 1)) for x in y_train]

training_data = list(zip(training_inputs, training_inputs_y))

test_inputs = [np.reshape(x, (data.shape[1]-1, 1)) for x in X_test.values]
test_data = list(zip(test_inputs, y_test))

# define a Neural Network with 30 hidden layers and 10 output layers
net = network.Network([data.shape[1]-1, 25, 1], is_binary_classification=True)

# train and evaluate the Neural Network
# training_data, epochs, mini_batch_size, eta, test_data=None
net.SGD(training_data, 120, 5, 3.0, test_data=test_data)
