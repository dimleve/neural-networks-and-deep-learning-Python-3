import mnist_loader
import network

# load the MNIST data set
training_data, validation_data, test_data = mnist_loader.load_data_wrapper()
training_data = list(training_data)

# define a Neural Network with 30 hidden layers and 10 output layers
net = network.Network([784, 30, 10])

# train and evaluate the Neural Network
net.SGD(training_data, 5, 10, 3.0, test_data=test_data)
