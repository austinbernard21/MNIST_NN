# MNIST_NN
Neural network built from scratch on MNIST dataset using only pandas and numpy


CSV files are split into two sets(training and test) with 1st column entry as value(1-10) and remaining 784 column entries as pixel value.
train.csv has a size of (60,000 x 785)
test.csv has a size of (10,000 x 785)

I used a neural network with one hidden layer and achieved a accuracy score of 90%, using mini-batch gradient descent as my optimization algorithm. I used a mini-batch size of 64 with a network with input layer as 784, a hidden layer as 275, and an output layer of 10. Learning rate was 0.475 an ran mini-batch- gradient descent for 10 epochs
