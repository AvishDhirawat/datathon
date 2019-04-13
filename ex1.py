import mnist_loader
training_data, validation_data, test_data = mnist_loader.load_data_wrapper()
import network
net = network.Network([784, 30, 10])
net.SGD(training_data, 10, 10, 0.1, test_data=test_data)
    Epoch 0: 5105 / 10000
    Epoch 1: 5887 / 10000
    Epoch 2: 7147 / 10000
    Epoch 3: 7566 / 10000
    Epoch 4: 7763 / 10000
    Epoch 5: 7869 / 10000
    Epoch 6: 7948 / 10000
    Epoch 7: 8019 / 10000
    Epoch 8: 8073 / 10000
    Epoch 9: 8111 / 10000