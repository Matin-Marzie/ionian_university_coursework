# Mohammad Matin Marzie
# inf2022001

# -*- coding: utf-8 -*-
"""
Created on Tue May  7 11:43:01 2019

@author: didep
"""
import numpy as np
import matplotlib.pyplot as plt
#
def sigmoid(x):
    sigm = 1 / (1 + (np.exp(-x)))
    return sigm
#
class neuralNetwork :
      def __init__(self, input_nodes, hidden_nodes, output_nodes, learning_rate):
#
#         arithmoi neurwnwn sthn eisodo, to krymmeno strwma kai thn exodo
#
          self.inodes = input_nodes
          self.hnodes = hidden_nodes
          self.onodes = output_nodes
          self.lr = learning_rate
#
#         arxikes tyxaies times gia ta barh
#         na sumplhrwthoun oi swstes diastaseis
#
          self.wih = (np.random.rand(self.hnodes, self.inodes) - 0.5)
          self.who = (np.random.rand(self.onodes, self.hnodes) - 0.5)
          pass
      
      def train(self, inputs_list, labels_list):
          inputs = np.array(inputs_list, ndmin=2).T
          labels = np.array(labels_list, ndmin=2).T
          hidden_inputs_a = np.dot(self.wih, inputs)
          hidden_outputs_b = sigmoid(hidden_inputs_a)
          final_inputs_c = np.dot(self.who, hidden_outputs_b)
          final_outputs_y = sigmoid(final_inputs_c)
          output_errors = labels - final_outputs_y
          hidden_errors = np.dot(self.who.T, output_errors)
          self.who += self.lr * np.dot((output_errors * final_outputs_y *
                      (1.0 - final_outputs_y)) , np.transpose(hidden_outputs_b))
          self.wih += self.lr * np.dot((hidden_errors * hidden_outputs_b *
                      (1.0 - hidden_outputs_b) ) , np.transpose(inputs) )
          pass
      def predict(self, inputs_list):
          inputs = np.array(inputs_list, ndmin=2).T
          hidden_inputs_a = np.dot(self.wih, inputs)
          hidden_outputs_b = sigmoid(hidden_inputs_a)
          final_inputs_c = np.dot(self.who, hidden_outputs_b)
          final_outputs_y = sigmoid(final_inputs_c)
          return final_outputs_y
#
#
#
input_nodes = 784
hidden_nodes = 100
output_nodes = 10
learning_rate = 0.3
#
n = neuralNetwork(input_nodes,hidden_nodes,output_nodes,learning_rate)
#
#   
# load the mnist training data CSV file into a list
#
training_data_file = open('mnist_train_100.csv', 'r')
training_data_list = training_data_file.readlines()
training_data_file.close()
# train the neural network
# go through all records in the training data set
for record in training_data_list:
    # split the record by the ',' commas
    all_values = record.split(',')
    # scale and shift the inputs
    inputs = (np.asarray(all_values[1:], dtype=np.float32) / 255.0 * 0.99) + 0.01
    # create the target output values (all 0.01, except the desired
    #label which is 0.99)
    labels = np.zeros(output_nodes) + 0.01
    # all_values[0] is the target label for this record
    labels[int(all_values[0])] = 0.99
    n.train(inputs, labels)
    pass
#
# load the mnist test data CSV file into a list
#
test_data_file = open('mnist_test_10.csv', 'r')
test_data_list = test_data_file.readlines()
test_data_file.close()
#
#
for test_number, test_digit in enumerate(test_data_list):
    all_values = test_digit.split(',')
    image_array = np.asarray(all_values[1:], dtype=np.float32).reshape((28,28))


    inputs = (np.asarray(all_values[1:], dtype=np.float32) / 255.0 * 0.99) + 0.01

    all_predict_y = n.predict(inputs)
    predicted_digit = np.argmax(all_predict_y)

    print('----------------------------------------')
    print("Test Number:      ", test_number)
    print("Prediction:       ", predicted_digit)
    print("With probability: ", all_predict_y[predicted_digit][0])
    print()
    print(n.predict(inputs))
    plt.title(f"Prediction: {predicted_digit}    With probability:  {int(round(all_predict_y[predicted_digit][0], 2)*100)}%")


    plt.imshow( image_array , cmap='Greys', interpolation='None')
    plt.show()
    #
    #
