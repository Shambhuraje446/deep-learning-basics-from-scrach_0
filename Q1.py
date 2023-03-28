#!/usr/bin/env python
# coding: utf-8

# In[1]:


import numpy as np
import matplotlib.pyplot as plt
import scipy.io
import random


# In[2]:


# Load SVHN dataset
data = scipy.io.loadmat("train_32x32.mat")
X_train = data['X']
Y_train = data['y']

data = scipy.io.loadmat("test_32x32.mat")
X_test = data['X']
Y_test = data['y']


# In[3]:


# Flatten the images

X_train = X_train.reshape(X_train.shape[0]*X_train.shape[1]*X_train.shape[2], -1)
X_test = X_test.reshape(X_test.shape[0]*X_test.shape[1]*X_test.shape[2], -1)

# Normalize the images

X_train = X_train/255.
X_test = X_test/255.

# Convert labels to one-hot encoding

Y_train = np.eye(10)[Y_train.flatten()-1].T
Y_test = np.eye(10)[Y_test.flatten()-1].T

# Print the shape of the data
print("X_train shape:", X_train.shape)
print("Y_train shape:", Y_train.shape)
print("X_test shape:", X_test.shape)
print("Y_test shape:", Y_test.shape)


# In[4]:


def sigmoid(x):
    return 1 / (1 + np.exp(-x))

def softmax(x):
    expX = np.exp(x)
    return expX/np.sum(expX, axis = 0)


# In[5]:


def sigmoid_derivative(x):
    s = sigmoid(x)
    return s * (1 - s)


# In[6]:


def initialize_parameters(n_x, n_h, n_y):
    
    np.random.seed(2)
    
    w1 = np.random.randn(n_h, n_x) * np.sqrt(1 / n_x)
    b1 = np.zeros((n_h, 1))
    
    w2 = np.random.randn(n_h, n_h) * np.sqrt(1 / n_h)
    b2 = np.zeros((n_h, 1))
    
    w3 = np.random.randn(n_y, n_h) * np.sqrt(1 / n_h)
    b3 = np.zeros((n_y, 1))
    
    parameters = {"w1": w1, 
                  "b1": b1, 
                  "w2": w2, 
                  "b2": b2,
                  "w3": w3, 
                  "b3": b3}
    return parameters


# In[7]:


def forward_propagation(x, parameters):
    
    w1 = parameters['w1']
    b1 = parameters['b1']
    w2 = parameters['w2']
    b2 = parameters['b2']
    w3 = parameters['w3']
    b3 = parameters['b3']
    
    
    z1 = np.dot(w1, x) + b1
    a1 = sigmoid(z1)
    
    z2 = np.dot(w2, a1) + b2
    a2 = sigmoid(z2)
    
    z3 = np.dot(w3, a2) + b3
    a3 = softmax(z3)
    
    
    forward_cache = {
        "z1" : z1,
        "a1" : a1,
        "z2" : z2,
        "a2" : a2,
        "z3" : z3,
        "a3" : a3
    }
    
    return forward_cache


# In[8]:


def cost_function(A2, Y):
    
    m = Y.shape[1]
    
    logprobs_1 = np.multiply(np.log(A2), Y)
    logprobs_2 = np.multiply(np.log(1 - A2),(1 - Y))
    logprobs = logprobs_1 + logprobs_2
    cost = - (1/m) * np.sum(logprobs)
    
    cost = float(np.squeeze(cost))
    
    return cost


# In[9]:


def backward_prop(x, y, parameters, forward_cache):
    
    w1 = parameters['w1']
    b1 = parameters['b1']
    w2 = parameters['w2']
    b2 = parameters['b2']
    w3 = parameters['w3']
    b3 = parameters['b3']
    
    a1 = forward_cache['a1']
    a2 = forward_cache['a2']
    a3 = forward_cache['a3']
    
    m = x.shape[1]
    
    dz3 = (a3 - y)
    dw3 = (1/m)*np.dot(dz3, a2.T)
    db3 = (1/m)*np.sum(dz3, axis = 1, keepdims = True)
    
    dz2 = (1/m)*np.dot(w3.T, dz3)*sigmoid_derivative(a2)
    dw2 = (1/m)*np.dot(dz2, a1.T)
    db2 = (1/m)*np.sum(dz2, axis = 1, keepdims = True)
    
    dz1 = (1/m)*np.dot(w2.T, dz2)*sigmoid_derivative(a1)
    dw1 = (1/m)*np.dot(dz1, x.T)
    db1 = (1/m)*np.sum(dz1, axis = 1, keepdims = True)
    
    gradients = {
        "dw1" : dw1,
        "db1" : db1,
        "dw2" : dw2,
        "db2" : db2,
        "dw3" : dw3,
        "db3" : db3
    }
    
    return gradients


# In[10]:


def update_parameters_sgd(parameters, gradients, learning_rate):
    
    w1 = parameters['w1']
    b1 = parameters['b1']
    w2 = parameters['w2']
    b2 = parameters['b2']
    w3 = parameters['w3']
    b3 = parameters['b3']
    
    dw1 = gradients['dw1']
    db1 = gradients['db1']
    dw2 = gradients['dw2']
    db2 = gradients['db2']
    dw3 = gradients['dw3']
    db3 = gradients['db3']
    
    w1 = w1 - learning_rate*dw1
    b1 = b1 - learning_rate*db1
    w2 = w2 - learning_rate*dw2
    b2 = b2 - learning_rate*db2
    w3 = w3 - learning_rate*dw3
    b3 = b3 - learning_rate*db3
    
    parameters = {
        "w1" : w1,
        "b1" : b1,
        "w2" : w2,
        "b2" : b2,
        "w3" : w3,
        "b3" : b3
    }
    
    return parameters


# In[11]:


def update_parameters_rmsprop(parameters, gradients, learning_rate, epsilon=1e-8):
    
    # Retrieve the parameters from the dictionary
    w1 = parameters["w1"]
    b1 = parameters["b1"]
    w2 = parameters["w2"]
    b2 = parameters["b2"]
    w3 = parameters["w3"]
    b3 = parameters["b3"]
    
    # Retrieve the gradients from the dictionary
    dw1 = gradients["dw1"]
    db1 = gradients["db1"]
    dw2 = gradients["dw2"]
    db2 = gradients["db2"]
    dw3 = gradients["dw3"]
    db3 = gradients["db3"]
    
    s_dW1 = np.zeros_like(dw1)
    s_db1 = np.zeros_like(db1)
    s_dW2 = np.zeros_like(dw2)
    s_db2 = np.zeros_like(db2)
    s_dW3 = np.zeros_like(dw3)
    s_db3 = np.zeros_like(db3)
    
    beta = 0.9
    
    s_dW1 = beta * s_dW1 + (1 - beta) * dw1 ** 2
    s_db1 = beta * s_db1 + (1 - beta) * db1 ** 2
    s_dW2 = beta * s_dW2 + (1 - beta) * dw2 ** 2
    s_db2 = beta * s_db2 + (1 - beta) * db2 ** 2
    s_dW3 = beta * s_dW3 + (1 - beta) * dw3 ** 2
    s_db3 = beta * s_db3 + (1 - beta) * db3 ** 2
    
    w1 -= learning_rate * dw1 / (np.sqrt(s_dW1) + epsilon)
    b1 -= learning_rate * db1 / (np.sqrt(s_db1) + epsilon)
    w2 -= learning_rate * dw2 / (np.sqrt(s_dW2) + epsilon)
    b2 -= learning_rate * db2 / (np.sqrt(s_db2) + epsilon)
    w3 -= learning_rate * dw3 / (np.sqrt(s_dW3) + epsilon)
    b3 -= learning_rate * db3 / (np.sqrt(s_db3) + epsilon)
    
    parameters = {"w1": w1, 
                  "b1": b1, 
                  "w2": w2, 
                  "b2": b2, 
                  "w3": w3, 
                  "b3": b3}
    
    return parameters


# In[ ]:





# In[12]:


def model(x, y, n_h, learning_rate, iterations, activation):
    
    n_x = x.shape[0]
    n_y = y.shape[0]
    
    cost_list = []
    
    parameters = initialize_parameters(n_x, n_h, n_y)
    
    for i in range(iterations):
        
        forward_cache = forward_propagation(x, parameters)
        
        cost = cost_function(forward_cache['a3'], y)
        
        gradients = backward_prop(x, y, parameters, forward_cache)
        
        if (activation == "SGD"):
            parameters = update_parameters_sgd(parameters, gradients, learning_rate)
        if (activation == "RMS_prop"):
            parameters = update_parameters_rmsprop(parameters, gradients, learning_rate, epsilon=1e-8)
        
        cost_list.append(cost)
        
        
    return parameters, cost_list


# In[13]:


def accuracy(inp, labels, parameters):
    forward_cache = forward_propagation(inp, parameters)
    a_out = forward_cache['a3']   # containes propabilities with shape(10, 1)
    
    a_out = np.argmax(a_out, 0)  # 0 represents row wise 
    
    labels = np.argmax(labels, 0)
    
    acc = np.mean(a_out == labels)*100
    
    return acc


# In[14]:


from sklearn.metrics import f1_score

iterations = [80, 120]
n_h = [800, 1600]
learning_rate = 0.001
activation = ["SGD", "RMS_prop"]


with open("data.txt", "a") as f:
    for i in range(2):
        for j in range(2):
            for k in range(2):
                
                f.write("A number of neurons in hidden layer is ")
                f.write(str(n_h[j]))
                f.write("\n")
                f.write("A number of iterations are ")
                f.write(str(iterations[i]))
                f.write("\n")
                f.write("The activation function used is   ")
                f.write(str(activation[k]))
                f.write("\n")
                
                Parameters, Cost_list = model(X_train, Y_train, n_h = n_h[j], learning_rate = learning_rate, iterations = iterations[i], activation=activation[k])
                
                for l in range(iterations[i]):
                    
                    if(l%(iterations[i]/10) == 0):
                        
                        f.write("Cost after )")
                        f.write(str(l))
                        f.write("iterations is : ")
                        f.write(str(Cost_list[l]))
                        f.write("\n")
                
                t = np.arange(0, iterations[i])
                plt.plot(t, Cost_list)
                plt.show()
                
                train_acc = accuracy(X_train, Y_train, Parameters)
                test_acc = accuracy(X_test, Y_test, Parameters)
                
                f.write("Accuracy of Train Dataset : ")
                f.write(str(train_acc))
                f.write("\n")
                
                f.write("Accuracy of Test Dataset : ")
                f.write(str(test_acc))
                f.write("\n")
                
                forward_cache = forward_propagation(X_test, Parameters)
                a_out = forward_cache['a3']
                y_pred = np.argmax(a_out, 0) 
                y_true= np.argmax(Y_test, 0)
                f1_score_1 = f1_score(y_true, y_pred, average='micro')
                            
                f.write("f1-score of model is ")
                f.write(str(f1_score_1))
                f.write("\n\n")


# In[ ]:




