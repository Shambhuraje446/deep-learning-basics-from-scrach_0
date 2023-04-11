# deep-learning-basics-from-scrach_0 (3 Layers neural network)
## Problem Statement :
- The Street View House Numbers Predictions.
#### ![Screenshot 2023-03-28 184733](https://user-images.githubusercontent.com/116808590/228262687-ebb264b1-4039-40c7-b8e8-8c33e1e1e66e.png)
## Requirements :
- python
- numpy
- scipy.io
## Dataset :
- link of dataset : http://ufldl.stanford.edu/housenumbers/   {Take test_32x32.mat and train_32x32.mat datasets}
## Architacture :
- Implement 4 layer deep feed-forward neural with a sigmoid activation function(1 Input, 2 Hidden, and 1 Output) layer.
- Implement Xavier weight initialization.
- Implement Cross-entropy loss
- Implement SGD and RMSProp optimizer to update the weights.
- Hyperparameter of the network : learning rate, number of neurons in hidden layers, and no. of iterations.
- Report following things : Accuracy, Cost, and F1-score.
## Usage :
- Run this command : python Q1.py
## Loss function curves are stored in (No. of neurons - No. of iterations - Optimizer) file.
- In which we have 8 curves of different Hyperparameters.
## Result :
- Train Accuracy : 18.92%
- Test Accuracy : 19.58%
- f1-score : 0.1958.
#### Results are stored in data.txt.
## Go through code and Enjoy the analysis.
