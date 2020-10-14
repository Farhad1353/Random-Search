import numpy as np 
np.set_printoptions(suppress=True) # dont print every number in scientific form
import matplotlib.pyplot as plt
import sys

from utils import get_regression_data # function to create dummy data for regression

n_samples = 20
X =  np.random.uniform(-5,5,(n_samples,1))
Y =  2.5*np.power(X,2) - (3.4*X) + 3.5


class LinearHypothesis:
    def __init__(self): # initalize parameters 
        self.w1 = np.random.randn()
        self.w2 = np.random.randn()## randomly initialise weight
        self.b = np.random.randn() ## randomly initialise bias
        
    def __call__(self, X): # how do we calculate output from an input in our model? 
        y_hat = self.w1*np.power(X,2) + self.w2*X + self.b ## make a prediction using a hypothesis
        return y_hat # return prediction
    
    def update_params(self, new_w1, new_w2, new_b):
        self.w1 = new_w1 ## set this instance's weights to the new weight value passed to the function
        self.w2 = new_w2
        self.b = new_b ## do the same for the bias


H = LinearHypothesis() # instantiate our model

y_hat = H(X) # make prediction

print(' These are the initial weight and bias prior to the random search : W1:{:.4f}, W2:{:.4f} and B: {:.4f} \n'.format(H.w1, H.w2, H.b))


def plot_h_vs_y(X, y_hat, Y):
    #print(X.shape,y_hat.shape,Y.shape)
    plt.figure()
    plt.scatter(X, Y, c='r', label='Label')
    plt.scatter(X, y_hat, c='b', label='Hypothesis', marker='x')
    plt.legend()
    plt.xlabel('X')
    plt.ylabel('Y')
    plt.show()
    
plot_h_vs_y(X, y_hat, Y)

def L(y_hat, labels): # define our criterion (loss function)
    #print(y_hat.shape,labels.shape)
    errors = y_hat - labels ## calculate errors
    squared_errors = errors ** 2 ## square errors
    mean_squared_error = sum(squared_errors) / len(squared_errors) ## calculate mean 
    return mean_squared_error # return loss

cost = L(y_hat, Y)

print("Initial Cost :{:.2f} \n".format(cost[0]))

def random_search(n_samples, limit=20):
    """Try out n_samples of random parameter pairs and return the best ones"""
    best_weights1 = None ## no best weight found yet
    best_weights2 = None
    best_bias = None ## no best bias found yet
    lowest_cost = float('inf') ## initialize it very high (how high can it be?)
    for i in range(0, n_samples): ## try this many different parameterisations
        w1 = np.random.uniform(-limit, limit) ## randomly sample a weight within the limits of the search
        w2 = np.random.uniform(-limit, limit)
        b = np.random.uniform(-limit, limit) ## randomly sample a bias within the limits of the search
        i=i
        H.update_params(w1, w2, b) ## update our model with these random parameters
        y_hat = H(X) ## make prediction
        cost = L(y_hat, Y) ## calculate loss
        if cost < lowest_cost: ## if this is the best parameterisation so far
            lowest_cost = cost ## update the lowest running cost to the cost for this parameterisation
            best_weights1 = w1 ## get best weights so far from the model
            best_weights2 = w2
            best_bias = b ## get best bias so far from the model
    print('Lowest cost of {:.2f} achieved with weights of W1:{:.4f}, W2:{:.4f} and bias of {:.4f}'.format(lowest_cost[0],best_weights1, best_weights2,best_bias))
    return best_weights1,best_weights2, best_bias ## return the best weight and best bias

best_weights1, best_weights2, best_bias = random_search(10000) # do 10000 samples in a random search 
H.update_params(best_weights1, best_weights2, best_bias) # make sure to set our model's weights to the best values we found
plot_h_vs_y(X, H(X), Y) # plot model predictions agains labels

