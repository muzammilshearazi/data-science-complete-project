
# ### Load the Data and Libraries

import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
import pandas as pd


data = pd.read_csv("DMV_Written_Tests.csv")
data.head()

data.info()

scores = data[['DMV_Test_1', 'DMV_Test_2']].values
results = data['Results'].values


# ###  Define the Logistic Sigmoid Function $\sigma(z)$

def logistic_function(x):    
    return 1/ (1 + np.exp(-x))


logistic_function(0)


# ###  Compute the Cost Function and Gradient


def compute_cost(theta, x, y):
    m = len(y)
    y_pred = logistic_function(np.dot(x , theta))
    error = (y * np.log(y_pred)) + ((1 - y) * np.log(1 - y_pred))
    cost = -1 / m * sum(error)
    gradient = 1 / m * np.dot(x.transpose(), (y_pred - y))
    return cost[0] , gradient


# ###  Cost and Gradient at Initialization


mean_scores = np.mean(scores, axis=0)
std_scores = np.std(scores, axis=0)
scores = (scores - mean_scores) / std_scores #standardization

rows = scores.shape[0]
cols = scores.shape[1]

X = np.append(np.ones((rows, 1)), scores, axis=1) #include intercept
y = results.reshape(rows, 1)

theta_init = np.zeros((cols + 1, 1))
cost, gradient = compute_cost(theta_init, X, y)

print("Cost at initialization", cost)
print("Gradient at initialization:", gradient)


# ### Gradient Descent


def gradient_descent(x, y, theta, alpha, iterations):
    costs = []
    for i in range(iterations):
        cost, gradient = compute_cost(theta, x, y)
        theta -= (alpha * gradient)
        costs.append(cost)
    return theta, costs


theta, costs = gradient_descent(X, y, theta_init, 1, 200)


print("Theta after running gradient descent:", theta)
print("Resulting cost:", costs[-1])

# ### Predictions using the optimized theta values



def predict(theta, x):
    results = x.dot(theta)
    return results > 0


p = predict(theta, X)
print("Training Accuracy:", sum(p==y)[0],"%")


import json
from flask import request
from flask import jsonify
from flask import Flask, render_template
app = Flask(__name__, template_folder='templates')  # still relative to module

@app.route('/')
def home():
    return render_template('dmv.html')
    
@app.route('/message',methods=['POST'])
def hello():
    message=request.get_json(force=True)
    score1=message['score1']
    score2=message['score2']
    test = np.array([int(score1),int(score2)])
    test = (test - mean_scores)/std_scores
    test = np.append(np.ones(1), test)
    probability = logistic_function(test.dot(theta))
    

    
    response={
        'greeting':'A person who scores '+ score1 + ' and ' + score2  +' on their DMV written tests have a '+ str(np.round(probability[0], 2))  +   '  probability of passing!'
    }
    return jsonify(response)
