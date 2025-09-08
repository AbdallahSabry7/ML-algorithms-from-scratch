import numpy as np
import pandas as pd
import random


class LinearRegression:
    def __init__(self,learning_rate=0.01,max_iter=1000):
        self.weights = None
        self.bias = 0
        self.learning_rate = learning_rate
        self.max_iter = max_iter

    def fit(self, x, y):
        self.x = x
        self.y = y
        self.weights = np.random.randn(x.shape[1])   
        for epoch in range(self.max_iter):
            y_pred = self.predict(x)
            dw = -2 * x.T.dot(y - y_pred) / len(y)
            db = -2 * np.sum(y - y_pred) / len(y)
            self.weights -= self.learning_rate * dw
            self.bias -= self.learning_rate * db

    def predict(self, x):
        if self.weights is None:
            raise ValueError("Model is not fitted yet.")
        x = np.array(x)
        if x.ndim == 1:
            x = x.reshape(1, -1)
        return np.dot(x, self.weights) + self.bias


    
    def simple_try(self,x,y):
        self.x = x
        self.y = y
        self.weights = np.random.normal()
        for epoch in range(self.max_iter):
            idx = random.randint(0, len(x)-1)
            data = x[idx]
            label = y[idx]
            y_pred = self.predict(data)
            if label > y_pred:
                if data > 0:
                    self.weights += self.learning_rate
                    self.bias += self.learning_rate
                else:
                    self.weights -= self.learning_rate
                    self.bias -= self.learning_rate
            elif label < y_pred:
                if data > 0:
                    self.weights -= self.learning_rate
                    self.bias -= self.learning_rate
                else:
                    self.weights += self.learning_rate
                    self.bias += self.learning_rate
        return self.weights, self.bias
    
    def abs_try(self,x,y):
        self.x = x
        self.y = y
        self.weights = np.random.normal()
        for epoch in range(self.max_iter):
            idx = random.randint(0, len(x)-1)
            data = x[idx]
            label = y[idx]
            y_pred = self.predict(data)
            if label > y_pred:
                self.weights += self.learning_rate * data
                self.bias += self.learning_rate
            elif label < y_pred:
                self.weights -= self.learning_rate * data
                self.bias -= self.learning_rate
        return self.weights, self.bias


    def sqr_try(self,x,y):
        self.x = x
        self.y = y
        self.weights = np.random.normal()
        for epoch in range(self.max_iter):
            idx = random.randint(0, len(x)-1)
            data = x[idx]
            label = y[idx]
            y_pred = self.predict(data)
            self.weights += self.learning_rate * (label - y_pred) * data
            self.bias += self.learning_rate * (label - y_pred)
        return self.weights, self.bias
    
if __name__ == "__main__":

    def standardize_fit(x):
        mean = np.mean(x, axis=0)  
        std = np.std(x, axis=0)    
        return mean, std
    def standardize_transform(x, mean, std):
        return (x - mean) / std

    data ={
        'price': [155,197,244,356,407,448],
        'no_of_rooms': [1, 2, 3, 5, 6, 7]
    }
    new_data = {
        'price': [155, 197, 244, 356, 407, 448],
        'no_of_rooms': [1, 2, 3, 5, 6, 7],
        'area': [98, 135, 215, 310, 344, 405]
    }

    df = pd.DataFrame(data)

    df2 = pd.DataFrame(new_data)

    x = df[['no_of_rooms']].values
    y = df['price'].values
    x2 = df2[['no_of_rooms','area']].values
    y2 = df2['price'].values

    x_mean, x_std = standardize_fit(x)
    x2_mean, x2_std = standardize_fit(x2)

    x_standardized = standardize_transform(x, x_mean, x_std)
    x2_standardized = standardize_transform(x2, x2_mean, x2_std)

    test1 = [[8], [4], [10]]  
    test2 = [[8, 400], [4, 200], [10, 500]]

    test1_standardized = standardize_transform(np.array(test1), x_mean, x_std)
    test2_standardized = standardize_transform(np.array(test2), x2_mean, x2_std)

    model = LinearRegression(learning_rate=0.0001, max_iter=10000)
    model.fit(x2_standardized, y2)
    print("Weights:", model.weights)
    print("Bias:", model.bias)
    predictions = model.predict(np.array(test2_standardized))
    print("Predictions:", predictions)

    model2 = LinearRegression(learning_rate=0.01, max_iter=1000)
    model2.sqr_try(x_standardized, y)
    print("Weights:", model2.weights)
    print("Bias:", model2.bias)
    predictions2 = model2.predict(np.array(test1_standardized))
    print("Predictions:", predictions2)
