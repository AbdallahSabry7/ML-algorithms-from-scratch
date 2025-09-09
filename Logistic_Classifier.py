import pandas as pd
import numpy as np

class LogisticClassifier:
    def __init__(self,learning_rate=0.01, max_iter=1000):
        self.learning_rate = learning_rate
        self.max_iter = max_iter
        self.weights = None
        self.bias = None

    def sigmoid(self,z):
        return 1/(1+np.exp(-z))
    
    def cross_entropy_loss(self,y,y_pred):
        loss = - (y * np.log(y_pred) + (1 - y) * np.log(1 - y_pred))
        return np.mean(loss)
    
    def fit(self,x,y):
        num_samples, num_features = x.shape
        self.weights = np.zeros(num_features)
        self.bias = 0

        for _ in range(self.max_iter):
            linear_model = np.dot(x, self.weights) + self.bias
            y_pred = self.sigmoid(linear_model)

            dw = (1 / num_samples) * np.dot(x.T, (y_pred - y))
            db = (1 / num_samples) * np.sum(y_pred - y)

            self.weights -= self.learning_rate * dw
            self.bias -= self.learning_rate * db

    def predict(self,x):
        linear_model = np.dot(x, self.weights) + self.bias
        y_pred = self.sigmoid(linear_model)
        y_pred_class = [1 if i > 0.5 else 0 for i in y_pred]
        return np.array(y_pred_class)
    
if __name__ == "__main__":
    from sklearn.model_selection import train_test_split
    from sklearn.datasets import load_iris
    from sklearn.preprocessing import StandardScaler
    from sklearn.metrics import accuracy_score

    iris = load_iris()
    X = iris.data
    y = (iris.target == 2).astype(int)  

    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.3, random_state=42)

    scaler = StandardScaler()
    X_train = scaler.fit_transform(X_train)
    X_test = scaler.transform(X_test)

    model = LogisticClassifier(learning_rate=0.1, max_iter=400)
    model.fit(X_train, y_train)

    y_pred = model.predict(X_test)

    accuracy = accuracy_score(y_test, y_pred)
    print(f"Accuracy: {accuracy * 100:.2f}%")