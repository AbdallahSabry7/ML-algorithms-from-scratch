import numpy as np

from sklearn.model_selection import train_test_split
from sklearn.datasets import load_iris

class KNN:
    def __init__(self,k=5):
        self.k = k
        self.X_train = None
        self.y_train = None
    
    def fit(self,x,y):
        self.X_train = x
        self.y_train = y

    def euclidean_distance(self,a,b):
         return np.sqrt(np.sum((a - b) ** 2, axis=-1))
    
    def predict(self,x):
        y_pred = [self._predict(xi) for xi in x]
        return np.array(y_pred)
    
    def _predict(self,x):
        distances = [self.euclidean_distance(x, x_train) for x_train in self.X_train]
        k_indices = np.argsort(distances)[:self.k]
        k_nearest_labels = [self.y_train[i] for i in k_indices]
        most_common = np.bincount(k_nearest_labels).argmax()
        return most_common
    


if __name__ == "__main__":
    iris = load_iris()
    X, y = iris.data, iris.target

    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.3, random_state=42)


    print(np.unique(y_test, return_counts=True))
    model = KNN(k=5)
    model.fit(X_train, y_train)

    predictions = model.predict(X_test)
    accuracy = np.sum(predictions == y_test) / len(y_test)
    print("Predictions:", predictions)
    print("True labels:", y_test)
    print(f"Accuracy: {accuracy:.2f}")