import numpy as np

class KNNRegressor:
    def __init__(self, k):
        self.k = k
        self.points = []

    def add_point(self, x, y):
        self.points.append((x, y))

    def predict(self, X):
        if self.k > len(self.points):
            raise ValueError("k cannot be larger than the number of data points")
        
        distances = np.array([np.sqrt((X - x)**2 + (y1 - y2)**2) for (x, y1), (_, y2) in zip(self.points, self.points)])
        nearest_indices = np.argsort(distances)[:self.k]
        average_y = np.mean([self.points[i][1] for i in nearest_indices])
        
        return average_y

def main():
    N = int(input("Enter the number of points (N): "))
    k = int(input("Enter the number of nearest neighbors (k): "))
    
    knn_regr = KNNRegressor(k)

    print("Please enter the points:")
    for _ in range(N):
        x = float(input("Enter x value: "))
        y = float(input("Enter y value: "))
        knn_regr.add_point(x, y)
    
    X = float(input("Enter the query point X to predict Y: "))
    
    try:
        Y = knn_regr.predict(X)
        print(f"The predicted Y value is: {Y}")
    except ValueError as e:
        print(e)

if __name__ == "__main__":
    main()
