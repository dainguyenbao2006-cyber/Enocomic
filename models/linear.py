import numpy as np

class LinearRegression :
    def __init__(self , learningrate ,alpha , epochs , verbose):
        self.learningrate = learningrate
        self.epochs = epochs
        self.verbose = verbose
        self.alpha = alpha
        self.w = None
        self.b = None

    
    def predict(self , X) :
        return np.dot(X, self.w) + self.b
    
    def compute_loss(self,y, y_pred):
        mse = np.mean((y_pred - y)**2)
        l2_penalty = self.alpha * np.sum(self.w ** 2)
        return mse + l2_penalty
        
    def fit(self , X , y) :
        n_samples, n_features = X.shape
        self.w = np.zeros((n_features,1))
        self.b = 0

        for epoch in range(self.epochs) :
            y_pred = self.predict(X)
            error = y_pred - y
            dw = (1 / n_samples) * np.dot(X.T, error) + (self.alpha * self.w / n_samples)
            db = (1 / n_samples) * np.sum(y_pred- y)
            self.w-= self.learningrate * dw
            self.b -= self.learningrate * db
            if self.verbose and epoch % 1 == 0 :
                loss = self.compute_loss(y, y_pred)
                print(f"Tap huan luyen : {epoch} , Loss tuong ung : {loss : .4f}")

    def get_pragrams(self) :
        return self.w , self.b
    

    
        