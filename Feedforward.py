import numpy as np
from sklearn.model_selection import train_test_split
from sklearn.metrics import mean_squared_error

# Single Responsibility: Generates random data
class RandomDataGenerator:
    def __init__(self, n):
        self.n = n

    def generate(self):
        Xdata = np.random.rand(100, self.n)
        Ydata = 4 + 3 * Xdata + np.random.randn(100, self.n)
        return Xdata, Ydata

# Single Responsibility: Splits data into training and testing sets
class DataSplitter:
    def __init__(self, test_size=0.2, random_state=None):
        self.test_size = test_size
        self.random_state = random_state

    def split(self, X, Y):
        return train_test_split(X, Y, test_size=self.test_size, random_state=self.random_state)

# Single Responsibility: Defines the model
class FeedForwardModel:
    def __init__(self, input_size, hidden_size, output_size):
        self.weights_input_hidden = np.random.rand(input_size, hidden_size)
        self.weights_hidden_output = np.random.rand(hidden_size, output_size)

    def forward(self, X):
        self.hidden = np.dot(X, self.weights_input_hidden)
        self.output = np.dot(self.hidden, self.weights_hidden_output)
        return self.output

# Single Responsibility: Trains the model
class ModelTrainer:
    def __init__(self, model, learning_rate=0.01, epochs=100):
        self.model = model
        self.learning_rate = learning_rate
        self.epochs = epochs

    def train(self, X, Y):
        for epoch in range(self.epochs):
            output = self.model.forward(X)
            error = Y - output
            self.model.weights_hidden_output += self.learning_rate * np.dot(self.model.hidden.T, error)
            self.model.weights_input_hidden += self.learning_rate * np.dot(X.T, np.dot(error, self.model.weights_hidden_output.T))
        return self.model

# Single Responsibility: Evaluates the model
class ModelEvaluator:
    def evaluate(self, model, X_test, Y_test):
        predictions = model.forward(X_test)
        return mean_squared_error(Y_test, predictions)

# Generate data
data_generator = RandomDataGenerator(n=1)
X, Y = data_generator.generate()

# Split data
data_splitter = DataSplitter(test_size=0.2, random_state=42)
X_train, X_test, Y_train, Y_test = data_splitter.split(X, Y)

# Create and train the model
model = FeedForwardModel(input_size=1, hidden_size=10, output_size=1)
trainer = ModelTrainer(model=model, learning_rate=0.01, epochs=1000)
trained_model = trainer.train(X_train, Y_train)

# Evaluate the model
evaluator = ModelEvaluator()
mse = evaluator.evaluate(trained_model, X_test, Y_test)

print(f'Mean Squared Error: {mse}')
