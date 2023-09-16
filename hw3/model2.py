import numpy as np
import matplotlib.pyplot as plt
import pandas as pd
import pickle as pkl



class NaiveBayes:

    def __init__(self, smoothing_alpha=0.000000001):
        self.smoothing_alpha = smoothing_alpha

    def fit(self, X, y):
        self.classes = np.unique(y)
        self.class_priors = {}
        self.distributions = {}
        self.features = X.shape[1]

        for c in self.classes:
            class_data = X[y == c]
            self.class_priors[c] = (len(class_data) + self.smoothing_alpha) / (len(X) + len(self.classes) * self.smoothing_alpha)
            self.distributions[c] = []

            for feature_idx in range(self.features):
                feature_data = class_data[:, feature_idx]
                if feature_idx in [0, 1]:
                    # Fit Gaussian distribution for X1 and X2
                    mean = np.mean(feature_data)
                    variance = np.var(feature_data) + self.smoothing_alpha
                    self.distributions[c].append(("gaussian", mean, variance))
                elif feature_idx in [2, 3]:
                    # Fit Bernoulli distribution for X3 and X4
                    p = (np.sum(feature_data) + self.smoothing_alpha) / (len(class_data) +  self.smoothing_alpha)
                    self.distributions[c].append(("bernoulli", p))
                elif feature_idx in [4, 5]:
                    # Fit Laplace distribution for X5 and X6
                    mu = np.mean(feature_data)
                    b = np.mean(np.abs(feature_data - mu)) + self.smoothing_alpha
                    self.distributions[c].append(("laplace", mu, b))
                elif feature_idx in [6, 7]:
                    # Fit Exponential distribution for X7 and X8
                    rate = 1 / (np.mean(feature_data) + self.smoothing_alpha)
                    self.distributions[c].append(("exponential", rate))
                elif feature_idx in [8, 9]:
                    # Fit Multinomial distribution for X9 and X10
                    k = int(np.max(feature_data)) + 1
                    counts = np.bincount(feature_data.astype(int), minlength=k) + self.smoothing_alpha
                    probabilities = counts / (len(feature_data))
                    self.distributions[c].append(("multinomial", probabilities))

    def predict(self, X):
        predictions = []

        for x in X:
            posteriors = {}

            for c in self.classes:
                posterior = np.log(self.class_priors[c])

                for feature_idx in range(self.features):
                    distribution = self.distributions[c][feature_idx]

                    if distribution[0] == "gaussian":
                        _, mean, variance = distribution
                        log_likelihood = -0.5 * (np.log(2 * np.pi * variance) + (x[feature_idx] - mean) ** 2 / variance)
                    elif distribution[0] == "bernoulli":
                        _, p = distribution
                        log_likelihood = (x[feature_idx] * np.log(p) + (1 - x[feature_idx]) * np.log(1 - p))
                    elif distribution[0] == "laplace":
                        _, mu, b = distribution
                        log_likelihood = -np.log(2 * b) - np.abs(x[feature_idx] - mu) / b
                    elif distribution[0] == "exponential":
                        _, rate = distribution
                        log_likelihood = np.log(rate) - rate * x[feature_idx]
                    elif distribution[0] == "multinomial":
                        _, probabilities = distribution
                        x_idx = int(x[feature_idx])
                        log_likelihood = np.log(probabilities[x_idx])

                    posterior += log_likelihood

                posteriors[c] = posterior

            predicted_class = max(posteriors, key=posteriors.get)
            predictions.append(predicted_class)

        return np.array(predictions)

    # ... (other methods and __init__ go here)

    def getParams(self):
        priors = {str(c): self.class_priors[c] for c in self.classes}
        gaussian = {str(c): [] for c in self.classes}
        bernoulli = {str(c): [] for c in self.classes}
        laplace = {str(c): [] for c in self.classes}
        exponential = {str(c): [] for c in self.classes}
        multinomial = {str(c): [] for c in self.classes}

        for c in self.classes:
        
            
            for feature_idx in range(self.features):
                distribution = self.distributions[c][feature_idx]
                
                if distribution[0] == "gaussian":
                    _, mean, variance = distribution
                    gaussian[str(c)].extend([mean,variance])

                elif distribution[0] == "bernoulli":
                    _, p = distribution
                    bernoulli[str(c)].append(p)
                elif distribution[0] == "laplace":
                    _, mu, b = distribution
                    laplace[str(c)].extend([mu, b])
                elif distribution[0] == "exponential":
                    _, rate = distribution
                    exponential[str(c)].append(rate)
                elif distribution[0] == "multinomial":
                    _, probabilities = distribution
                    multinomial[str(c)].append(probabilities)
            
        return priors, gaussian, bernoulli, laplace, exponential, multinomial
        
       


def save_model(model,filename="model.pkl"):
    """

    You are not required to modify this part of the code.

    """
    file = open("model.pkl","wb")
    pkl.dump(model,file)
    file.close()

def load_model(filename="model.pkl"):
    """

    You are not required to modify this part of the code.

    """
    file = open(filename,"rb")
    model = pkl.load(file)
    file.close()
    return model

def visualise(data_points,labels):
    """
    datapoints: np.array of shape (n,2)
    labels: np.array of shape (n,)
    """

    plt.figure(figsize=(8, 6))
    plt.scatter(data_points[:, 0], data_points[:, 1], c=labels, cmap='viridis')
    plt.colorbar()
    plt.title('Generated 2D Data from 5 Gaussian Distributions')
    plt.xlabel('Feature 1')
    plt.ylabel('Feature 2')
    plt.show()


def net_f1score(predictions, true_labels):
    """Calculate the multclass f1 score of the predictions.
    For this, we calculate the f1-score for each class 

    Args:
        predictions (np.array): The predicted labels.
        true_labels (np.array): The true labels.

    Returns:
        float(list): The f1 score of the predictions for each class
    """


    def precision(predictions, true_labels, label):
        """Calculate the multclass precision of the predictions.
        For this, we take the class with given label as the positive class and the rest as the negative class.

        Args:
            predictions (np.array): The predicted labels.
            true_labels (np.array): The true labels.

        Returns:
            float: The precision of the predictions.
        """
        """Start of your code."""
        true_positive = np.sum((predictions == label) & (true_labels == label))
        false_positive = np.sum((predictions == label) & (true_labels != label))

        if true_positive + false_positive == 0:
            return 0.0  # Avoid division by zero
        return true_positive / (true_positive + false_positive)



        
        """End of your code."""
        


    def recall(predictions, true_labels, label):
        """Calculate the multclass recall of the predictions.
        For this, we take the class with given label as the positive class and the rest as the negative class.
        Args:
            predictions (np.array): The predicted labels.
            true_labels (np.array): The true labels.

        Returns:
            float: The recall of the predictions.
        """
        """Start of your code."""
        true_positive = np.sum((predictions == label) & (true_labels == label))
        false_negative = np.sum((predictions != label) & (true_labels == label))

        if true_positive + false_negative == 0:
            return 0.0  # Avoid division by zero
        return true_positive / (true_positive + false_negative)

         



        """End of your code."""
        

    def f1score(predictions, true_labels, label):
        """Calculate the f1 score using it's relation with precision and recall.

        Args:
            predictions (np.array): The predicted labels.
            true_labels (np.array): The true labels.

        Returns:
            float: The f1 score of the predictions.
        """

        """Start of your code."""
        prec = precision(predictions, true_labels, label)
        rec = recall(predictions, true_labels, label)

        if prec + rec == 0:
            return 0.0  # Avoid division by zero
        return 2 * (prec * rec) / (prec + rec)




        """End of your code."""
        #return f1
    

    f1s = []
    for label in np.unique(true_labels):
        f1s.append(f1score(predictions, true_labels, label))
    return f1s

def accuracy(predictions,true_labels):
    """

    You are not required to modify this part of the code.

    """
    return np.sum(predictions==true_labels)/predictions.size



if __name__ == "__main__":
    """

    You are not required to modify this part of the code.

    """

    # Load the data
    train_dataset = pd.read_csv('./data/train_dataset.csv',index_col=0).to_numpy()
    validation_dataset = pd.read_csv('./data/validation_dataset.csv',index_col=0).to_numpy()

    # Extract the data
    train_datapoints = train_dataset[:,:-1]
    train_labels = train_dataset[:, -1]
    validation_datapoints = validation_dataset[:, 0:-1]
    validation_labels = validation_dataset[:, -1]

    # Visualize the data
    # visualise(train_datapoints, train_labels, "train_data.png")

    # Train the model
    model = NaiveBayes()
    model.fit(train_datapoints, train_labels)

    # Make predictions
    train_predictions = model.predict(train_datapoints)
    validation_predictions = model.predict(validation_datapoints)

    # Calculate the accuracy
    train_accuracy = accuracy(train_predictions, train_labels)
    validation_accuracy = accuracy(validation_predictions, validation_labels)

    # Calculate the f1 score
    train_f1score = net_f1score(train_predictions, train_labels)
    validation_f1score = net_f1score(validation_predictions, validation_labels)

    # Print the results
    print('Training Accuracy: ', train_accuracy)
    print('Validation Accuracy: ', validation_accuracy)
    print('Training F1 Score: ', train_f1score)
    print('Validation F1 Score: ', validation_f1score)

    # Save the model
    save_model(model)

    # Visualize the predictions
    #visualise(validation_datapoints, validation_predictions, "validation_predictions.png")

