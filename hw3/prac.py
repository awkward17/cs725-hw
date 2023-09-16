import numpy as np
import matplotlib.pyplot as plt
import pandas as pd
import pickle as pkl


class NaiveBayes:
    def fit(self, X, y):
        # Calculate the number of unique classes
        num_classes = len(np.unique(y))

        # Calculate prior probabilities for each class
        priors = {}
        for class_label in range(num_classes):
            priors[str(class_label)] = np.mean(y == class_label)

        # Fit distribution for each feature for each class
        gaussian = {}
        bernoulli = {}
        laplace = {}
        exponential = {}
        multinomial = {}

        for class_label in range(num_classes):
            # Filter data for the current class
            class_data = X[y == class_label]

            # Calculate Gaussian MLE parameters (μ, σ²) for X1 and X2
            gaussian[str(class_label)] = [np.mean(class_data[:, 0]), np.var(class_data[:, 0]),
                                          np.mean(class_data[:, 1]), np.var(class_data[:, 1])]

            # Calculate Bernoulli MLE parameters (p_x3, p_x4) for X3 and X4
            bernoulli[str(class_label)] = [np.mean(class_data[:, 2]), np.mean(class_data[:, 3])]

            # Calculate Laplace MLE parameters (μ, b) for X5 and X6
            laplace[str(class_label)] = [np.mean(class_data[:, 4]), np.mean(np.abs(class_data[:, 4] - np.mean(class_data[:, 4])))]

            # Calculate Exponential MLE parameters (λ) for X7 and X8
            exponential[str(class_label)] = [1 / np.mean(class_data[:, 6]), 1 / np.mean(class_data[:, 7])]

            # Calculate Multinomial MLE parameters (p_x9, p_x10) for X9 and X10
            p_x9 = (class_data[:, 8] == 1).mean()
            p_x10 = (class_data[:, 9] == 1).mean()
            multinomial[str(class_label)] = [p_x9, p_x10]

        self.priors = priors
        self.gaussian = gaussian
        self.bernoulli = bernoulli
        self.laplace = laplace
        self.exponential = exponential
        self.multinomial = multinomial

    def predict(self, X):
        predictions = []
        for datapoint in X:
            class_probabilities = {}
            for class_label, prior in self.priors.items():
                # Calculate the posterior probability for each class
                posterior = np.log(prior)

                # Calculate Gaussian posterior
                for i in range(2):
                    posterior += self.log_gaussian_pdf(datapoint[i], self.gaussian[class_label][i * 2], np.sqrt(self.gaussian[class_label][i * 2 + 1]))

                # Calculate Bernoulli posterior
                for i in range(2, 4):
                    posterior += self.log_bernoulli_pmf(datapoint[i], self.bernoulli[class_label][i - 2])

                # Calculate Laplace posterior
                for i in range(4, 6):
                    posterior += self.log_laplace_pdf(datapoint[i], self.laplace[class_label][i - 4], self.laplace[class_label][i - 5])

                # Calculate Exponential posterior
                for i in range(6, 8):
                    posterior += self.log_exponential_pdf(datapoint[i], 1 / self.exponential[class_label][i - 6])

                # Calculate Multinomial posterior
                for i in range(8, 10):
                    posterior += np.log(self.multinomial[class_label][i - 8]) * datapoint[i]

                class_probabilities[class_label] = posterior

            # Assign the class with the highest posterior probability as the prediction
            predicted_class = max(class_probabilities, key=class_probabilities.get)
            predictions.append(int(predicted_class))

        return np.array(predictions)

    def getParams(self):
        return self.priors, self.gaussian, self.bernoulli, self.laplace, self.exponential, self.multinomial

    def log_gaussian_pdf(self, x, mean, var):
        return -0.5 * (np.log(2 * np.pi * var) + (x - mean) ** 2 / var)

    def log_bernoulli_pmf(self, x, p):
        return x * np.log(p) + (1 - x) * np.log(1 - p)

    def log_laplace_pdf(self, x, mean, b):
        return -np.log(2 * b) - np.abs(x - mean) / b

    def log_exponential_pdf(self, x, rate):
        return np.log(rate) - rate * x 

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
    """Calculate the multiclass F1 score of the predictions.

    Args:
        predictions (np.array): The predicted labels.
        true_labels (np.array): The true labels.

    Returns:
        float(list): The F1 score of the predictions for each class.
    """



    def precision(predictions, true_labels, label):
        """Calculate the multiclass precision of the predictions for a specific class.

        Args:
            predictions (np.array): The predicted labels.
            true_labels (np.array): The true labels.
            label (int): The class label for which precision is calculated.

        Returns:
            float: The precision of the predictions.
        """
        true_positive = np.sum((predictions == label) & (true_labels == label))
        false_positive = np.sum((predictions == label) & (true_labels != label))

        if true_positive + false_positive == 0:
            return 0.0
        else:
            return true_positive / (true_positive + false_positive)

    def recall(predictions, true_labels, label):
        """Calculate the multiclass recall of the predictions for a specific class.

        Args:
            predictions (np.array): The predicted labels.
            true_labels (np.array): The true labels.
            label (int): The class label for which recall is calculated.

        Returns:
            float: The recall of the predictions.
        """
        true_positive = np.sum((predictions == label) & (true_labels == label))
        false_negative = np.sum((predictions != label) & (true_labels == label))

        if true_positive + false_negative == 0:
            return 0.0
        else:
            return true_positive / (true_positive + false_negative)

    def f1score(predictions, true_labels, label):
        """Calculate the F1 score for a specific class using its precision and recall.

        Args:
            predictions (np.array): The predicted labels.
            true_labels (np.array): The true labels.
            label (int): The class label for which F1 score is calculated.

        Returns:
            float: The F1 score of the predictions.
        """
        precision_value = precision(predictions, true_labels, label)
        recall_value = recall(predictions, true_labels, label)

        if precision_value + recall_value == 0:
            return 0.0
        else:
            return 2 * (precision_value * recall_value) / (precision_value + recall_value)

    f1s = []
    for label in np.unique(true_labels):
        f1s.append(f1score(predictions, true_labels, label))
    return f1s

def accuracy(predictions, true_labels):
    """Calculate the accuracy of the predictions.

    Args:
        predictions (np.array): The predicted labels.
        true_labels (np.array): The true labels.

    Returns:
        float: The accuracy of the predictions.
    """
    return np.sum(predictions == true_labels) / predictions.size



if __name__ == "__main__":
    """

    You are not required to modify this part of the code.

    """

    # Load the data
    train_dataset = pd.read_csv('./train_dataset.csv',index_col=0).to_numpy()
    validation_dataset = pd.read_csv('./validation_dataset.csv',index_col=0).to_numpy()

    # Extract the data
    train_datapoints = train_dataset[:,:-1]
    train_labels = train_dataset[:, -1]
    validation_datapoints = validation_dataset[:, 0:-1]
    validation_labels = validation_dataset[:, -1]

    # Visualize the data
    visualise(train_datapoints, train_labels)

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
    visualise(validation_datapoints, validation_predictions)