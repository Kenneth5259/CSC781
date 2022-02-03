import knn
from sklearn.datasets import load_digits
from random import randint
import matplotlib.pyplot as plt
import numpy as np

# Find Highest Performing K for both distance equations
def get_metrics(train_x, train_y, input_x, input_y, input_name, k_max, k_inc, metric_function, metric_name, iteration):
    # generate list of K from 1 to max under given increment
    k_list = list(range(1, k_max, k_inc))

    # initialize empty performance arrays
    performance_euclidean = []
    performance_manhattan = []

    # iterate over K values
    for k in k_list:

        # Notice of Progress
        print('Testing K value: %s Metric: %s'%(k, metric_name))

        # Initialize empty prediction arrays
        pred_labels_e = []
        pred_labels_m = []

        # iterate over data in input set
        for data in input_x:

            # get predictions for distance functions
            pred_e = knn.prediction_classification(train_x, train_y, data, k, knn.euclidean_distance)
            pred_m = knn.prediction_classification(train_x, train_y, data, k, knn.manhattan_distance)

            # append to prediction arrays
            pred_labels_e.append(pred_e)
            pred_labels_m.append(pred_m)

        # evaluate metrics
        met_e = metric_function(input_y, pred_labels_e)
        met_m = metric_function(input_y, pred_labels_m)

        print(met_e, met_m)

        #append to metrics array
        performance_euclidean.append(met_e)
        performance_manhattan.append(met_m)

    # Initialize figure
    plt.figure(figsize=(20,6))
    # Add plots for distance functions
    plt.plot(k_list, performance_euclidean, marker="o", label="Euclidean Distance")
    plt.plot(k_list, performance_manhattan, marker="o", label="Manhattan Distance")
    # Add Legend
    plt.legend()
    plt.xlabel("K Values")
    plt.ylabel("Accuracy")
    plt.title("Performace on %s Set"%(input_name))
    plt.savefig("%s_%s_%s.png"%(input_name, metric_name, iteration))

def test_accuracy():
    # Prediction array
    predictions = []
    # Iterate over the data
    for data in test_x:
        pred = knn.prediction_classification(train_x, train_y, data, 1, knn.euclidean_distance)
        predictions.append(pred)
    # Generate the Score
    score = knn.accuracy_score(test_y, predictions)
    # Output the result
    print("Test Set Accuracy Score: %s"%(score))

def random_10_tests():
    # 10 Values
    for _ in range (10):
        # Generate random index
        index = randint(0, len(test_x) -1)
        # Get test row
        x = test_x[index]
        # Get real label value
        y = test_y[index]
        # Get prediction
        pred = knn.prediction_classification(train_x, train_y, x, 1, knn.euclidean_distance)
        # Visualize Prediction vs Actual
        print("Predicted Label: %s, Actual Label: %s"%(pred, y))


# Load the digits
digits = load_digits()

# Grab Data and Labels
data = digits.data
labels = digits.target

# Adjust dimensions
labels = np.expand_dims(labels, 1)

# Append
data = np.append(data, labels, 1)

# Randomize the data 
np.random.shuffle(data)

# create dev, train, test sets
dev, train, test = knn.train_dev_test_split(data, 70, 15)

# check for data leaks
knn.data_leaking_check(dev, train)
knn.data_leaking_check(dev, test)
knn.data_leaking_check(train, test)

# split into x, y
dev_x, dev_y = knn.get_features_and_labels(dev)
train_x, train_y = knn.get_features_and_labels(train)
test_x, test_y = knn.get_features_and_labels(test)

# Dev Metrics
#get_metrics(train_x, train_y, dev_x, dev_y, "Dev", 15, 1, knn.accuracy_score, "Accuracy", 10)

# Test Set Accuracy
test_accuracy()
