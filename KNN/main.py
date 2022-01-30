from sklearn.datasets import load_digits
import numpy as np

digits = load_digits()

print(digits.data.shape)

'''
    Modified from lecture, converted into a reusable function
'''
def train_dev_test_split(data, p_train, p_dev):
    total_number = len(data)
    # convert percentage to decimal
    p_train = p_train/100
    # convert percentage to decimal, append percentage from train
    p_dev = p_dev/100 + p_train
    # train off first x percentage
    train = data[:int(total_number * p_train)]
    # dev off the next x percent
    dev = data[int(total_number*p_train):int(total_number*p_dev)]
    # test off remaining percent
    test = data[int(total_number*p_dev):]
    # return all portions of data
    return train, dev, test

'''
    From Lecture, I do not claim to have written this piece of code
'''
def get_features_and_labels(data):
    features = data[:, :-1]
    labels = data[:, -1]
    return features, labels

'''
    From Lecture, I do not claim to have written this piece of code
'''
def data_leaking_check(data1, data2):
    data_leaking = False
    for d1 in data1:
        for d2 in data2:
            if(np.array_equal(d1, d2)):
                data_leaking = True
                print("Find same sample: ")
                print(d1)
    if(not data_leaking):
        print("No Data Leaking!")

'''
    From Lecture, I do not claim to have written this piece of code
'''
def euclidean_distance(row1, row2):
    distance = 0.0
    for i in range(len(row1)-1):
        distance += (row1[i] - row2[i]) ** 2
    return np.sqrt(distance)

# Step 1 calculate distance

# Step 2 Store Distance in Array

# Step 3 Sort Array

# Step 4 Select first/last K elements in array for each label

# Step 5 Perform the majority voting, assing label to majority occurrences


# For Later
# - evaluation metrics: Accuracy, Precision, Recall, F1 Score