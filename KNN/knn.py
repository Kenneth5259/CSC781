from sklearn.metrics import accuracy_score
import numpy as np

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

'''
    Implementation of the manhattan distance equation
'''
def manhattan_distance(row1, row2):
    distance = 0.0
    for i in range(len(row1)-1):
        distance += abs(row1[i] - row2[i])
    return distance

'''
    Modified from lecture to allow distance metric as a parameter
'''
def get_neighbors(train_x, train_y, test_row, num_neighbors, dist_func):
    distances = []

    # get all distances
    for index in range(len(train_x)):
        train_row = train_x[index]
        train_label = train_y[index]
        dist = dist_func(train_row, test_row)
        distances.append((train_row, train_label, dist))

    # sort the distance list by distance
    distances.sort(key=lambda i: i[2])

    # get the k nearest neightbors and return
    output_neighbors = []
    output_labels = []
    output_distances = []
    for index in range(num_neighbors):
            output_neighbors.append(distances[index][0])
            output_labels.append(distances[index][1])
            output_distances.append(distances[index][2])
    
    return output_neighbors, output_labels, output_distances

'''
    From Lecture, I do not claim to have written this piece of code
'''
def prediction_classification(train_x, train_y, test_row, num_neighbors, distance_function):
    output_neighbors, output_labels, output_distances = get_neighbors(train_x, train_y, test_row, num_neighbors, distance_function)
    label_cnt = np.bincount(output_labels)
    prediction = np.argmax(label_cnt)
    return prediction
