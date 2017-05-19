import numpy as np
from numpy import array, fliplr, flipud, linalg, isnan, subtract, dot, random, multiply
from random import choice, shuffle
import mnist_data as md
import struct
from sklearn.neural_network import MLPClassifier

# features
def avg_value_by_grid(m):
    values =  [[0 for i in range(7)] for j in range(7)]
    features = []
    for i in range(len(m)):
        for j in range(len(m[0])):
            values[i // 4][j // 4] += m[i][j]

    for row in values:
        for j in row:
            features.append(j / 16)
    return features

def generate_features(training_data):
    feature_data = []
    for img in training_data:
        feature_data.append((img[0], avg_value_by_grid(img[1]))) # (label, [features])
    return feature_data

def getData():
    #get training set
    with open('train-labels-idx1-ubyte','rb') as fl:
        magic,num = struct.unpack(">II",fl.read(8))
        label = np.fromfile(fl,dtype=np.int8)
    with open('train-images-idx3-ubyte','rb') as fi:
        magic, num, rows, cols = struct.unpack(">IIII",fi.read(16))
        image = np.fromfile(fi,dtype=np.uint8).reshape(len(label),rows,cols)
        get_image = lambda idx: (label[idx],image[idx])
    ret = []
    for i in range(len(label)):
        ret.append(get_image(i))
    return ret

##################################

def run_classifier(training_features, training_labels, prediction_features, prediction_labels):
    classifier = MLPClassifier()
    classifier.fit(training_features,training_labels)
    predictions = classifier.predict(prediction_features)
    errors = 0
    for p in range(len(predictions)):
        if predictions[p] != prediction_labels[p]:
            errors += 1
    return predictions, errors

def main():
    training_data = generate_features(getData())
    prediction_data = []
    test_set_size = 6000; # pulling out random 10% of 60000 features = 6000
    for i in range(0, test_set_size): 
        random_idx = random.randint(0,len(training_data))
        prediction_data.append(training_data.pop(random_idx))

    training_features = [t[1] for t in training_data]
    training_labels = [t[0] for t in training_data]
    prediction_features = [t[1] for t in prediction_data]
    prediction_labels = [t[0] for t in prediction_data]

    predictions, errors = run_classifier(training_features, training_labels, prediction_features, prediction_labels)
    correct = len(predictions) - errors
    
    accuracy = int((float(len(predictions)) - float(errors)) / float(len(predictions)) * 100)
    print("Training Set Size: {}".format(len(training_data)))
    print("Test Set Size: {}".format(test_set_size))
    print("Correct: {}/{}, Accuracy: {} %".format(correct, len(predictions),  accuracy))
    return 0

main()
