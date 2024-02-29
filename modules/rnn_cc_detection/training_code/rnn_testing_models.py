import sys
import numpy as np
import pandas as pd
import argparse

import sklearn.ensemble
from sklearn import metrics
from sklearn.model_selection import train_test_split
from random import shuffle

import tensorflow as tf
import sklearn as sk
from tensorflow.keras import layers
from tensorflow.keras.preprocessing.sequence import pad_sequences
from tensorflow.keras.preprocessing.text import one_hot
from tensorflow.keras.models import load_model
from tensorflow.keras.utils import to_categorical

parser = argparse.ArgumentParser()
parser.add_argument(
    '-D',
    '--dataset_file',
    help='File containing data for training',
    type=str,
    required=True,
    default="../dataset_more_labels.dat",
)
parser.add_argument(
    '-M',
    '--max_letters',
    help='Max sequence length',
    type=int,
    required=False,
    default=500,
)
parser.add_argument(
    '-m',
    '--min_letters',
    help='Min sequence length',
    type=int,
    required=False,
    default=5,
)
parser.add_argument(
    '-v',
    '--verbose',
    help='Level of verbosity',
    type=bool,
    required=False,
    default=False,
)
parser.add_argument(
    '-b',
    '--batch_size',
    help='Size of the minibatch',
    type=int,
    required=False,
    default=100,
)
parser.add_argument(
    '-e',
    '--epochs',
    help='Number of epochs in training',
    type=int,
    required=False,
    default=200,
)
parser.add_argument(
    '-S',
    '--model_file',
    help='Where to store the train model',
    type=str,
    required=False,
)
args = parser.parse_args()

# dataset_file = "./dataset_more_labels.dat"


# Load the dataset
# Cut the max amount of letters in the state to a maximum.
# Better to do it here in the read_csv so we dont use memory later. Here those lines never go into memory.
f = lambda x: x[: args.max_letters]
with open( args.dataset_file, 'rb') as csvfile:
    df = pd.read_csv(
        csvfile,
        delimiter='|',
        names=['note', 'label', 'model_id', 'state'],
        skipinitialspace=True,
        converters={'state': f},
    )

if  args.verbose:
    df.describe()


# Clean the dataset
df.dropna(axis=0, how='any', inplace=True)
df.drop(axis=1, columns=['note', 'model_id'], inplace=True)

# Delete the strings of letters with less than a certain amount
indexNames = df[df['state'].str.len() <  args.min_letters].index
df.drop(indexNames, inplace=True)


# Add a new column to the dataframe with the label. The label is 'Normal' for the normal data and 'Malcious' for the malware data
df.loc[df.label.str.contains('Normal'), 'label'] = 'Normal'
df.loc[df.label.str.contains('Botnet'), 'label'] = 'Malicious'
df.loc[df.label.str.contains('Malware'), 'label'] = 'Malicious'

# Change the labels from Malicious/Normal to 1/0 integers in the df
df.label.replace('Malicious', 1, inplace=True)
df.label.replace('Normal', 0, inplace=True)


# Convert each of the stratosphere letters to an integer. There are 50
vocabulary = list('abcdefghiABCDEFGHIrstuvwxyzRSTUVWXYZ1234567890,.+*')
int_of_letters = {}
for i, letter in enumerate(vocabulary):
    int_of_letters[letter] = float(i)
if args.verbose:
    print(
        f'There are {len(int_of_letters)} letters in total. From letter index {min(int_of_letters.values())} to letter index {max(int_of_letters.values())}.'
    )
vocabulary_size = len(int_of_letters)


# Change the letters in the state to an integer representing it uniquely. We 'encode' them.
df['state'] = df['state'].apply(lambda x: [[int_of_letters[i]] for i in x])
# So far, only 1 feature per letter
features_per_sample = 1



# Split the data in training and testing
train_data, test_data = train_test_split(df, test_size=0.2, shuffle=True)

# CONVERT THE DATA TO NUMPY FORMAT
x_data = train_data['state'].to_numpy()
y_data = train_data['label'].to_numpy()
x_test_data = test_data['state'].to_numpy()
y_test_data = test_data['label'].to_numpy()
# PAD THE DATA
max_length_of_outtupple = max([len(sublist) for sublist in df.state.to_list()])
padded_x_data = pad_sequences(x_data, maxlen=max_length_of_outtupple, padding='post')
padded_x_test_data = pad_sequences(x_test_data, maxlen=max_length_of_outtupple, padding='post')
train_x_data = padded_x_data
train_y_data = y_data
# reshape data
train_x_data = train_x_data.reshape((49, 500))
padded_x_test_data = padded_x_test_data.reshape((13, 500))

def rf_model(train_x_data,  padded_x_test_data, train_y_data, y_test_data):
    # TRAIN THE RANDOM FOREST MODEL
    from sklearn.ensemble import RandomForestClassifier
    model = RandomForestClassifier(n_estimators = 100)
    model.fit(train_x_data, train_y_data)

    from sklearn.metrics import accuracy_score, f1_score

    # Predict using the trained model on the test data
    predicted_labels = model.predict(padded_x_test_data)

    # Calculate accuracy
    accuracy = accuracy_score(y_test_data, predicted_labels)

    # Calculate F1 score
    f1 = f1_score(y_test_data, predicted_labels)

    print("RF measures:")
    print("Accuracy:", accuracy)
    print("F1 Score:", f1)
    # Accuracy: 1.0
    # F1 Score: 1.0


# Accuracy: 0.8461538461538461
# F1 Score: 0.9166666666666666
def svm_model(X_train, X_test, y_train, y_test):
    from sklearn.svm import SVC
    from sklearn.metrics import accuracy_score, f1_score
    from sklearn.model_selection import train_test_split
    from sklearn.preprocessing import StandardScaler
    from sklearn.pipeline import make_pipeline

    # Create a pipeline with StandardScaler and SVM
    model = make_pipeline(StandardScaler(), SVC())

    # Train the SVM model
    model.fit(X_train, y_train)

    # Predict using the trained model on the test data
    predicted_labels = model.predict(X_test)

    # Calculate accuracy
    accuracy = accuracy_score(y_test, predicted_labels)

    # Calculate F1 score
    f1 = f1_score(y_test, predicted_labels)
    print("SVM measures:")
    print("Accuracy:", accuracy)
    print("F1 Score:", f1)


# Accuracy: 0.7692307692307693
# F1 Score: 0.8695652173913043

def knn_model(X_train, X_test, y_train, y_test):
    from sklearn.neighbors import KNeighborsClassifier
    from sklearn.metrics import accuracy_score, f1_score
    from sklearn.model_selection import train_test_split
    from sklearn.preprocessing import StandardScaler
    from sklearn.pipeline import make_pipeline


    # Create a pipeline with StandardScaler and KNN
    model = make_pipeline(StandardScaler(), KNeighborsClassifier(n_neighbors=5))

    # Train the KNN model
    model.fit(X_train, y_train)

    # Predict using the trained model on the test data
    predicted_labels = model.predict(X_test)

    # Calculate accuracy
    accuracy = accuracy_score(y_test, predicted_labels)

    # Calculate F1 score
    f1 = f1_score(y_test, predicted_labels)
    print("KNN Measures:")
    print("Accuracy:", accuracy)
    print("F1 Score:", f1)


# Test Loss: 0.6770883798599243
# Test Accuracy: 0.8461538553237915
def rnn_model(X_train, X_test, y_train, y_test):
    import tensorflow as tf
    from tensorflow.keras import layers

    # Define the vocabulary size (replace this with your actual vocabulary size)
    vocabulary_size = 10000

    # Define your RNN model
    model = tf.keras.models.Sequential([
        layers.Embedding(vocabulary_size, 16, mask_zero=True),
        layers.Bidirectional(layers.GRU(32, return_sequences=False), merge_mode='concat'),
        layers.Dense(32, activation='relu'),
        layers.Dropout(0.5),
        layers.Dense(1, activation='sigmoid')
    ])

    # Compile the model
    model.compile(
        loss='binary_crossentropy',
        optimizer=tf.keras.optimizers.RMSprop(learning_rate=0.0001, momentum=0.05),
        metrics=['accuracy']
    )

    # Train the model
    history = model.fit(
        X_train, y_train,  # Replace X_train and y_train with your training data
        epochs=10,  # Number of epochs
        batch_size=32,  # Batch size
        validation_data=(X_test, y_test)  # Replace X_test and y_test with your testing data
    )

    # Evaluate the model
    test_loss, test_accuracy = model.evaluate(X_test, y_test)  # Replace X_test and y_test with your testing data
    print("RNN Efficiency measures:")
    print("Test Loss:", test_loss)
    print("Test Accuracy:", test_accuracy)


rf_model(train_x_data,  padded_x_test_data, train_y_data, y_test_data)
svm_model(train_x_data,  padded_x_test_data, train_y_data, y_test_data)
knn_model(train_x_data,  padded_x_test_data, train_y_data, y_test_data)
rnn_model(train_x_data,  padded_x_test_data, train_y_data, y_test_data)