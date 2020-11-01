

# Imports
import numpy as np

import pandas as pd

from sklearn.preprocessing import LabelEncoder, OneHotEncoder, StandardScaler
from sklearn.compose import ColumnTransformer
from sklearn.model_selection import train_test_split
from sklearn.metrics import confusion_matrix

import keras
from keras.models import Sequential
from keras.layers import Dense


# Dataset
dataSet = pd.read_csv('/Users/yanndebain/MEGA/MEGAsync/Code/Data Science/Deep Learning/Churn_Modelling.csv')
X = dataSet.iloc[:, 3:-1]
y = dataSet.iloc[:, -1]

# Categorical Variables
columnTransformer = ColumnTransformer([('dummyColumn', OneHotEncoder(), [1])], remainder='passthrough')
X = columnTransformer.fit_transform(X)
X = X[:, 1:]

lbEncoder = LabelEncoder()
X[:, 3] = lbEncoder.fit_transform(X[:, 3])

# Training and test set
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2)

# Feature scaling
stdScaler = StandardScaler()
X_train = stdScaler.fit_transform(X_train)
X_test = stdScaler.transform(X_test)




# Initialization ANN
classifier = Sequential()

# Input layer
classifier.add(Dense(units=6, activation='relu', kernel_initializer='uniform', input_dim=11))

# Other layers
classifier.add(Dense(units=6, activation='relu', kernel_initializer='uniform'))

# Output layer
classifier.add(Dense(units=1, activation='sigmoid', kernel_initializer='uniform'))

# Compilation
classifier.compile(optimizer='adam', loss='binary_crossentropy', metrics=['accuracy'])

# Training
classifier.fit(X_train, y_train, batch_size=10, epochs=100)




# Prediction
y_pred = classifier.predict(X_test)
y_pred = (y_pred > 0.5)

# Confusion Matrix
CM = confusion_matrix(y_test, y_pred)
Accuracy = (CM[0, 0] + CM[1, 1])/len(X_test)


# New client
newClient = np.array([[0.0, 0, 600, 1, 40, 3, 60000, 2, 1, 1, 50000]])
newClient = stdScaler.fit_transform(newClient)

newClient_prediction = classifier.predict(newClient)
#newClient_prediction = (newClient_prediction > 0.5)