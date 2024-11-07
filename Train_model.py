import pickle
from sklearn.ensemble import RandomForestClassifier
from sklearn.model_selection import train_test_split
from sklearn.metrics import accuracy_score
import numpy as np

# Load the preprocessed data and labels from the pickle file
data_dict = pickle.load(open('./data_new.pickle', 'rb'))

# Convert the data and labels into NumPy arrays for compatibility with scikit-learn
data = np.asarray(data_dict['data'])
labels = np.asarray(data_dict['labels'])

# Split the data into training and testing sets
# 80% of the data is used for training, 20% for testing
# The data is shuffled and stratified to maintain the proportion of classes
x_train, x_test, y_train, y_test = train_test_split(data, labels, test_size=0.2, shuffle=True, stratify=labels)

# Initialize a RandomForestClassifier model
model = RandomForestClassifier()

# Train the model on the training data
model.fit(x_train, y_train)

# Use the trained model to predict the labels for the test data
y_predict = model.predict(x_test)

# Calculate the accuracy of the model by comparing predicted labels to actual labels
score = accuracy_score(y_predict, y_test)

# Print the accuracy of the model as a percentage
print('{}% of samples were classified correctly !'.format(score * 100))

# Save the trained model into a pickle file for later use
with open('model_new.p', 'wb') as f:
    pickle.dump({'model': model}, f)
