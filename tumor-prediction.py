import pandas
import numpy
from sklearn import ensemble
from sklearn.model_selection import train_test_split

# Read in the breast cancer data.csv
data = pandas.read_csv("data.csv", header=0)

# Take a look at pandas dataframe format
# print(data.head())

# Data cleaning
mapping = {'M' : 0, 'B' : 1}
data['diagnosis'] = data['diagnosis'].map(mapping)

features = list(data.columns[1:31]) # Appending all the columns in feature vector
train_features, test_features, train_labels, test_labels = train_test_split(data[features], data['diagnosis'].values, test_size=0.20, random_state=10)

# Get the random forest classifier from the scikit library we imported
classifier = ensemble.RandomForestClassifier()

# Train your classifier with our training data split
trained_classifier = classifier.fit(train_features.values, train_labels)

# Let's try out our trained classifier
y_prediction = trained_classifier.predict(test_features.values)

# Print out the predictions vs the actual values
print(y_prediction)
print(test_labels)

num_correct_predictions = numpy.sum(y_prediction == test_labels)
num_test_samples = float(len(test_labels))

print ("ML Accuracy", num_correct_predictions / num_test_samples)