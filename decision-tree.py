from sklearn import tree, ensemble

# Let's suppose we have a dataset of four fruits
# For each fruit, we know its weight and texture

# This is our raw data. We need to get rid of the words in the features
features = [[320, "smooth"], [340, "smooth"], [25, "bumpy"], [30, "bumpy"]]
labels = ["watermelon", "watermelon", "orange", "orange"]

# This is our training data
features = [[320, 0], [340, 0], [25, 1], [30, 1]]

# Get the decision tree classifier from the scikit library we imported
classifier = tree.DecisionTreeClassifier()

# Train your decision tree classifier with our training data
trained_classifier = classifier.fit(features, labels)

# trained_classifier now has learned that objects that are heavy and smooth
# are watermelons, and objects that are light and bumpy are oranges!

# Make a prediction!
prediction = trained_classifier.predict([[100, 1]])
print(prediction)

# Yay, we made our first machine learning program!

# What would it look like with biological data? Let's pretend we have a tumor dataset.

# So how can we improve the accuracy of the classifier?
# 1. We can feed it more training data
# 2. We can also try a smarter classifier than a Decision Tree. 
# Luckily for us, scikit-learn has a ton of amazing classifiers
# classifier = ensemble.RandomForestClassifier()

# We're going to turn it over to you now and give you
# some time to play around with your first model before we bring
# it in for questions and start Exercise 2.