import pandas as pd
from sklearn.feature_extraction.text import CountVectorizer
from sklearn.model_selection import train_test_split
from sklearn.naive_bayes import MultinomialNB
from sklearn.metrics import classification_report
from sklearn.externals import joblib

dataFrame = pd.read_csv("spam.csv", encoding="latin-1")
dataFrame.drop(['Unnamed: 2', 'Unnamed: 3', 'Unnamed: 4'], axis=1, inplace=True)

# Features and labels
dataFrame['label'] = dataFrame['class'].map({'ham': 0, 'spam': 1})
x = dataFrame['message']
y = dataFrame['label']
cv = CountVectorizer()
x = cv.fit_transform(x) # fitting the data
x_train, x_test, y_train, y_test = train_test_split(x, y, test_size=0.33, random_state=42)

# Naive Bayes Classifier
clf = MultinomialNB()
clf.fit(x_train, y_train)
clf.score(x_test, y_test)
y_predict = clf.predict(x_test)
print(classification_report(y_test, y_predict))

# Saving model
joblib.dump(clf, 'NB_spam_detect_model.pkl')
NB_spam_detect_model = open('NB_spam_detect_model.pkl', 'rb')
clf = joblib.load(NB_spam_detect_model)