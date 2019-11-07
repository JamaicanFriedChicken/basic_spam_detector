import pandas as pd
from flask import Flask, render_template, url_for, request
from sklearn.feature_extraction.text import CountVectorizer
from sklearn.model_selection import train_test_split
from sklearn.naive_bayes import MultinomialNB

app = Flask(__name__)


@app.route('/')
def home():
    return render_template('home.html')


@app.route('/predict', methods=['POST'])
def predict():
    dataFrame = pd.read_csv("spam.csv", encoding="latin-1")
    dataFrame.drop(['Unnamed: 2', 'Unnamed: 3', 'Unnamed: 4'], axis=1, inplace=True)

    # Features and labels
    dataFrame['label'] = dataFrame['class'].map({'ham': 0, 'spam': 1})
    x = dataFrame['message']
    y = dataFrame['label']

    # Extract features with CountVectorizer
    cv = CountVectorizer()
    x = cv.fit_transform(x) # fitting the data
    x_train, x_test, y_train, y_test = train_test_split(x, y, test_size=0.33, random_state=42)

    # Naive Bayes Classifier
    clf = MultinomialNB()
    clf.fit(x_train, y_train)
    clf.score(x_test, y_test)

    if request.method == 'POST':
        message = request.form['message']
        data = [message]
        vect = cv.transform(data).toarray()
        prediction = clf.predict(vect)
    return render_template('result.html', prediction=prediction)

if  __name__ == '__main__':
    app.run(debug=True)