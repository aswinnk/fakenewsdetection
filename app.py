import numpy as np
import pandas as pd
import itertools
from sklearn.model_selection import train_test_split
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.linear_model import PassiveAggressiveClassifier
from sklearn.metrics import accuracy_score, confusion_matrix
from flask import Flask,redirect,render_template,request

app = Flask(__name__)

@app.route('/')
def home():
    return render_template("index.html")

@app.route('/fakedetection',methods=['POST'])
def fakedetection():
    #Read the data
    df=pd.read_csv('D:\\fakenewsdetection\\news.csv')

    #Get shape and head
    df.shape
    df.head()


    #DataFlair - Get the labels
    labels=df.label
    labels.head()

    #DataFlair - Split the dataset
    x_train,x_test,y_train,y_test=train_test_split(df['text'], labels, test_size=0.2, random_state=7)
    #DataFlair - Initialize a TfidfVectorizer
    tfidf_vectorizer=TfidfVectorizer(stop_words='english', max_df=0.7)

    #DataFlair - Fit and transform train set, transform test set
    tfidf_train=tfidf_vectorizer.fit_transform(x_train) 
    tfidf_test=tfidf_vectorizer.transform(x_test)

    #DataFlair - Initialize a PassiveAggressiveClassifier
    pac=PassiveAggressiveClassifier(max_iter=50)
    pac.fit(tfidf_train,y_train)

    #DataFlair - Predict on the test set and calculate accuracy
    y_pred=pac.predict(tfidf_test)
    score=accuracy_score(y_test,y_pred)
    confusion_matrix(y_test,y_pred, labels=['FAKE','REAL'])
    print(f'Accuracy: {round(score*100,2)}%')
    if request.method == "POST":
        message=request.form['message']
        data=[message]
        vect=tfidf_vectorizer.transform(data).toarray()
        my_prediction=pac.predict(vect)
    print(my_prediction)
        

    #DataFlair - Build confusion matrix
    
    print(confusion_matrix)
    return render_template("index.html",prediction=my_prediction)

if __name__ == '__main__':
    app.run(debug=True) 