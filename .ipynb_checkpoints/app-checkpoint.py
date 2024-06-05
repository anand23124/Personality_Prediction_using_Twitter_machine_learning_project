from flask import Flask, request, render_template
from ntscraper import Nitter
import csv
import pandas as pd
import numpy as np
import pickle
from sklearn.feature_extraction.text import TfidfVectorizer
from collections import Counter

app = Flask(__name__)
scraper = Nitter()

# Load models
model_IE = pickle.load(open("BNIEFinal.sav", 'rb'))
model_SN = pickle.load(open("BNSNFinal.sav", 'rb'))
model_TF = pickle.load(open('BNTFFinal.sav', 'rb'))
model_PJ = pickle.load(open('BNPJFinal.sav', 'rb'))

# Load vocabulary
with open('newfrequency300.csv','rt') as f:
    csvReader = csv.reader(f)
    mydict = {rows[1]: int(rows[0]) for rows in csvReader}

def get_tweets(name):
    tweetList = []
    tweets = scraper.get_tweets(name, mode="user", number=100)
    for tweet in tweets['tweets']:
        data = tweet['text']
        tweetList.append(data)
    return tweetList

def predict_personality(tweetList):
    if len(tweetList) > 0:
        vectorizer = TfidfVectorizer(vocabulary=mydict, min_df=1)
        x = vectorizer.fit_transform(tweetList).toarray()
        df = pd.DataFrame(x)
    else:
        return "No tweets found. Check your data loading process."

    IE = model_IE.predict(df)
    SN = model_SN.predict(df)
    TF = model_TF.predict(df)
    PJ = model_PJ.predict(df)

    answer = []
    b = Counter(IE)
    value = b.most_common(1)
    if value[0][0] == 1.0:
        answer.append("I")
    else:
        answer.append("E")

    b = Counter(SN)
    value = b.most_common(1)
    if value[0][0] == 1.0:
        answer.append("S")
    else:
        answer.append("N")

    b = Counter(TF)
    value = b.most_common(1)
    if value[0][0] == 1:
        answer.append("T")
    else:
        answer.append("F")

    b = Counter(PJ)
    value = b.most_common(1)
    if value[0][0] == 1:
        answer.append("P")
    else:
        answer.append("J")

    return "".join(answer)

@app.route('/', methods=['GET', 'POST'])
def index():
    if request.method == 'POST':
        username = request.form['username']
        tweetList = get_tweets(username)
        mbti = predict_personality(tweetList)
        return render_template('result.html', username=username, mbti=mbti)
    return render_template('index.html')

if __name__ == '__main__':
    app.run(debug=True)