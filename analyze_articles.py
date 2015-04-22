import pandas
import os
from scipy.sparse import coo_matrix
from sklearn.ensemble import RandomForestClassifier
from sklearn.feature_extraction.text import CountVectorizer
from sklearn import cross_validation
import settings
import re
import numpy as np
import matplotlib.pyplot as plt


def read_sentiment():
    data = []
    for d in settings.NEG_SENT_DIRS:
        for f in os.listdir(d):
            with open(os.path.join(d, f)) as negfile:
                data.append([negfile.read(), 0])

    for d in settings.POS_SENT_DIRS:
        for f in os.listdir(d):
            with open(os.path.join(d, f)) as posfile:
                data.append([posfile.read(), 1])

    for d in data:
        d[0] = preprocess_text(d[0])
    return pandas.DataFrame(data, columns=["text", "score"])

def preprocess_text(text):
    text = text.lower()
    text = re.sub("[^0-9a-zA-Z ]+", " ", text)
    text = re.sub("\s+", " ", text)
    return text

def find_term_sentiment(term, data, vectorizer, clf):
    selected = data[data["term"] == term]

    selected = selected.sort(['pub_date'], ascending=[1])

    articles = selected["headline__main"].apply(preprocess_text)
    features = vectorizer.transform(articles)
    selected["sentiment"] = clf.predict_proba(features)[:,1]
    return selected

def multiply_afinn_rows(row):
    return sum(row * afinn["score"]) / len(row[row!=0])

def find_term_afinn_sentiment(term, data, *args):
    vectorizer = CountVectorizer(ngram_range=(1, 1), vocabulary=afinn["word"])
    selected = data[data["term"] == term]
    selected = selected.sort(['pub_date'], ascending=[1])
    articles = selected["snippet"].apply(preprocess_text)
    count_mats = vectorizer.fit_transform(articles)
    scores = []
    for row_ind in range(0, count_mats.shape[0]):
        row = np.asarray(count_mats[row_ind, :].todense())[0]
        score = multiply_afinn_rows(row)
        scores.append(score)
    selected["sentiment"] = scores
    return selected

def running_mean(x, N):
    return np.convolve(x, np.ones((N,))/N)[(N-1):]

def get_afinn_frame():
    with open(settings.AFINN_WORDLIST) as f:
        data = pandas.read_csv(f, sep="\t")
    data.columns = ["word", "score"]
    return data

def generate_review_frame(sent_frame):
    words = {}
    for i, row in sent_frame.iterrows():
        for w in row["text"].split(" "):
            if w not in words:
                words[w] = {"score": 0, "count": 0}
            words[w]["count"] += 1
            words[w]["score"] += row["score"]
    frame = []
    for w in words:
        frame.append([w, words[w]["score"]/words["w"]["count"]])
    return pandas.DataFrame(frame, columns=["word", "score"])

def score_text(text):
    score = 0
    for r in afinn.iterrows():
        w = r[0]
        s = r[1]
        if w in text.split(" "):
            score += int(s)
    return score

def search_for_term(term, field, data):
    found = []
    for i in data[field]:
        if term in i:
            found.append(i)
    return found

sent_frame = read_sentiment()

vectorizer = CountVectorizer(stop_words='english', ngram_range=(1, 2))
train_features = vectorizer.fit_transform(sent_frame["text"])

clf = RandomForestClassifier(n_estimators=150, min_samples_leaf=15, n_jobs=3)

scores = cross_validation.cross_val_score(clf, train_features, sent_frame["score"], cv=3)
print("Accuracy: %0.2f (+/- %0.2f)" % (scores.mean(), scores.std() * 2))


clf.fit(train_features, sent_frame["score"])

afinn = generate_review_frame(sent_frame)

with open("data/articles.csv", 'r') as f:
    data = pandas.read_csv(f)

data.dropna(subset=["snippet", "pub_date", "headline__main"], inplace=True)
autopilot = find_term_afinn_sentiment("computer", data, vectorizer, clf)

smoothed_sent = running_mean(autopilot["sentiment"], 30)

plt.plot(smoothed_sent)
print("Done")

plt.show()