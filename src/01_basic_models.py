from dev.preprocessing import clean
import pandas as pd
import re

from sklearn.feature_extraction.text import TfidfVectorizer
from nltk.corpus import stopwords
from nltk.stem import WordNetLemmatizer
from sklearn.metrics import f1_score
from sklearn.model_selection import GridSearchCV

STOPWORDS = stopwords.words('english')


def get_vocabulary(iter_sents):
    vocab = set()
    max_length = 0
    for text in iter_sents:
        tokens = text.split(' ')
        if len(tokens) > max_length:
            max_length = len(tokens)
        vocab.update(tokens)
    return vocab, max_length


# Read & split train test
filename = './data/entire_corpus.csv'
df = pd.read_csv(filename)
df_train = df[df['source'] == 'train'].copy()
df_test = df[df['source'] == 'test'].copy()

# Clean
df_train['text'] = df_train['text'].apply(clean)
df_test['text'] = df_test['text'].apply(clean)

# Remove digits
df_train['text'] = df_train['text'].apply(lambda x: re.sub(r'\w*\d\w*', '', x).strip())
df_test['text'] = df_test['text'].apply(lambda x: re.sub(r'\w*\d\w*', '', x).strip())

# Remove stop words
df_train['text'] = df_train['text'].apply(lambda x: ' '.join([w for w in x.split(' ') if w not in STOPWORDS]))
df_test['text'] = df_test['text'].apply(lambda x: ' '.join([w for w in x.split(' ') if w not in STOPWORDS]))

# Lemmatizing
lemm = WordNetLemmatizer()


def lemmatize(text):
    words = text.split(' ')
    words = [lemm.lemmatize(w) for w in words]
    words = [lemm.lemmatize(w, pos='a') for w in words]
    words = [lemm.lemmatize(w, pos='v') for w in words]
    return ' '.join(words)


df_train['text'] = df_train['text'].apply(lemmatize)
df_test['text'] = df_test['text'].apply(lemmatize)

# Vectorize
tfidf = TfidfVectorizer()
data_matrix = tfidf.fit_transform(df_train.text)

X_train = data_matrix.toarray()
X_test = tfidf.transform(df_test.text).toarray()
y_train = df_train['target'].values
y_test = df_test['target'].values


def performance(model, print_score=True):
    model.fit(X_train, y_train)
    preds = model.predict(X_test)
    score = f1_score(preds, y_test)
    print(score) if print_score else 0
    return score


# Models

## Logistic regression ~0.74
from sklearn.linear_model import LogisticRegression

logreg_c = [10, 1, 0.1, 0.01, 0.001, 0.0001]
logreg_scores = []
for c in logreg_c:
    logreg = LogisticRegression(C=c)
    logreg_scores.append(performance(logreg))

## SVM ~0.73
from sklearn.svm import SVC

svc = SVC()
performance(svc)

## MLP : ~0.70
from sklearn.neural_network import MLPClassifier

mlp = MLPClassifier(hidden_layer_sizes=100, activation='relu')
performance(mlp)

parameters = {'C': [10, 1, 0.1, 0.01, 0.001]}
logreg = LogisticRegression()
model = GridSearchCV(logreg, parameters)
model.fit(X_train, y_train)



