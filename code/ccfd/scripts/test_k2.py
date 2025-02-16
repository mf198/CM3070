import numpy as np
import pandas as pd
import seaborn as sns
import matplotlib.pyplot as plt


df = pd.read_csv("ccfd/data/creditcard.csv")
df.head(5)

df = df.drop(columns = ["Time"])

from sklearn.preprocessing import StandardScaler
df["Amount"] = StandardScaler().fit_transform(df["Amount"].values.reshape(-1,1))

df.head(5)

X = df.drop(columns = ["Class"])
y = df["Class"]

x = y.value_counts().sort_index(ascending=False)
print(x)
plt.bar(["1", "0"], list(x.values), color =["b", "r"],width = 0.4)


import time
from sklearn.model_selection import train_test_split
from sklearn.metrics import confusion_matrix
from sklearn.neighbors import KNeighborsClassifier as knn_normal
X_train, X_test, y_train, y_test = train_test_split(X.values, y.values.reshape(-1,1), test_size = 0.1)


frauds = df[df["Class"] == 1]
print(X_train.dtype)
X_train = X_train.astype("float32")
X_test = X_test.astype("float32")
frauds = frauds.astype("float32")
print(X_train.dtype)

t0 = time.time()
knn = knn_normal(n_neighbors = 5)
knn.fit(X_train, y_train)
print(confusion_matrix(y_test, knn.predict(X_test)))
time.time() - t0

import cuml
t0 = time.time()
from cuml.neighbors import KNeighborsClassifier
knn = KNeighborsClassifier(n_neighbors=5)
knn.fit(X_train, y_train)
cf = confusion_matrix(y_test, knn.predict(X_test))
print(cf)
time.time() - t0

from cuml.ensemble import RandomForestClassifier
from cuml.linear_model import LogisticRegression
from cuml.neighbors import KNeighborsClassifier
from cuml.svm import SVC


from sklearn.model_selection import train_test_split
from sklearn.model_selection import RandomizedSearchCV, GridSearchCV

f1_scores_fraud = {"rf" : "", "lr" : "", "knn": "","svc": ""}
f1_scores_non_fraud = {"rf" : "", "lr" : "", "knn": "","svc": ""}
fraud_only_acc = {"rf" : "", "lr" : "", "knn": "", "svc": ""}
from sklearn.metrics import accuracy_score
def train_data(X_train,X_test,y_train,y_test):
    lr = LogisticRegression()
    rf = RandomForestClassifier(max_features=1.0,
                   n_bins=8,
                   n_estimators=40)
    knn = KNeighborsClassifier(n_neighbors=10)
    svc = SVC(kernel='poly', degree=2, gamma='auto', C=1)
    
    algorithms = {"lr" : lr, "knn" : knn, "rf": rf, "svc": svc}
    
    for key, a in algorithms.items():
        print(key, "trained")
        a.fit(X_train, y_train)
    
    for key, a in algorithms.items():
        f1_scores_non_fraud[key] = float(classification_report(y_test, a.predict(X_test)).split()[7])
        f1_scores_fraud[key] = float(classification_report(y_test, a.predict(X_test)).split()[12])
        fraud_only_acc[key] = accuracy_score(frauds["Class"], a.predict(frauds.drop(columns = ["Class"])))

f1_fraud = {}
f1_non_fraud = {}
fraud_acc = {}

import time
def train(X_train, X_test, y_train, y_test):
    t0 = time.time()
    train_data(X_train, X_test, y_train, y_test)
    print("Runtime(s) : ", time.time() - t0)
train(X_train, X_test, y_train, y_test)

def plot_f1_scores(tag):
    labels = f1_scores.keys()
    for x in f1_scores.values():
        non_fraud.append(x[0])
        fraud.append(x[1])
    x = np.arange(len(labels))  # the label locations
    width = 0.35  # the width of the bars
    fig, ax = plt.subplots()
    rects1 = ax.bar(x - width/2, f1_scores_non_fraud.values(), width, label='Non Fraud')
    rects2 = ax.bar(x + width/2, f1_scores_fraud.values(), width, label='Fraud')
    # Add some text for labels, title and custom x-axis tick labels, etc.
    ax.set_ylabel('Scores')
    ax.set_title('F1 Scores')
    ax.set_xticks(x)
    ax.set_xticklabels(labels)
    ax.legend()
    ax.bar_label(rects1, padding=3)
    ax.bar_label(rects2, padding=3)
    fig.tight_layout()
    plt.show()
    for key, val in f1_scores_fraud.items():
        f1_fraud[key+ tag] = val
    for key, val in f1_scores_non_fraud.items():
        f1_non_fraud[key + tag] = val
plot_f1_scores("_n")

def plot_fraud_only(tag):
    plt.bar(fraud_only_acc.keys(), fraud_only_acc.values(), color =["b", "r", "c", "m", "y"],width = 0.4)
    print(fraud_only_acc)
    for key,value in fraud_only_acc.items():
        fraud_acc[key + tag] = value
plot_fraud_only("_n")