from sklearn.metrics import classification_report
from sklearn.naive_bayes import MultinomialNB

def run_model_training(x_train, x_test, y_train, y_test):
    clf = MultinomialNB()
    clf.fit(x_train, y_train)
    clf.score(x_test, y_test)
    y_pred = clf.predict(x_test)
    print(classification_report(y_test, y_pred))
    
    return clf