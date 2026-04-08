from sklearn.metrics import accuracy_score, classification_report

def evaluate_model(model, X_test, y_test):
    y_pred = model.predict(X_test)
    # measure accuracy of the model
    acc = accuracy_score(y_test, y_pred)
    return acc