from sklearn.linear_model import LogisticRegression
from sklearn.model_selection import train_test_split

def train_model(X, y):
    # split the data into training and testing sets
    X_train, X_test, y_train, y_test = train_test_split(
        X, y, test_size=0.2, random_state=42
    )
    # train a lofistic regression model
    model = LogisticRegression(max_iter=1000)
    model.fit(X_train, y_train)
    
    return model, X_test, y_test