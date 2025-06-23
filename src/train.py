import joblib
from sklearn.linear_model import LogisticRegression
from sklearn.naive_bayes import MultinomialNB
from sklearn.svm import LinearSVC
from sklearn.metrics import classification_report,accuracy_score
from preprocess import prepare_data

def train_and_evaluate(X_train, X_test, y_train, y_test):
    models = {
        "Logistic Regression": LogisticRegression(max_iter=1000),
        "Naive Bayes": MultinomialNB(),
        "SVM": LinearSVC()
    }
    best_model = None
    best_score = 0
    best_name = ""
    for name, model in models.items():
        print("Training:", name)
        model.fit(X_train, y_train)
        prediction = model.predict(X_test)
        acc = accuracy_score(y_test, prediction)
        print("Accuracy:", acc)
        print(classification_report(y_test, prediction))
        if acc > best_score:
            best_score = acc
            best_model = model
            best_name = name
    print("Best model:", best_name)
    print("Best score:", best_score)
    return best_model
def save_model(model, vectorizer, model_path="models/best_model.pkl", vec_path="models/vectorizer.pkl"):
    joblib.dump(model, model_path)
    joblib.dump(vectorizer, vec_path)
    print("Model and vectorizer saved")
def main():
    X_train, X_test, y_train, y_test, vectorizer = prepare_data("data/spam.csv")
    best_model = train_and_evaluate(X_train, X_test, y_train, y_test)
    save_model(best_model, vectorizer)
if __name__ == "__main__":
    main()