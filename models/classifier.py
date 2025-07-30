import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.svm import LinearSVC
from sklearn.metrics import classification_report
from sklearn.preprocessing import LabelEncoder

class JobClassifier:
    def __init__(self, max_features=5000):
        self.vectorizer = TfidfVectorizer(max_features=max_features)
        self.model = LinearSVC()
        self.label_encoder = LabelEncoder()

    def load_and_train(self, filepath, test_size=0.2):
        df = pd.read_csv(filepath)[['title', 'skillset', 'category']].dropna()
        df['text'] = df['title'].astype(str) + ' ' + df['skillset'].astype(str)

        y = self.label_encoder.fit_transform(df['category'])
        X_train, X_test, y_train, y_test = train_test_split(df['text'], y, test_size=test_size, random_state=42)

        self.vectorizer.fit(X_train)
        X_train_vec = self.vectorizer.transform(X_train)
        X_test_vec = self.vectorizer.transform(X_test)

        self.model.fit(X_train_vec, y_train)
        preds = self.model.predict(X_test_vec)
        report = classification_report(y_test, preds, output_dict=True)

        return {
            'f1_macro': report['macro avg']['f1-score'],
            'accuracy': report['accuracy']
        }

    def predict(self, title, skillset):
        text = title + " " + skillset
        vec = self.vectorizer.transform([text])
        pred_id = int(self.model.predict(vec)[0])  # ensure it's a Python int

        return {
            'job_category': pred_id
        }

# Training script - only run when needed
def train_and_save_model():
    """Train and save the model - run this separately"""
    import pickle
    
    clf = JobClassifier()
    
    # Check if training data exists
    import os
    if not os.path.exists("rule_based_labeled_jobs.csv"):
        print("❌ Training data file 'rule_based_labeled_jobs.csv' not found!")
        return False
    
    print("🔄 Training classifier...")
    try:
        results = clf.load_and_train("rule_based_labeled_jobs.csv")
        print(f"✅ Training completed - F1: {results['f1_macro']:.3f}, Accuracy: {results['accuracy']:.3f}")
        
        with open("job_classifier_model.pkl", "wb") as f:
            pickle.dump(clf, f)
        print("✅ Model saved to job_classifier_model.pkl")
        return True
        
    except Exception as e:
        print(f"❌ Training failed: {e}")
        return False

if __name__ == "__main__":
    train_and_save_model()

