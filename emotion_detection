import pandas as pd
import neattext.functions as nfx
import joblib
from sklearn.pipeline import Pipeline
from sklearn.ensemble import VotingClassifier
from sklearn.linear_model import LogisticRegression
from sklearn.svm import LinearSVC
from sklearn.calibration import CalibratedClassifierCV
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.metrics import classification_report, accuracy_score
from xgboost import XGBClassifier


def load_data(path):
    df = pd.read_csv(path, sep=';', header=None, names=['text', 'emotion'])
    df['clean_text'] = df['text'].apply(lambda x: nfx.remove_stopwords(nfx.remove_punctuations(str(x))))
    return df

train_df = load_data("train.txt")
val_df = load_data("val.txt")
test_df = load_data("test.txt")

train_df = pd.concat([train_df, val_df])

X_train = train_df['clean_text']
y_train = train_df['emotion']
X_test = test_df['clean_text']
y_test = test_df['emotion']

logreg = LogisticRegression(max_iter=1000, class_weight='balanced')

svm_base = LinearSVC(class_weight='balanced')
svm = CalibratedClassifierCV(svm_base)

xgb = XGBClassifier(use_label_encoder=False, eval_metric='mlogloss')

ensemble_pipeline = Pipeline([
    ('tfidf', TfidfVectorizer(ngram_range=(1, 2), max_features=15000)),
    ('voting', VotingClassifier(
        estimators=[
            ('lr', logreg),
            ('svm', svm),
            ('xgb', xgb)
        ],
        voting='soft'
    ))
])

ensemble_pipeline.fit(X_train, y_train)

y_pred = ensemble_pipeline.predict(X_test)

print("\n✅ Accuracy:", accuracy_score(y_test, y_pred))
print("\n📊 Classification Report:\n", classification_report(y_test, y_pred))

joblib.dump(ensemble_pipeline, 'final_boss_emotion_model.pkl')
print("\n📦 Model saved as final_boss_emotion_model.pkl")

def predict_emotion(text):
    clean = nfx.remove_stopwords(nfx.remove_punctuations(text))
    return ensemble_pipeline.predict([clean])[0]

if __name__ == "__main__":
    print("\n🎯 Final Boss Emotion Detector — Type 'exit' to quit")
    while True:
        inp = input("\nEnter text: ")
        if inp.lower() == 'exit':
            break
        print("Predicted Emotion:", predict_emotion(inp))
