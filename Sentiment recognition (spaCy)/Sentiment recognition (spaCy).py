import pandas as pd
import numpy as np
import random
import spacy
import matplotlib.pyplot as plt
import seaborn as sns

from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.model_selection import train_test_split
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import confusion_matrix, accuracy_score, classification_report
from textblob import TextBlob

# === 1. Зчитування даних ===
data = pd.read_csv('Sentiment_Stock_data2.csv', encoding='utf-8')
labels = data['Sentiment'].values
features = data['Sentence'].values

# Очистка: замінюємо NaN на пусті рядки і перетворюємо в str
features = pd.Series(features).fillna('').astype(str).values

# Попередня обробка з spaCy (open source lib btw)
nlp = spacy.load("en_core_web_sm") #web - навчена на інфі з інтернету, sm - зменшена, але швидка

def spacy_preprocess(texts):
    clean_texts = []
    for doc in nlp.pipe(texts, disable=["ner", "parser"]): #parser - виявляє синтаксичні зв'язки (як пов'язані слова за частинами мови), ner - визначення іменованних сутностей (люди, назви організацій і т.д.)
        tokens = [token.lemma_.lower() for token in doc 
                  if not token.is_stop and not token.is_punct and not token.is_space]
        clean_texts.append(" ".join(tokens))
    return clean_texts

clean = spacy_preprocess(features)

# === 3. Векторизація ===
vect = TfidfVectorizer(min_df=50, max_df=0.8)
td = vect.fit_transform(clean).toarray()

# === 4. Розділення на навчальну і тестову вибірки ===
X_train, X_test, y_train, y_test = train_test_split(td, labels, test_size=0.2, random_state=0)

# === 5. Класифікація: логістична регресія ===
lr = LogisticRegression(max_iter=1000, random_state=0)
lr.fit(X_train, y_train)
prediction = lr.predict(X_test)

# === 6. Оцінка логістичної регресії ===
print("\n=== Logistic Regression Results ===")
cm_lr = confusion_matrix(y_test, prediction)
print("\nConfusion matrix:")
print(cm_lr)
print("Accuracy:", accuracy_score(y_test, prediction))
print("\nClassification Report:\n", classification_report(y_test, prediction))

# === 7. Аналіз з TextBlob ===
tb_preds = [0 if TextBlob(text).sentiment.polarity <= 0 else 1 for text in clean]

print("=== TextBlob Results ===")
cm_tb = confusion_matrix(labels, tb_preds)
print("\nConfusion matrix:")
print(cm_tb)
print("Accuracy:", accuracy_score(labels, tb_preds))

# === 8. Випадкові приклади ===
print("\n=== Random Examples ===")
random_indices = random.sample(range(len(y_test)), 3)
for idx in random_indices:
    print("\nOriginal sentence:\n", features[idx])
    print("True label: ", y_test[idx])
    print("Logistic Regression prediction: ", prediction[idx])
    print("TextBlob prediction: ", tb_preds[idx])

# === 9. Порівняння точності ===
acc_lr = accuracy_score(y_test, prediction)
acc_tb = accuracy_score(labels, tb_preds)

plt.figure(figsize=(6, 4))
sns.barplot(x=['Logistic Regression', 'TextBlob'], y=[acc_lr, acc_tb], palette='pastel')
plt.ylabel('Accuracy')
plt.title('Accuracy Comparison')
plt.ylim(0, 1)
plt.show()
