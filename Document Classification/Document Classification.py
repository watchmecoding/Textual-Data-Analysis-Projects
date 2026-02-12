import pandas as pd
import csv
import nltk
import re
import numpy as np

csvfile = open ("ecommerceDataset3.csv", encoding ='utf-8', mode = "r", newline='')
reader = csv.DictReader(csvfile)

corpus = []
category = []
ids = []
for row in reader:
    ids.append(row[''])
    category.append(row['category'])
    corpus.append(row['text'])

data_df = pd.DataFrame({'': ids, 'category': category, 'text': corpus})
data_df = data_df[:10000]

# Імпортуємо лематизатор
from nltk.stem import WordNetLemmatizer
wpt = nltk.WordPunctTokenizer()
stop_words = nltk.corpus.stopwords.words('english')
lemmatizer = WordNetLemmatizer()  # Ініціалізація лематизатора

# Функція для перетворення POS тегів на формат, який використовує WordNetLemmatizer
def get_wordnet_pos(treebank_tag):
    if treebank_tag.startswith('J'):
        return 'a'  # adjective
    elif treebank_tag.startswith('V'):
        return 'v'  # verb
    elif treebank_tag.startswith('N'):
        return 'n'  # noun
    elif treebank_tag.startswith('R'):
        return 'r'  # adverb
    else:
        return 'n'

p_corpus = []

for doc in corpus:
    doc = doc.lower()
    doc = re.sub(r'[^a-zA-Z0-9]', ' ', doc)

    # Токенізація, видалення стоп-слів та POS-тегування
    tokens = wpt.tokenize(doc)
    tokens = [token for token in tokens if token not in stop_words]
    pos_tags = nltk.pos_tag(tokens)

    # Лематизація з врахуванням частин мови
    lemmatized_tokens = []
    for token, tag in pos_tags:
        wordnet_pos = get_wordnet_pos(tag)  # Перетворення на формат для WordNet
        lemmatized_tokens.append(lemmatizer.lemmatize(token, wordnet_pos))

    doc = ' '.join(lemmatized_tokens)
    p_corpus.append(doc)

data_df['Clean Comment'] = p_corpus
print(data_df)

# Очищення даних
print("\n\nSearch & delete null rows\n")
data_df = data_df.replace(r'^(\s?)+$', np.nan, regex=True)
data_df = data_df.dropna().reset_index(drop=True)
data_df.info()

# Розбиття на train і test
from sklearn.model_selection import train_test_split
train_corpus, test_corpus, train_category, test_category = train_test_split(np.array(data_df['Clean Comment']),
   np.array(data_df['category']), test_size=0.3, random_state=0)
print("\n\nTrain corpus:   {}".format(train_corpus.shape))
print("Test corpus:   {}".format(test_corpus.shape))

# Токенізація
tokenized_train = [wpt.tokenize(text) for text in train_corpus]
tokenized_test = [wpt.tokenize(text) for text in test_corpus]

# Створення Word2Vec моделі
from gensim.models import word2vec
w2v_model  = word2vec.Word2Vec(tokenized_train, vector_size=100, window=40, min_count=2,
    sample=1e-3, sg=1)

# Векторизація документів
def document_vectorizer(corpus, model, num_features):
    vocabulary = set(model.wv.index_to_key)  # всі слова в моделі
    def average_word_vectors(words, model, vocabulary, num_features):
        feature_vector = np.zeros((num_features,), dtype="float64")
        nwords = 0.
        for word in words:
            if word in vocabulary:
                nwords = nwords + 1.
                feature_vector = np.add(feature_vector, model.wv[word])
        if nwords:
            feature_vector = np.divide(feature_vector, nwords)
        return feature_vector
    features = [average_word_vectors(tokenized_sentence, model,
                vocabulary, num_features) for tokenized_sentence in corpus]
    return np.array(features)

avg_wv_train_features = document_vectorizer(corpus=tokenized_train,
    model=w2v_model, num_features=100)
avg_wv_test_features = document_vectorizer(corpus=tokenized_test, model=w2v_model,
    num_features=100)
print('Word2Vec model:> Train features shape:', avg_wv_train_features.shape,' Test features shape:', avg_wv_test_features.shape, '\n')

# Логістична регресія
from sklearn.model_selection import cross_val_score
from sklearn.linear_model import LogisticRegression
lr = LogisticRegression() #penalty='l2', max_iter=2000, C=1, random_state=0

lr.fit(avg_wv_train_features, train_category)

lr_bow_w2v_scores = cross_val_score(lr, avg_wv_train_features, train_category, cv=5)
print('LogReg w2v Accuracy (5-fold):', lr_bow_w2v_scores)

lr_bow_w2v_mean_score = np.mean(lr_bow_w2v_scores)
print('LogReg Mean w2v Accuracy:', lr_bow_w2v_mean_score)

lr_bow_test_score = lr.score(avg_wv_test_features, test_category)
print('LogReg W2V Test Accuracy:', lr_bow_test_score)

# Випадковий ліс
from sklearn.ensemble import RandomForestClassifier
rfc = RandomForestClassifier(n_estimators=10, random_state=0)

rfc.fit(avg_wv_train_features, train_category)

rfc_bow_w2v_scores = cross_val_score(rfc, avg_wv_train_features, train_category, cv=5)
print('RF w2v Accuracy (5-fold):', rfc_bow_w2v_scores)

rfc_bow_w2v_mean_score = np.mean(rfc_bow_w2v_scores)
print('RF Mean w2v Accuracy:', rfc_bow_w2v_mean_score)

rfc_bow_test_score = rfc.score(avg_wv_test_features, test_category)
print('RF W2V Test Accuracy:', rfc_bow_test_score)

from sklearn.feature_extraction.text import TfidfVectorizer

# TF-IDF Векторизація
vectorizer = TfidfVectorizer(use_idf=True, min_df=0.0, max_df=1.0) #, max_features=5000
train_tfidf_features = vectorizer.fit_transform(train_corpus)
test_tfidf_features = vectorizer.transform(test_corpus)
print('TF-IDF model:> Train features shape:', train_tfidf_features.shape,' Test features shape:', test_tfidf_features.shape, '\n')


# Логістична регресія (TF-IDF) - алгоритм максимізує ймовірності прогнозованих значень до спостережуваних значень за допомогою оцінки максимальної ймовірності
#penalty='l2', max_iter=2000, C=1, random_state=0               #C - сила регуляції (0.1-10)
lr_tfidf = LogisticRegression()                                 #l2 - зменшує вагу значущих слів, не обнуляючи (як l1) для запобіганню перенавчання
lr_tfidf.fit(train_tfidf_features, train_category)          
lr_tfidf_scores = cross_val_score(lr_tfidf, train_tfidf_features, train_category, cv=5)
print('LogReg TF-IDF Accuracy (5-fold):', lr_tfidf_scores)
lr_tfidf_mean_score = np.mean(lr_tfidf_scores)
print('LogReg TF-IDF Mean Accuracy:', lr_tfidf_mean_score)
lr_tfidf_test_score = lr_tfidf.score(test_tfidf_features, test_category)
print('LogReg TF-IDF Test Accuracy:', lr_tfidf_test_score)

# Випадковий ліс (TF-IDF)
rfc_tfidf = RandomForestClassifier(n_estimators=10, random_state=0) #к-ть дерев в лісі - 10, і коеф. випадковості
rfc_tfidf.fit(train_tfidf_features, train_category)
rfc_tfidf_scores = cross_val_score(rfc_tfidf, train_tfidf_features, train_category, cv=5)
print('RF TF-IDF Accuracy (5-fold):', rfc_tfidf_scores)
rfc_tfidf_mean_score = np.mean(rfc_tfidf_scores)
print('RF TF-IDF Mean Accuracy:', rfc_tfidf_mean_score)
rfc_tfidf_test_score = rfc_tfidf.score(test_tfidf_features, test_category)
print('RF TF-IDF Test Accuracy:', rfc_tfidf_test_score)

# GridSearchCV для налаштування гіперпараметрів
from sklearn.model_selection import GridSearchCV
grid = {"C": np.logspace(-3, 3, 7), "penalty": ["l2"]}
logreg = LogisticRegression() #max_iter=2000

# GridSearchCV для логістичної регресії (Word2Vec)
logreg_cv = GridSearchCV(logreg, grid, cv=5) #Адаптивно покращує гіперпараметри методів
logreg_cv.fit(avg_wv_train_features, train_category)
print("GS LogReg W2V Tuned hyperparameters (best parameters): ", logreg_cv.best_params_)
print("GS LogReg W2V Accuracy: ", logreg_cv.best_score_)

# GridSearchCV для логістичної регресії (TF-IDF)
logreg_cv_tfidf = GridSearchCV(logreg, grid, cv=5)
logreg_cv_tfidf.fit(train_tfidf_features, train_category)
print("GS LogReg TF-IDF Tuned hyperparameters (best parameters): ", logreg_cv_tfidf.best_params_)
print("GS LogReg TF-IDF Accuracy: ", logreg_cv_tfidf.best_score_)

# GridSearchCV для випадкових лісів (Word2Vec)
rfc_param_grid = {'n_estimators': [10, 50, 100], 'max_depth': [None, 10, 20]}
rfc_cv_w2v = GridSearchCV(RandomForestClassifier(random_state=0), rfc_param_grid, cv=5)
rfc_cv_w2v.fit(avg_wv_train_features, train_category)
print("GS RF W2V Tuned hyperparameters (best parameters): ", rfc_cv_w2v.best_params_)
print("GS RF W2V Accuracy: ", rfc_cv_w2v.best_score_)

# GridSearchCV для випадкових лісів (TF-IDF)
rfc_cv_tfidf = GridSearchCV(RandomForestClassifier(random_state=0), rfc_param_grid, cv=5)
rfc_cv_tfidf.fit(train_tfidf_features, train_category)
print("GS RF TF-IDF Tuned hyperparameters (best parameters): ", rfc_cv_tfidf.best_params_)
print("GS RF TF-IDF Accuracy: ", rfc_cv_tfidf.best_score_)

# Порівняння точності
results = {
    'LogReg_Word2Vec': lr_bow_w2v_mean_score,
    'LogReg_TFIDF': lr_tfidf_mean_score,
    'RF_Word2Vec': rfc_bow_w2v_mean_score,
    'RF_TFIDF': rfc_tfidf_mean_score,
    'LogReg_Word2Vec_GridSearch': logreg_cv.best_score_,
    'LogReg_TFIDF_GridSearch': logreg_cv_tfidf.best_score_,
    'RF_Word2Vec_GridSearch': rfc_cv_w2v.best_score_,
    'RF_TFIDF_GridSearch': rfc_cv_tfidf.best_score_
}

for model, score in results.items():
    print(f'{model}: {score:.4f}')
