import pandas as pd
import csv
import nltk
import re
import numpy as np
from sklearn.decomposition import NMF
from sklearn.feature_extraction.text import TfidfVectorizer
from nltk.stem import WordNetLemmatizer
from nltk.corpus import gutenberg, stopwords, wordnet
from nltk.tokenize import word_tokenize
from nltk import pos_tag

def get_wordnet_pos(tag):
    if tag.startswith('J'):
        return wordnet.ADJ
    elif tag.startswith('V'):
        return wordnet.VERB
    elif tag.startswith('N'):
        return wordnet.NOUN
    elif tag.startswith('R'):
        return wordnet.ADV
    else:
        return wordnet.NOUN  # за замовчуванням

documents = pd.read_csv('IT_tickets1.csv', encoding='utf-8', on_bad_lines='skip')

wpt = nltk.WordPunctTokenizer()
stop_words = nltk.corpus.stopwords.words('english')
lemmatizer = WordNetLemmatizer()
clear = []

for doc in documents.Document:
    doc = doc.lower()
    doc = re.sub(r'[^a-zA-Z0-9\s]', '', doc)

    tokens = wpt.tokenize(doc)
    filtered_tokens = [token for token in tokens if token not in stop_words]
    pos_tags = nltk.pos_tag(filtered_tokens)
    lemmatized_tokens = [lemmatizer.lemmatize(token, get_wordnet_pos(tag)) for token, tag in pos_tags]
    #doc = ' '.join(filtered_tokens)
    doc = ' '.join(lemmatized_tokens)
    clear.append(doc)

documents['Clean'] = clear
print (documents)

vect = TfidfVectorizer(min_df=25, use_idf=True)
X = vect.fit_transform(documents.Clean)


model = NMF(n_components=10, random_state=5) # Create an NMF instance
model.fit(X)# Fit the model to TF-IDF
nmf_features = model.transform(X)# Transform the TF-IDF: nmf_features
components_df = pd.DataFrame(model.components_, columns=vect.get_feature_names_out()) 
print (components_df)

for topic in range(components_df.shape[0]):
    tmp = components_df.iloc[topic]
    print("For topic " + str(topic) + " the words with the highest value are: ")
    print(tmp.nlargest(5))
    print()

#print (nmf_features)

# topic_names = []
for topic in range(nmf_features.shape[1]):  
    tmp = []
    for doc in range(nmf_features.shape[0]):
        tmp.append(nmf_features[doc, topic])
    maxi = max(tmp)
    i = tmp.index(maxi)
    # topic_name = documents.iloc[i]['Topic_group']  # Витягуємо назву теми з відповідного документа
    # topic_names.append(topic_name)
    print(f"For topic {topic} the document with the highest value has id: {i}")
    # print(maxi)
    # print (tmp[i])
    # print(documents.Clean[i])

print()
# print (pd.DataFrame(nmf_features).idxmax(axis=1).value_counts())

new_doc = []
new_doc.append("please provide access to full reports catalogue by end of day tuesday issue with viewing transactions pm thanks application engineer")
new_doc.append("connection lost multiple times during login received error please check if server is stable need access for weekend shift regards system analyst")
new_doc.append("hello unable to submit form due to status error please delete and reopen ticket need to reprocess by friday thanks hr coordinator")

for i in range(len(new_doc)):  
    print (new_doc[i])

X = vect.transform(new_doc)
nmf_features = model.transform(X)
print(pd.DataFrame(nmf_features))
print(pd.DataFrame(nmf_features).idxmax(axis=1))

# Використати текст chesterton-brown.txt з корпусу gutenberg бібліотеки nltk та вивести ключові біграми.

from nltk.collocations import BigramCollocationFinder
from nltk.collocations import BigramAssocMeasures

# Отримання сирого тексту
text = gutenberg.raw('chesterton-brown.txt').lower()

# Очистка тексту
text = re.sub(r'[^a-zA-Z0-9\s]', '', text)

# Токенізація
tokens = word_tokenize(text)

# Видалення стоп-слів
stop_words = set(stopwords.words('english'))
tokens = [token for token in tokens if token not in stop_words]

# POS-тегування
tagged_tokens = pos_tag(tokens)

# Лематизація
lemmatizer = WordNetLemmatizer()
lemmatized_tokens = [
    lemmatizer.lemmatize(word, get_wordnet_pos(tag))
    for word, tag in tagged_tokens
]

# Пошук біграмів з найвищим PMI
bigram_measures = BigramAssocMeasures()
finder = BigramCollocationFinder.from_words(lemmatized_tokens)

# Виведення топ-10 біграмів за PMI
print("Top 10 bigrams with highest PMI:")
print(finder.nbest(bigram_measures.pmi, 10))