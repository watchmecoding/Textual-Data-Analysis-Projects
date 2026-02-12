import re
import nltk
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from nltk.corpus import stopwords
from gensim.models import word2vec
from sklearn.feature_extraction.text import CountVectorizer
from sklearn.feature_extraction.text import TfidfVectorizer

fil = open ("doc8.txt", encoding ='utf-8', mode = "r")
text = fil.read()
fil.close()

print("Text read:")
print(text)
docs = text.split("\n")
#print(docs)

wpt = nltk.WordPunctTokenizer()
stop_words = nltk.corpus.stopwords.words('english')
corpus = []

for doc in docs:
    doc = doc.lower()
    doc = re.sub(r'[^a-zA-Z0-9\s]', '', doc)

    tokens = wpt.tokenize(doc)
    filtered_tokens = [token for token in tokens if token not in stop_words]
    doc = ' '.join(filtered_tokens) 
    corpus.append(doc)
    # print(doc)

# print(corpus)

print ("\n\nBag of 2-Grams")
bv = CountVectorizer(ngram_range=(2,2))
bv_matrix = bv.fit_transform(corpus) #документи корпусу у вектори
bv_matrix = bv_matrix.toarray()
vocab = bv.get_feature_names_out()
print (pd.DataFrame(bv_matrix, columns=vocab))

print ("\n\nBag of 1-Grams")
cv = CountVectorizer(ngram_range=(1,1))
cv_matrix = cv.fit_transform(corpus)
cv_matrix = cv_matrix.toarray()
cvocab = cv.get_feature_names_out()
data = pd.DataFrame(cv_matrix, columns=cvocab)
print (data)

print ("\n\nVector for 'profits'")
print(data['profits'])

###############################################################

print ("\n\nTF-IDF Model")
tv = TfidfVectorizer(min_df=0., max_df=1., norm='l2', use_idf=True, smooth_idf=True ) #min/max частоти, які враховуються, нормалізація до [0...1]
tv_matrix = tv.fit_transform(corpus)
tv_matrix = tv_matrix.toarray()
tvocab = tv.get_feature_names_out()
data1 = pd.DataFrame(np.round(tv_matrix, 2), columns=tvocab)
print (data1)

#Косинусна подібність
from sklearn.metrics.pairwise import cosine_similarity
similarity_matrix = cosine_similarity(tv_matrix)
print (pd.DataFrame(similarity_matrix))

from scipy.cluster.hierarchy import dendrogram, linkage
links = linkage(similarity_matrix, 'ward')
plt.figure(figsize=(8, 3))
plt.title('Дендрограма')
plt.xlabel('Документи')
plt.ylabel('Відстань')
dendrogram(links)


from scipy.cluster.hierarchy import fcluster
max_dist = 1.5
cluster_labels = fcluster(links, max_dist, criterion='distance')
print ("\n\nClasters on 1.5")
print (cluster_labels)
max_dist = 1.3
cluster_labels = fcluster(links, max_dist, criterion='distance')
print ("\nClasters on 1.3")
print (cluster_labels)

###############################################################

wpt = nltk.WordPunctTokenizer()
tokenized_corpus = [wpt.tokenize(document) for document in corpus]

feature_size = 10    # Розмір векторів слів (кожне слово - вектор з 10 компонент)
window_context = 10  # Довжина вікна контексту (10 слів до та після цільового слова - його контекст)
min_word_count = 1   # Мінімальна частота слів для врахування в словнику
sample = 1e-3        # Зменшення частоти повторення слів

w2v_model = word2vec.Word2Vec(tokenized_corpus, vector_size=feature_size,
                          window=window_context, min_count = min_word_count,
                          sample=sample) #Повертає вектори слів
print ("\n\npepper")
print (w2v_model.wv['pepper']) #Повертає вектор слова 'pepper'

print("3 words similar to pepper")
words = w2v_model.wv.most_similar("pepper", topn=3) # Повертає послідовність (key, similarity)
for word in words:
  print(word)

print ("\n\npompeii")
print (w2v_model.wv['pompeii'])

print("3 words similar to pompeii")
words = w2v_model.wv.most_similar("pompeii", topn=3)
for word in words:
  print(word)

plt.show()