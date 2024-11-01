import math
import string
from pprint import pprint
import gensim.test.utils
from gensim import corpora, models
from gensim.similarities import SparseTermSimilarityMatrix, WordEmbeddingSimilarityIndex
import gensim.downloader as api
from scipy.linalg import lstsq
import numpy as np
import matplotlib.pyplot as plt
from sklearn.decomposition import PCA
from sympy.stats.sampling.sample_scipy import scipy

text1 = 'I love playing football'
text2 = 'I like watching football'
text3 = 'Мне нравиться играть футбал'
text4 = 'I am going home'
text5 = 'я иду домой'
text6 = 'я люблю смотреть футбал'

def preprocess (text):
    clean_text = text.translate(str.maketrans('', '', string.punctuation))
    token = clean_text.split(' ')
    return token

def embedded_soft_similarity (model, corpus1, corpus2, dic):
    similarity_index = WordEmbeddingSimilarityIndex(model)
    similarity_matrix = SparseTermSimilarityMatrix(similarity_index, dic)
    similarity = similarity_matrix.inner_product(corpus1, corpus2, normalized=(True, True))
    pprint(similarity)
    #print(similarity_matrix.matrix)

def simple_soft_similarity (vec1, vec2, simMat):
    x = 0
    y = 0
    z = 0
    for i in range(0, 10):
        for j in range(0, 10):
            x += simMat[i][j] * vec1[i] * vec2[j]
            y += simMat[i][j] * vec1[i] * vec1[j]
            z += simMat[i][j] * vec2[i] * vec2[j]
    print(x, y, z)
    sim = x / (math.sqrt(y) * math.sqrt(z))
    print(sim)

def mat_similarity (vec1, vec2):
    simMat = []
    for i in range(0, len(vec1)):
        row = []
        for j in range(0, len(vec2)):
            if (vec1[i] != vec2[j]):
                row.append(0)
            else:
                row.append(1)
        simMat.append(row)
    return simMat

def vector_space_visualizer(model):
    # Using PCA for dimensionality reduction
    pca = PCA(n_components=2)
    reduced_vectors = pca.fit_transform(model.wv.vectors)
    # Plotting the vectors
    plt.figure(figsize=(12, 8))
    plt.scatter(reduced_vectors[:, 0], reduced_vectors[:, 1])
    # Annotate the points with the corresponding words
    for i, word in enumerate(list(model.wv.index_to_key)):
        plt.annotate(word, (reduced_vectors[i, 0], reduced_vectors[i, 1]), fontsize=9)
    plt.title('Word Vectors Visualization')
    plt.xlabel('PCA Component 1')
    plt.ylabel('PCA Component 2')
    plt.grid()
    plt.show()

def save_embeddings(model, filepath):
    with open(filepath, 'w') as f:
        f.write(f"{len(model.wv)} {model.vector_size}\n")  # Header
        for word in model.wv.index_to_key:
            vector = ' '.join(map(str, model.wv[word]))
            f.write(f"{word} {vector}\n")

'''
# Define the vectors and similarity matrix
v1 = [1, 1, 1, 1, 0, 0, 0, 0, 0, 0]
v2 = [1, 0, 0, 1, 1, 1, 0, 0, 0, 0]
v3 = [0, 0, 0, 0, 0, 0, 1, 1, 1, 1]

simple_soft_similarity(v1, v2, mat_similarity(v2, v1))
simple_soft_similarity(v1, v3, mat_similarity(v3, v1))
simple_soft_similarity(v1, v3, mat_similarity(v3, v1))
'''

#single language embedding
text = []
text.append(preprocess(text1))
text.append(preprocess(text2))
text.append(preprocess(text3))
text.append(preprocess(text4))
text.append(preprocess(text5))
text.append(preprocess(text6))
dic = corpora.Dictionary(text)
corpus1 = dic.doc2bow(text[0])
corpus2 = dic.doc2bow(text[1])
corpus3 = dic.doc2bow(text[2])
corpus4 = dic.doc2bow(text[3])
corpus5 = dic.doc2bow(text[4])
corpus6 = dic.doc2bow(text[5])

#word2vec model
#model = models.Word2Vec(sentences=text, vector_size=10, window=5, min_count=1, workers=4)
#model.save("word2vec.model")
#model = models.Word2Vec.load("word2vec.model")
'''
embedded_soft_similarity(model.wv, corpus1, corpus2, dic)
embedded_soft_similarity(model.wv, corpus2, corpus3, dic)
embedded_soft_similarity(model.wv, corpus1, corpus3, dic)

#GloVe model
model = api.load("glove-wiki-gigaword-50")
embedded_soft_similarity(model, corpus1, corpus2, dic)
embedded_soft_similarity(model, corpus2, corpus3, dic)
embedded_soft_similarity(model, corpus1, corpus3, dic)
'''

'''
#bilingual vector alignment
engtext = []
rustext = []
engtext.append(preprocess(text1))
engtext.append(preprocess(text2))
engtext.append(preprocess(text4))
rustext.append(preprocess(text3))
rustext.append(preprocess(text5))
rustext.append(preprocess(text6))
eng = models.Word2Vec(sentences=engtext, vector_size=10, window=5, min_count=1, workers=4)
rus = models.Word2Vec(sentences=rustext, vector_size=10, window=5, min_count=1, workers=4)
train = [
    ("I", "я"), ("football", "футбал"), ("like", "нравиться"), ("playing", "играть"), ("love", "люблю"), ("watching", "смотреть"), ("going", "иду"), ("home", "домой")
]

#transMat = models.TranslationMatrix(eng.wv, rus.wv, train)
#bilingual_model = models.TranslationMatrix(eng.wv, rus.wv, word_pairs=train)

# Align the vectors

#method 1: using orthogonal matrix to find nearest neighbor
engVecs = eng.wv.vectors
rusVecs = rus.wv.vectors

leftMat, _, rightMat = np.linalg.svd(engVecs.T @ rusVecs) #return only orthogonal matrices of left and right singular vectors
mixMat = leftMat @ rightMat.T
# Combine the models
combmodel = models.Word2Vec(vector_size=10, window=5)
for word in eng.wv.index_to_key:
    combmodel.wv.add_vector(word, eng.wv[word])
for word in rus.wv.index_to_key:
    combmodel.wv.add_vector(word, rus.wv[word])
combmodel.wv.vectors = np.concatenate((engVecs @ mixMat, rusVecs))
#vector_space_visualizer(combmodel)
#vector_space_visualizer(model)

#embedded_soft_similarity(combmodel.wv, corpus6, corpus3, dic)
'''
'''
#method 2: using anchor pairs
engAnchor_words = np.array(["I", "home"])
rusAnchor_words = np.array(["я", "домой"])
engAnchor = []
rusAnchor = []
#create anchor (word:value)
for eng_word, rus_word in zip(engAnchor_words, rusAnchor_words):
    if eng_word in eng.wv and rus_word in rus.wv:
        engAnchor.append(eng.wv[eng_word])
        rusAnchor.append(rus.wv[rus_word])
M, _, _, _ = lstsq(engAnchor, rusAnchor) #find closest matrix of rusAnchor to engAnchor
aligned_engVecs = engVecs @ M
# Combine the models
combmodel = models.Word2Vec(vector_size=10, window=5)
for word in eng.wv.index_to_key:
    combmodel.wv.add_vector(word, aligned_engVecs[eng.wv.key_to_index[word]])
for word in rus.wv.index_to_key:
    combmodel.wv.add_vector(word, rus.wv[word])
#vector_space_visualizer(combmodel)
'''

'''
model = models.Word2Vec(sentences=text, vector_size=10, window=5, min_count=1, workers=4)
model.save("word2vec.model")
model = models.Word2Vec.load("word2vec.model")
embedded_soft_similarity(model.wv, corpus1, corpus6, dic)
'''

'''
#visualization
vector_space_visualizer(eng)
vector_space_visualizer(rus)
'''
#vector_space_visualizer(combmodel)


#fasttext
corpus = [["I", "я"], ["football", "футбал"], ["like", "нравиться"], ["playing", "играть"],
          ["love", "люблю"], ["watching", "смотреть"], ["going", "иду"], ["home", "домой"]]
for sentence in text:
    corpus.append(sentence)
print(corpus)
fastModel = models.FastText(vector_size = 10, window=5, min_count=1)
fastModel.build_vocab(corpus_iterable=corpus)
fastModel.train(corpus_iterable=corpus, total_examples=len(corpus), epochs=10)
#embedded_soft_similarity(fastModel.wv, corpus5, corpus6, dic)
#vector_space_visualizer(fastModel)

embedded_soft_similarity(fastModel.wv, corpus1, corpus2, dic)
embedded_soft_similarity(fastModel.wv, corpus1, corpus4, dic)
embedded_soft_similarity(fastModel.wv, corpus2, corpus4, dic)
embedded_soft_similarity(fastModel.wv, corpus3, corpus5, dic)
embedded_soft_similarity(fastModel.wv, corpus3, corpus6, dic)
embedded_soft_similarity(fastModel.wv, corpus5, corpus6, dic)
