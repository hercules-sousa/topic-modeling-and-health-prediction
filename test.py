import numpy as np
from sklearn.datasets import fetch_20newsgroups
from sklearn import decomposition
from sklearn.feature_extraction.text import CountVectorizer, TfidfVectorizer
from scipy import linalg


def get_topics(matrix, vocab, num_top_words):
    top_words = lambda t: [vocab[i] for i in np.argsort(t)[:-num_top_words - 1:-1]]
    topic_words = ([top_words(t) for t in matrix])
    return [' '.join(t) for t in topic_words]


def get_topics_with_svd(database, num_top_words, stop_words='english'):
    vectorizer = CountVectorizer(stop_words=stop_words)
    vectors = vectorizer.fit_transform(database.data).todense()
    vocab = np.array(vectorizer.get_feature_names())
    U, s, Vh = linalg.svd(vectors, full_matrices=False)
    return get_topics(Vh, vocab, num_top_words)

'''
class TopicModeler():
    def __init__(self):
        pass

    def get_topics(self, matrix, vocab, num_top_words):
        top_words = lambda t: [vocab[i] for i in np.argsort(t)[:-num_top_words - 1:-1]]
        topic_words = ([top_words(t) for t in matrix])
        return [' '.join(t) for t in topic_words]

    def get_topics_with_svd(self, database, num_top_words, stop_words='english'):
        vectorizer = CountVectorizer(stop_words=stop_words)
        vectors = vectorizer.fit_transform(database.data).todense()
        vocab = np.array(vectorizer.get_feature_names())
        U, s, Vh = linalg.svd(vectors, full_matrices=False)
        return self.get_topics(Vh, vocab, num_top_words)

    def nmf(self, database, topic_number, stop_words='english'):
        vectors = self.turn_into_vectors(database)

        clf = decomposition.NMF(n_components=topic_number, random_state=1)

        W1 = clf.fit_transform(np.asarray(vectors))
        H1 = clf.components_
        return W1, H1
'''