import pandas as pd

from .manifold import *


def read_questions(questions_file):
    """
    Read the word2vec analogy questions given
    the path to the questions file. Returns a
    list of question quadruples [a, b, c, d] with
    the analogy relation a:b = c:d.
    """
    questions = []
    with open(questions_file) as f:
        for line in f:
            if line[0] == ':' or line == '':
                continue
            w1, w2, w3, w4 = line.strip('\n').split(' ')
            questions.append([w1, w2, w3, w4])
    return questions


def analogy(a, b, c):
    """
    Tranport the vector a->b so that its tail is at c.
    """
    tangent = logarithm(a, b)
    direction = logarithm(a, c)
    transported = geodesic_parallel_transport(a, direction, tangent)
    return exponential(c, transported)


def closest_words(vec, word_vecs, n=10):
    """
    Return the closest 'n' words to the vector 'vec' in the
    pd.DataFrame 'word_vecs'.
    """
    mdps = minkowski_dot_matrix(vec[np.newaxis, :], word_vecs.values)[0]
    mdps[mdps > -1] = -1
    dists = pd.Series(np.arccosh(-mdps), index=word_vecs.index)
    dists = dists.sort_values()
    return dists.head(n)


def word_analogy(a, b, c, word_vecs):
    """
    Returns a list of candidates X that should satisfy:
    a:b = c:X. 'vecs' is a pd.DataFrame holding the word vectors
    indexed by the vocabulary.
    """
    vec = analogy(np.array(word_vecs.loc[a]), np.array(word_vecs.loc[b]), np.array(word_vecs.loc[c]))
    return closest_words(vec, word_vecs)
