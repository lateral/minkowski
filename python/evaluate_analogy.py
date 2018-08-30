# coding: utf-8
from hyperboloid_helpers.analogy import *
from hyperboloid_helpers.fasttext import *

ANALOGY_QUESTIONS_FILE = '/home/ubuntu/data/questions-words.txt'


def normalize(array, l=2, axis=None, return_norm=False):
    div = np.linalg.norm(array, ord=l, axis=axis, keepdims=True)
    if return_norm:
        nrm = div.copy().squeeze()

    div[np.isclose(div, 0)] = 1.
    if return_norm:
        return array / div, nrm
    return array / div


def compute_accuracy_euclidean(word_vecs, questions):
    """
    Computes the accuracy of the analogy task according to 
    Mikolov's script, i.e. normalizing the word vectors and
    excluding the 3 question words before scoring.
    'word_vecs' is a DataFrame containing the word vectors,
    'questions' is a list in which each element is a list
    of four words, representing one line of the analogy task.
    The similarity function used for scoring is the cosine similarity.
    """
    vecs = normalize(word_vecs.as_matrix(), l=2, axis=1)

    num_correct = 0
    num_processed = 0
    for i, q in enumerate(questions):

        w0 = q[0].lower()
        w1 = q[1].lower()
        w2 = q[2].lower()
        w3 = q[3].lower()

        try:
            vec0 = normalize(np.array(word_vecs.loc[w0]), l=2)
            vec1 = normalize(np.array(word_vecs.loc[w1]), l=2)
            vec2 = normalize(np.array(word_vecs.loc[w2]), l=2)

            left_side = vec1 - vec0 + vec2
            cos_sim = np.dot(left_side, vecs.T)

            # exclude query words
            keys = list(word_vecs.index)
            idx_0 = keys.index(w0)
            idx_1 = keys.index(w1)
            idx_2 = keys.index(w2)
            cos_sim[[idx_0, idx_1, idx_2]] = 0

            max_idx = np.argmax(cos_sim)
            nearest_word = word_vecs.index[max_idx]

            if nearest_word.lower() == w3:
                num_correct += 1
                
            num_processed += 1
                
        except Exception as e:
            print(str(e))
            continue
            
    accuracy = float(num_correct) / float(num_processed)
    return accuracy, num_processed


def compute_accuracy_hyperbolic(word_vecs, questions):
    """
    Computes the accuracy of the analogy task according to 
    Mikolov's script, i.e. normalizing the word vectors and
    excluding the 3 question words before scoring.
    'word_vecs' is a DataFrame containing the word vectors,
    'questions' is a list in which each element is a list
    of four words, representing one line of the analogy task.
    The distance function used for scoring is the hyperbolic
    distance.
    """
    num_correct = 0
    num_processed = 0
    for i, q in enumerate(questions):

        w0 = q[0].lower()
        w1 = q[1].lower()
        w2 = q[2].lower()
        w3 = q[3].lower()

        try:
            closest = list(word_analogy(w0, w1, w2, word_vecs).index)

            # exclude query words
            if w0 in closest:
                closest.remove(w0)
            if w1 in closest:
                closest.remove(w1)
            if w2 in closest:
                closest.remove(w2)
 
            nearest_word = closest[0]

            if nearest_word.lower() == w3:
                num_correct += 1
                
            num_processed += 1
                
        except Exception as e:
            print(str(e))
            continue
            
    accuracy = float(num_correct) / float(num_processed)
    return accuracy, num_processed


def run_analogy_evaluation(hyperbolic_files, euclidean_files):

    hyperbolic_analogy = {}
    euclidean_analogy = {}

    questions = read_questions(ANALOGY_QUESTIONS_FILE)

    for f in hyperbolic_files:
        vecs = load_minkowski_vectors(f)
        accuracy, num_processed = compute_accuracy_hyperbolic(vecs, questions)
        hyperbolic_analogy[f] = accuracy
        print(f)
        print('Processed {} out of {} questions.'.format(num_processed, len(questions)))
        print('Accuracy = {}'.format(accuracy))
        print('\n\n')

    for f in euclidean_files:
        vecs = load_fasttext_vectors(f)
        accuracy, num_processed = compute_accuracy_euclidean(vecs, questions)
        euclidean_analogy[f] = accuracy
        print(f)
        print('Processed {} out of {} questions.'.format(num_processed, len(questions)))
        print('Accuracy = {}'.format(accuracy))
        print('\n\n')

    with open('results_analogy.txt', 'w') as f:
        for k in hyperbolic_analogy:
            f.write('{}\t{}\n'.format(k, hyperbolic_analogy[k]))
        for k in euclidean_analogy:
            f.write('{}\t{}\n'.format(k, euclidean_analogy[k]))

    return hyperbolic_analogy, euclidean_analogy


if __name__ == '__main__':

    hyperbolic_files = ['vecs-wikipedia-hyperbolic-dim-100-epochs-3-lr-0.01-t-1e-05-ws-10-minCount-15-neg-10-initStddev-0.01.csv',
                        'vecs-wikipedia-hyperbolic-dim-20-epochs-3-lr-0.005-t-1e-05-ws-10-minCount-15-neg-10-initStddev-0.01.csv',
                        'vecs-wikipedia-hyperbolic-dim-5-epochs-3-lr-0.005-t-1e-05-ws-10-minCount-15-neg-10-initStddev-0.01.csv',
                        'vecs-wikipedia-hyperbolic-dim-50-epochs-3-lr-0.005-t-1e-05-ws-10-minCount-15-neg-10-initStddev-0.01.csv']

    euclidean_files = ['vecs-wikipedia-euclidean-dim-100-epochs-3-lr-0.05-t-1e-05-ws-10-minCount-15-neg-10.vec',
                       'vecs-wikipedia-euclidean-dim-20-epochs-3-lr-0.01-t-1e-05-ws-10-minCount-15-neg-10.vec',
                       'vecs-wikipedia-euclidean-dim-5-epochs-3-lr-0.005-t-1e-05-ws-10-minCount-15-neg-10.vec',
                       'vecs-wikipedia-euclidean-dim-50-epochs-3-lr-0.1-t-1e-05-ws-10-minCount-15-neg-10.vec']

    hyperbolic_analogy, euclidean_analogy = run_analogy_evaluation(hyperbolic_files=hyperbolic_files,
                                                                   euclidean_files=euclidean_files)
