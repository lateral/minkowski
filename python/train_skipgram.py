# coding: utf-8
import os
from itertools import product

INPUT_FILE = '/home/ubuntu/data/wikipedia.txt'
INPUT_LABEL = INPUT_FILE.split('/')[-1].split('.')[0]

FASTTEXT_DIR = '/home/ubuntu/fastText'
MINKOWSKI_DIR = '/home/ubuntu/minkowski'

EPOCHS = 3
T = 1e-5
WS = 10
MIN_COUNT = 15
NEG = 10
INIT_STDDEV = 0.01


def run_training(dim, epochs, lr, t, ws, min_count, neg, init_stddev):
    """
    Train a hyperbolic and a Euclidean skip-gram model given the provided
    hyperparameters. Returns the file names where the resulting embeddings
    were stored.
    """

    output_file_hyperbolic = 'vecs-{}-hyperbolic-dim-{}-epochs-{}-lr-{}-t-{}' \
                             '-ws-{}-minCount-{}-neg-{}-initStddev-{}'.format(
        INPUT_LABEL, dim, epochs, lr, t, ws, min_count, neg, init_stddev)

    os.system('/home/ubuntu/minkowski-build/minkowski -input {} -output {} '
              '-max-step-size 1.0 -dimension {} -start-lr {} -end-lr 0 '
              '-epochs {} -init-std-dev {} -min-count {} -t {} -window-size {} '
              '-number-negatives {} -threads 64'.format(
        INPUT_FILE, output_file_hyperbolic, dim + 1, lr, epochs, init_stddev,
        min_count, t, ws, neg))

    output_file_euclidean = 'vecs-{}-euclidean-dim-{}-epochs-{}-lr-{}-t-{}' \
                            '-ws-{}-minCount-{}-neg-{}'.format(
        INPUT_LABEL, dim, epochs, lr, t, ws, min_count, neg)

    os.system('/home/ubuntu/fastText-build-euc/fasttext skipgram -input {} '
              '-output {} -dim {} -lr {} -epoch {} -minCount {} -minn 0 '
              '-maxn 0 -t {} -ws {} -loss ns -neg {} -thread 64'.format(
        INPUT_FILE, output_file_euclidean, dim, lr, epochs, min_count, t, ws,
        neg))

    return output_file_hyperbolic + '.csv', output_file_euclidean + '.vec'


def sweep_lr_and_dimension(learning_rates, dimensions):
    """
    Trains a hyperbolic and Euclidean skip-gram model for each combination
    of learning rate and dimension provided. Stores the embeddings in csv
    file in the current working directory and returns two lists of file names
    for hyperbolic and Euclidean embeddings.
    """
    hyperbolic_files = []
    euclidean_files = []

    for lr, dim in product(learning_rates, dimensions):

        print('\nLearning rate {}, dimension {}\n'.format(lr, dim))
        try:
            h_file, e_file = run_training(dim, EPOCHS, lr, T, WS, MIN_COUNT,
                                          NEG, INIT_STDDEV)
            hyperbolic_files.append(h_file)
            euclidean_files.append(e_file)
        except Exception as e:
            print('Training failed: {}'.format(str(e)))

    return hyperbolic_files, euclidean_files


if __name__ == '__main__':
    learning_rates = [0.1, 0.05, 0.01, 0.005]
    dimensions = [5, 20, 50, 100]

    print('Train word embeddings for {} different settings.'.format(
        len(learning_rates) * len(dimensions)))
    hyperbolic_files, euclidean_files = sweep_lr_and_dimension(
        learning_rates=learning_rates,
        dimensions=dimensions)

    print('Finished training. Word vectors have been written to:')

    [print(f) for f in hyperbolic_files]
    [print(f) for f in euclidean_files]
