import pandas as pd


def load_fasttext_vectors(fname):
    """
    Loads a fasttext word vectors text file, returns DataFrame.
    """
    syn0 = pd.read_csv(fname, header=None, sep=' ', skiprows=1,
                       na_values=None, keep_default_na=False # these two are needed since otherwise Pandas maps "null" and "nan" to np.nan!
                       ).set_index(0)
    syn0 = syn0.drop(len(syn0.columns), axis=1)
    syn0.index = syn0.index.map(lambda x: str(x))
    return syn0


def load_minkowski_vectors(fname):
    """
    Loads a minkowski word vectors text file, returns DataFrame.
    """
    syn0 = pd.read_csv(fname, header=None, sep=' ',
                       na_values=None, keep_default_na=False # these two are needed since otherwise Pandas maps "null" and "nan" to np.nan!
                       ).set_index(0)
    syn0.index = syn0.index.map(lambda x: str(x))
    return syn0
