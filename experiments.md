## Hyperbolic skip-gram experiments

The **python** subfolder contains scripts to train and compare skip-gram word
embedding models in Euclidean and hyperbolic space. For Euclidean training, the
standard implementation in [fastText](https://fasttext.cc/) is used.

In order to run the experiments the following variables have to be set in the
scripts:

#### train_skipgram.py

**INPUT_FILE** should point to a preprocessed text file containing one document
per line. For our experiments, a 2013 dump of Wikipedia was used. Preprocessing
included lower-casing, removal of punctuation and removing articles with less
than 20 page views.

**FASTTEXT_DIR** should point to the build directory of fastText's master branch
.

**MINKOWSKI_DIR** should point to the build directroy of this project.

Training is then run for a combination of learning rates and dimensions and the
resulting word vectors stored in .vec files in the same directory.

#### evaluate_similarity.py

**SIMLARITY_DIR** is a directory that holds 13 similarity datasets that can be
obtained from [here](https://github.com/mfaruqui/eval-word-vectors/tree/master/data/word-sim).

#### evaluate_analogy.py

**ANALOGY_QUESTIONS_FILE** is the questions file from the original word2vec
repository, available [here](https://github.com/imsky/word2vec/blob/master/questions-words.txt).

#### Python dependencies

Please make sure that the **hyperboloid_helpers** module is on the PYTHONPATH.
To install other dependencies run

```bash
pip3 install -r requirements.txt
```

from the python subdirectory.


#### Usage

The scripts can be run as follows

```bash
python3 train_skipgram.py
python3 evaluate_similarity.py
python3 evaluate_analogy.py
```

and produce printed output on the command line as well as two files
**results_similarity.txt** and **results_analogy.txt** that save the scores per
embedding file.
