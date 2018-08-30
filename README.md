## Minkowski


### Overview
**Minkowski** implements skip-gram training for learning word embeddings from continous text in hyperbolic space.
Based on code from [fastText](https://fasttext.cc/), each word in the vocabulary is represented by a point on the hyperboloid model
in Minkowski space. The embeddings are then optimized by negative sampling to
minimize the hyperbolic distance of co-occurring words.

The differences to fastText are as follows:

- Word vectors are situated on the hyperboloid model of hyperbolic space.
- The similarity of two vectors is anti-proportional to their hyperbolic distance.
- In multithreaded training, individual word vectors are locked while being updated, so that no other thread can overwrite them and thus violate the constraint of the hyperboloid.
- The option to specify start and end learning rates and a number of _burnin_ epochs with lower learning rate.
- It is possible to store intermediate word vectors using the _checkpoint_ command line argument.
- It is possible to specify the power to which the unigram distribution is raised for negative sampling.

### Installation

In order to build the executable, a recent C++ compiler and CMake need to be
installed (tested with g++ 5.4.0 and CMake 3.11.0-rc2).

The following commands produce the executable *minkowski* in the build directory
:

```bash
git clone ... ./minkowski
cd .. & mkdir minkowski-build & cd minkowski-build
cmake ../minkowski
make
```

### Usage
The following command line parameters are available:

```bash
$ ./minkowski 
Empty input or output path.
  -input                  training file path
  -output                 output file path
  -min-count              minimal number of word occurences [5]
  -t                      sub-sampling threshold (0=no subsampling) [0.0001]
  -start-lr               start learning rate [0.05]
  -end-lr                 end learning rate [0.05]
  -burnin-lr              fixed learning rate for the burnin epochs [0.05]
  -max-step-size          max. dist to travel in one update [2]
  -dimension              dimension of the Minkowski ambient [100]
  -window-size            size of the context window [5]
  -init-std-dev           stddev of the hyperbolic distance from the base point for initialization [0.1]
  -burnin-epochs          number of extra prelim epochs with burn-in learning rate [0]
  -epochs                 number of epochs with learning rate linearly decreasing from -start-lr to -end-lr [5]
  -number-negatives       number of negatives sampled [5]
  -distribution-power     power used to modified distribution for negative sampling [0.5]
  -checkpoint-interval    save vectors every this many epochs [-1]
  -threads                number of threads [12]
  -seed                   seed for the random number generator [1]
                          n.b. only deterministic if single threaded!
```

An example call looks like this:

```bash
$ ./minkowski -input textfile.txt -output embeddings -dimension 50 -start-lr 0.1
-end-lr 0 -epochs 3 -min-count 15 -t 1e-5 -window-size 10 -number-negatives 10
-threads 64
```

### References

[1] Tomas Mikolov, Kai Chen, Greg Corrado, Jeffrey Dean: Efficient estimation of
 word representations in vector space. arXiv preprint arXiv:1301.3781. [pdf]
 (https://arxiv.org/pdf/1301.3781.pdf?)

[2] Maximilian Nickel, Douwe Kiela: Poincaré Embeddings for Learning
Hierarchical Representations. NIPS 2017. [pdf](https://papers.nips
.cc/paper/7213-poincare-embeddings-for-learning-hierarchical-representations.pdf)
