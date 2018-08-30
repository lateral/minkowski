#pragma once

#include <vector>
#include <string>
#include <istream>
#include <ostream>
#include <random>
#include <memory>
#include <unordered_map>

#include "args.h"
#include "real.h"

namespace minkowski {

struct entry {
    std::string word;
    int64_t count;
};

class Dictionary {
protected:
    static const int32_t HASHTABLE_SIZE = 100000000;

    /*
     * Return the index into word2int_ of the specified word, or, if the
     * word is not in the dictionary, the index of the next available slot.
     * Post: word2int_[result] == -1 || words_[word2int_[result]].word == word
     */
    int32_t find(const std::string& word) const;

    /*
     * Calculate the discard probabilities (used for subsampling).
     */
    void calculate_retention_probas();

    std::shared_ptr<Args> args_;

    /*
     * Hash table implementation.  Collisions are resolved by moving to the
     * next available slot.
     * word2int_ is a vector mapping hashes of strings (so ints) to indices of word_
     * (so most of its values are -1)
     * words_ is a vector of entry structs (which consist of a word (string) and its
     * occurrence count)
     */
    uint32_t hash(const std::string& str) const;
    std::vector<int32_t> word2int_;

    /*
     * Record an occurrence of the specified word, adding it to the dictionary
     * if it is not already there.
     */
    void record_occurrence(const std::string&);

    std::vector<real> retention_probas; // retention probability for each word

    /*
     * Discard all words that occur less than the specified number of times.
     */
    void threshold(int64_t);

public:
    std::vector<entry> words_;
    static const std::string EOS;
    int32_t nwords_;
    int64_t ntokens_;
    int32_t size_; // number of words in vocab (FIXME difference from nwords_?)

    explicit Dictionary(std::shared_ptr<Args>);

    /*
     * Return whether the specified word should be discarded, given the
     * specified random outcome.
     */
    bool discard(int32_t id, real rand) const;

    /*
     * Extract the next word (=sequence of chars unbroken by whitespace) from
     * the input stream.  A single EOS character is extracted when a line break
     * is detected. Return whether any characters were extracted (a bool).
     */
    bool read_word(std::istream&, std::string&) const;

    /*
     * Determine the vocabulary by counting the occurrences of tokens in the
     * provided input stream.
     */
    void determine_vocabulary(std::istream&);

    /*
     * Return a vector giving the occurrence count of the words in the dictionary.
     */
    std::vector<int64_t> get_counts() const;

    /*
     * Populate `words` with tokens read from the input stream, performing any
     * subsampling.  Does not continue over linebreaks (\n).
     * Returns the number of dictionary tokens consumed from the input stream
     * in the processed (regardless of whether they were subsequently discarded
     * due to subsampling).
     */
    int32_t get_line(std::istream& in, std::vector<int32_t>& words,
                     std::minstd_rand& rng) const;
};

}
