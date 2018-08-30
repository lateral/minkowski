#include "dictionary.h"

#include <assert.h>

#include <iostream>
#include <fstream>
#include <algorithm>
#include <iterator>
#include <cmath>

namespace minkowski {

const std::string Dictionary::EOS = "</s>";

Dictionary::Dictionary(std::shared_ptr<Args> args) : args_(args),
    word2int_(HASHTABLE_SIZE, -1), size_(0), nwords_(0),
    ntokens_(0) {}

int32_t Dictionary::find(const std::string& w) const {
    uint32_t h = hash(w);
    int32_t idx = h % HASHTABLE_SIZE;
    while (word2int_[idx] != -1 && words_[word2int_[idx]].word != w) {
        idx = (idx + 1) % HASHTABLE_SIZE;
    }
    return idx;
}

void Dictionary::record_occurrence(const std::string& w) {
    int32_t h = find(w);
    ntokens_++;
    if (word2int_[h] == -1) {
        // word is not yet in the dictionary, so add it
        entry e;
        e.word = w;
        e.count = 1;
        words_.push_back(e);
        word2int_[h] = size_++;
    } else {
        // word _is_ in the dictionary, so just increment its count
        words_[word2int_[h]].count++;
    }
}

bool Dictionary::discard(int32_t id, real rand) const {
    assert(id >= 0);
    assert(id < nwords_);
    return rand > retention_probas[id];
}

uint32_t Dictionary::hash(const std::string& str) const {
    uint32_t h = 2166136261;
    for (size_t i = 0; i < str.size(); i++) {
        h = h ^ uint32_t(str[i]);
        h = h * 16777619;
    }
    return h;
}

bool Dictionary::read_word(std::istream& in, std::string& word) const
{
    char c;
    std::streambuf& sb = *in.rdbuf();
    word.clear();
    while ((c = sb.sbumpc()) != EOF) {
        if (c == ' ' || c == '\n' || c == '\r' || c == '\t' || c == '\v' ||
                c == '\f' || c == '\0') {
            if (word.empty()) {
                if (c == '\n') {
                    word += EOS;
                    return true;
                }
                continue; // skip over leading whitespace that isn't \n
            } else {
                // whitespace occurred, so stop collecting chars
                if (c == '\n')
                    sb.sungetc();
                return true;
            }
        }
        word.push_back(c);
    }
    // trigger eofbit
    in.get();
    return !word.empty();
}

void Dictionary::determine_vocabulary(std::istream& in) {
    std::string word;
    int64_t minThreshold = 1;
    while (read_word(in, word)) {
        record_occurrence(word);
        if (ntokens_ % 1000000 == 0) {
            std::cerr << "\rRead " << ntokens_  / 1000000 << "M words" << std::flush;
        }
        if (size_ > 0.75 * HASHTABLE_SIZE) {
            throw std::invalid_argument("Vocabulary getting too large for hash table: try a higher -min-count.");
        }
    }
    threshold(args_->min_count);
    calculate_retention_probas();
    std::cerr << "\rRead " << ntokens_  / 1000000 << "M words" << std::endl;
    std::cerr << "Number of words:  " << nwords_ << std::endl;
    if (size_ == 0) {
        throw std::invalid_argument(
                    "Empty vocabulary. Try a smaller -min-count value.");
    }
}

void Dictionary::threshold(int64_t t) {
    sort(words_.begin(), words_.end(), [](const entry& e1, const entry& e2) {
        return e1.count > e2.count;
    });
    words_.erase(remove_if(words_.begin(), words_.end(), [&](const entry& e) {
        return (e.count < t);
    }), words_.end());
    words_.shrink_to_fit();
    size_ = 0;
    nwords_ = 0;
    std::fill(word2int_.begin(), word2int_.end(), -1);
    for (auto it = words_.begin(); it != words_.end(); ++it) {
        int32_t h = find(it->word);
        word2int_[h] = size_++;
        nwords_++;
    }
}

void Dictionary::calculate_retention_probas() {
    retention_probas.resize(size_);
    real proba;
    for (size_t i = 0; i < size_; i++) {
        if (args_->t > 0) {
            real f = real(words_[i].count) / real(ntokens_);
            proba = std::sqrt(args_->t / f) + args_->t / f;
            if (proba > 1.) {
                proba = 1.;
            }
            retention_probas[i] = proba;
        } else {
            retention_probas[i] = 1.;
        }
    }
}

std::vector<int64_t> Dictionary::get_counts() const {
    std::vector<int64_t> counts;
    for (auto& w : words_) {
        counts.push_back(w.count);
    }
    return counts;
}

int32_t Dictionary::get_line(std::istream& in,
                             std::vector<int32_t>& words,
                             std::minstd_rand& rng) const {
    std::uniform_real_distribution<> uniform(0, 1);
    std::string token;
    int32_t ntokens = 0; // number of vocab tokens consumed

    // reset to the beginning of the stream, if at the end
    if (in.eof()) {
        in.clear();
        in.seekg(std::streampos(0));
    }

    words.clear();
    while (read_word(in, token)) {
        int32_t h = find(token);
        int32_t wid = word2int_[h];
        if (wid < 0) continue;

        ntokens++;
        if (!discard(wid, uniform(rng))) {
            words.push_back(wid);
        }
        if (token == EOS) break;
    }
    return ntokens;
}

}
