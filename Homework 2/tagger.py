import os
import sys
from collections import defaultdict

class Tagger:
    def __init__(self):
        """ Initialize class variables here """
        self.initial_tag_probs = None
        self.transition_probs = None
        self.emission_probs = None

    def load_corpus(self, path):
        """
        Returns all sentences as a sequence of (word, tag) pairs found in all
        files from as directory
        `path`.

        Inputs:
            path (str): name of directory
        Outputs:
            word_tags: 2d-list that represent sentences in the corpus. Each
            sentence is then represented as a list of tuples (word, tag)
        """
        if not os.path.isdir(path):
            sys.exit("Input path is not a directory")

        sentence_list = []
        for filename in os.listdir(path):
            # Iterates over files in directory
            with open(path + filename, 'r') as file:
                for line in file:
                    words = line.lower().split()
                    word_tags = []
                    for word in words:
                        word_tags += [tuple(word.split('/'))]
                    if word_tags:
                        sentence_list += [word_tags]
        return sentence_list


    def initialize_probabilities(self, sentences):
        """
        Initializes the initial, transition and emission probabilities into
        class variables

        Inputs:
            sentences: a 2d-list of sentences, usually the output of
            load_corpus
        Outputs:
            None
        """
        if type(sentences) != list:
            sys.exit("Incorrect input to method")


        word_counts = defaultdict(int)
        tag_counts = defaultdict(int)
        total_words = 0
        transition_counts = defaultdict(lambda: 1)
        first_tag_counts = defaultdict(lambda: 1)
        word_tag_counts = defaultdict(lambda: 1)

        for sentence in sentences:
            first_tag_counts[sentence[0][1]] += 1
            for i, (word, tag) in enumerate(sentence):
                total_words += 1
                tag_counts[tag] += 1
                word_counts[word] += 1
                word_tag_counts[word + '//' + tag] += 1
                if i != 0:
                    transition_counts[sentence[i-1][1] + '//' + tag] += 1

        self.initial_tag_probs = defaultdict(lambda: (1/len(sentences)))
        self.transition_probs = defaultdict(lambda: (1/len(tag_counts))) 
        self.emission_probs = defaultdict(lambda: (1/len(word_counts))) 

        """ 1. Compute Initial Tag Probabilities """
        for tag in first_tag_counts:
            self.initial_tag_probs[tag] = first_tag_counts[tag]/(len(sentences) + tag_counts[tag])

        """ 2. Compute Transition Probabilities """
        for pair in transition_counts:
            self.transition_probs[pair] = transition_counts[pair]/(tag_counts[pair.split('/')[2]] + len(tag_counts))

        """ 3. Compute Emission Probabilities """
        for pair in word_tag_counts:
            self.emission_probs[pair] = word_tag_counts[pair]/(tag_counts[pair.split('/')[2]] + len(word_counts))

        return


    def viterbi_decode(self, sentence):
        """
        Implementation of the Viterbi algorithm

        Inputs:
            sentence (str): a sentence with N tokens, be those words or
            punctuation, in a given language
        Outputs:
            likely_tags (list[str]): a list of N tags that most likely match
            the words in the input sentence. The i'th tag corresponds to
            the i'th word.
        """


        sentence = sentence.lower()
        if type(sentence) != str:
            sys.exit("Incorrect input to method")

        """ Tokenize sentence """
        words = ['Start']
        words += sentence.split()

        """ Implement the Viterbi algorithm """
        viterbi = defaultdict(list)
        backpointer = defaultdict(list)

        for tag in self.initial_tag_probs:
            viterbi[tag] = [self.initial_tag_probs[tag] * self.emission_probs[words[0] + '//' + tag]]
            backpointer[tag] = ['START']

        tag_list = list(self.initial_tag_probs.keys())
        for word in words:
            for tag in tag_list:
                p, t = max([[viterbi[old_tag][-1] * self.transition_probs[old_tag + '//' + tag] * self.emission_probs[word + '//' + tag], old_tag] for old_tag in tag_list])
                viterbi[tag] += [p]
                backpointer[tag] += [t]
                
        p, t = max([viterbi[old_tag][-1], old_tag] for old_tag in tag_list)
        
        viterbi['END'] = [p]
        backpointer['END'] = [t] * len(words)

        curr = 'END'
        likely_tags = []
        for i in range(len(words)-1, -1, -1):
            curr = backpointer[curr][i]
            likely_tags += [curr]
        

        return likely_tags[-2::-1]

if __name__ == "__main__":
    hmm_tagger = Tagger()
    sentences = hmm_tagger.load_corpus('./modified_brown/')
    hmm_tagger.initialize_probabilities(sentences)
    tests = ['the planet jupiter and its moons are in effect a mini solar system .', 'computers process programs accurately .']
    for test in tests:
        pos = hmm_tagger.viterbi_decode(test)
        print(test)
        print(pos)
