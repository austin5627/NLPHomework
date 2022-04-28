import sys
from collections import defaultdict


def get_counts(file_name):
    u_grams = defaultdict(int)
    bi_grams = defaultdict(int)

    with open(file_name, 'r') as train:
        for line in train:
            words = line.strip('\n').split(' ')
            for i, w in enumerate(words):
                u_grams[w] += 1
                if i != len(words) - 1:
                    bi_grams[(w, words[i + 1])] += 1
    return u_grams, bi_grams


def process_sentence(sentence, u_grams, bi_grams, smoothing):
    print(f"Sentence: {sentence}")
    words = sentence.split(" ")
    s_bg = []

    corpus = sum([u_grams[i] for i in u_grams])
    vocab = len(u_grams)

    print("Sentence Bi-grams:")
    for i, w in enumerate(words[:-1]):
        bi_gram = (w, words[i + 1])
        s_bg += [bi_gram]
        print(f"\t'{bi_gram[0]}', '{bi_gram[1]}': {bi_grams[bi_gram] + (1 if smoothing else 0)}")

    print("Sentence Bi-gram probabilities")
    if smoothing:
        prob_sentence = (u_grams[words[0]] + 1) / (corpus + vocab)
    else:
        prob_sentence = u_grams[words[0]] / corpus
    for i in s_bg:
        if not smoothing:
            prob = (bi_grams[i]) / u_grams[i[0]]
        else:
            prob = (bi_grams[i] + 1) / (u_grams[i[0]] + vocab)
        print(f"\t'{i[0]}', '{i[1]}': {prob}")
        prob_sentence *= prob
    print(f"The probability of the sentence is: {prob_sentence}")


def main():

    if len(sys.argv) != 3:
        print("Invalid Arguments")

    train_file = sys.argv[1]
    smoothing = int(sys.argv[2])

    u_grams, bi_grams = get_counts(train_file)

    sentence_1 = "mark antony , heere take you caesars body : you shall not come to them poet ."
    sentence_2 = "no , sir , there are no comets seen , the heauens speede thee in thine enterprize ."
    print(f"{'With' if smoothing else 'No'} smoothing")
    print()
    process_sentence(sentence_1, u_grams, bi_grams, smoothing=smoothing)
    print()
    process_sentence(sentence_2, u_grams, bi_grams, smoothing=smoothing)


if __name__ == '__main__':
    main()
