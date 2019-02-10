from builtins import dict
import random
from gensim.models import Word2Vec
from gensim.models import KeyedVectors

# TODO: account for sentences (in embeddings, in data and sources, etc.)

def load_data(path, sentences=False):
    """
    Function to load annotated data.
    :param path: path to file containing annotated data
    :return: a list, containing lists with token, lemma, POS (and morph if provided)
    """
    data = []
    with open(path, 'r') as f:
        for line in f.readlines():
            data.append(line.rstrip().split('\t'))

    return data


def load_embed_data(path):
    """
    Loads data for training embeddings (based on lemma value)
    :param path: path to data
    :return: a list containing lists (sentences) of lemmata
    """
    data = []
    with open(path, 'r') as f:
        sentence = [] # initial sentence
        for line in f.readlines():
            line = line.rstrip().split('\t')
            if line == [''] or line[1] in ['.', '?', '!']:
                #Possible d'ajouter une longue max ici
                # type: or len(sentence) < n
                if not line == ['']:
                    sentence.append(line[1]) # keep punctuation.

                data.append(sentence)
                sentence = [] # Sentence is finished and we go for a new one

            else:
            sentence.append(line[1])

    return data


def augment(data, sources, use_morph=False, use_lemma=False, use_embeddings=False, embed_data_path=None, pretrained_embeddings=None):
    """
    Given a input dataset, augment it by randomly selecting alternative
    words (based on lemma and POS) in one or several sources.
    :param data: input dataset to augment
    :param sources: sources to use for augmentation
    :param use_morph: whether to account for morphological information
    :param use_lemma: whether to account for lemma (implemented only with use_morph)
    :param use_embeddings: whether to use word embeddings to extend the previous option, with
    the use of other lemmata with similar semantic properties
    :param embed_data_path: path to data to train embeddings (optional)
    :param pretrained_embeddings: path to pretrained embeddings.
    :return: augmented dataset
    """
    augm = []

    # Now, deal with the embeddings
    if use_embeddings:
        if pretrained_embeddings is None:
            # TODO: account for sentences in the default here
            if embed_data_path is None:
                embed_data = [[i[1] for i in data+sources]]

            if embed_data_path is not None:
                embed_data = load_embed_data(embed_data_path)

            embs = Word2Vec(embed_data, iter=50, size=150,
                            window=5, negative=5, min_count=1, workers=4)
            embs.train(embed_data, total_examples=len(embed_data), epochs=100)

        else:
            embs = KeyedVectors.load_word2vec_format(pretrained_embeddings, binary=False)

    alternatives = get_alternatives(sources, use_lemma=use_lemma, use_morph=use_morph)
    # and a fallback, for when there is not alternative in sources
    fallbacks = get_alternatives(data, use_lemma=use_lemma, use_morph=use_morph)
    for row in data:
        if not use_morph:
            augm.append(random.sample(alternatives[row[2]], 1))

        if use_morph and not use_lemma:
            augm.append(random.sample(alternatives[row[2]][row[3]], 1))

        if use_morph and use_lemma and not use_embeddings:
            # check if morph and lemma key exist in dic
            if row[3] in alternatives[row[2]] and row[1] in alternatives[row[2]][row[3]]:
                augm.append(random.sample(alternatives[row[2]][row[3]][row[1]], 1))
            # or use the fallback
            else:
                augm.append(random.sample(fallbacks[row[2]][row[3]][row[1]], 1))

        if use_morph and use_lemma and use_embeddings:
            # Get similar lemmas (and concatenate it with original lemma)
            synonyms = [i[0] for i in embs.wv.most_similar(positive=row[1])] + [row[1]]
            # filter to keep only those that are in dic
            valid_syns = [s for s in synonyms if s in alternatives[row[2]][row[3]]]
            if not valid_syns == []:
                # pick one
                my_syn = random.sample(valid_syns, 1)
                # find good alternates
                # NB: du coup, en filtrant avant, on pert l'indication de fréquence, qu'on aurait si on
                # prenait toutes les occurrences des synonymes
                valid_alts = [alternatives[row[2]][row[3]][s] for s in my_syn]
                augm.append(random.sample(valid_alts[:][0], 1))# C'est moche, à simplifier (set dans une liste)

            # or use the fallback
            # TODO: could be improved. Concatenate data and sources with this option?
            else:
                augm.append(random.sample(fallbacks[row[2]][row[3]][row[1]], 1))

    return augm


def get_alternatives(sources, use_morph=False, use_lemma=False):
    """
    Creates sets of alternatives for each available category (now: POS only)
    :param sources: dataset(s) to use for creating the sets of alternatives
    :param use_lemma: whether to account for lemma
    :param use_morph: whether to account for morphological information
    :return: a dictionary containing sets of alternatives
    """
    alternatives = dict()
    POS = {i[2] for i in sources}
    for tag in POS:
        if not use_morph:
            alternatives[tag] = {tuple(r[0:3]) for r in sources if r[2] == tag}

        if use_morph:
            alternatives[tag] = dict()
            entries = {tuple(r) for r in sources if r[2] == tag}
            morph = {i[3] for i in entries}
            for m in morph:
                if not use_lemma:
                    alternatives[tag][m] = {tuple(r) for r in entries if r[3] == m}

                if use_lemma:
                    alternatives[tag][m] = dict()
                    entriesMorph = {tuple(r) for r in entries if r[3] == m}
                    lemmata = {i[1] for i in entriesMorph}
                    for l in lemmata:
                        alternatives[tag][m][l] = {tuple(r) for r in entriesMorph if r[1] == l}

    return alternatives



if __name__ == '__main__':
    # transdial.
    #data = load_data('data/data_morph.tsv')
    #sources = load_data('data/sources-AN.tsv')
    #augm = augment(data, sources, use_morph=True, use_lemma=True)

    # with embeds
    data = load_data('data/data_morph.tsv')
    sources = load_data('data/sources-full.tsv')
    augm = augment(data, sources, use_morph=True, use_lemma=True, use_embeddings=True, pretrained_embeddings='embeds/embeds_fro.txt')

    # other corpus
    data = load_data('data/data.tsv')
    sources = load_data('data/class-src.tsv')
    sources = data+sources
    augm = augment(data, sources, use_morph=True, use_lemma=True, use_embeddings=True, embed_data_path='data/class-embeds.tsv')

    with open('data/out.tsv', 'w') as out:
        #if use_morph == True:
        for line in augm:
            out.write('{}\t{}\t{}\t{}\n'.format(*line[0]))
            #out.write(str(line))
