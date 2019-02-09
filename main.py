from builtins import dict
import random

def load_data(path):
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

def augment(data, sources, use_morph = False, use_lemma = False):
    """
    Given a input dataset, augment it by randomly selecting alternative
    words (based on lemma and POS) in one or several sources.
    :param data: input dataset to augment
    :param sources: sources to use for augmentation
    :param use_lemma: whether to account for lemma (implemented only with use_morph)
    :param use_morph: whether to account for morphological information
    :return: augmented dataset
    """
    augm = []
    alternatives = get_alternatives(sources, use_lemma = use_lemma, use_morph = use_morph)
    # and a fallback, for when there is not alternative in sources
    fallbacks = get_alternatives(data, use_lemma = use_lemma, use_morph = use_morph)
    for row in data:
        if use_morph == False:
            augm.append(random.sample(alternatives[row[2]], 1))

        if use_morph == True and use_lemma == False:
            augm.append(random.sample(alternatives[row[2]][row[3]], 1))

        if use_morph == True and use_lemma == True:
            # check if morph and lemma key exist in dic
            if row[3] in alternatives[row[2]] and row[1] in alternatives[row[2]][row[3]]:
                augm.append(random.sample(alternatives[row[2]][row[3]][row[1]], 1))
            # or use the fallback
            else:
                augm.append(random.sample(fallbacks[row[2]][row[3]][row[1]], 1))

    return augm

def get_alternatives(sources, use_morph = False, use_lemma = False):
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
        if use_morph == False:
            alternatives[tag] = {tuple(r[0:3]) for r in sources if r[2] == tag}

        if use_morph == True:
            alternatives[tag] = dict()
            entries = {tuple(r) for r in sources if r[2] == tag}
            morph = {i[3] for i in entries}
            for m in morph:
                if use_lemma == False:
                    alternatives[tag][m] = {tuple(r) for r in entries if r[3] == m}

                if use_lemma == True:
                    alternatives[tag][m] = dict()
                    entriesMorph = {tuple(r) for r in entries if r[3] == m}
                    lemmata = {i[1] for i in entriesMorph}
                    for l in lemmata:
                        alternatives[tag][m][l] = {tuple(r) for r in entriesMorph if r[1] == l}

    return alternatives


if __name__ == '__main__':

    data = load_data('data/data_morph.tsv')
    sources = load_data('data/sources-AN.tsv')

    augm = augment(data, sources, use_morph=True, use_lemma=True)

    with open ('data/out.tsv', 'w') as out:
        if use_morph == True:
            for line in augm:
                out.write('{}\t{}\t{}\t{}\n'.format(*line[0]))
