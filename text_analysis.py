import numpy as np
import pandas as pd
from ccg_nlpy import remote_pipeline
import nltk
from nltk import pos_tag
from nltk.stem import WordNetLemmatizer
from nltk.corpus import stopwords
from nltk.corpus import wordnet as wn
import re
import string
import itertools
import code

# # UNCOMMENT THE TWO LINES BELOW TO RUN FOR THE FIRST TIME
# nltk.download('wordnet')
# nltk.download('stopwords')
# nltk.download('averaged_perceptron_tagger')

# Initialize WordNetLemmatizer
lemmatizer = WordNetLemmatizer()
# Set of stopwords
stopWords = set(stopwords.words('english'))
# Regex to remove punctuation
regex = re.compile('[%s]' % re.escape(string.punctuation))

def tokenize(caption):
    words = []

    caption_array = regex.sub('', caption)
    caption_array = ' '.join(word for word in caption_array.split() if word not in stopWords)
    data = caption_array.split(" ")

    for word in data:
        # Make word lower-case and append lemmatized word
        words.append(lemmatizer.lemmatize(word.lower()))

    return words

def transform_caption(dataframe):
    dataframe['tokenized caption'] = ""
    for i in range(len(dataframe)):
        dataframe.at[i, 'tokenized caption'] = tokenize(str(dataframe.at[i, 'caption']))
    return dataframe

def synset_pos_tag(tag):
    if tag.startswith('V'):
        return 'v'
    elif tag.startswith('N'):
        return 'n'
    elif tag.startswith('R'):
        return 'r'
    elif tag.startswith('J'):
        return 'a'
    else:
        return None

def word_to_synset(word, pos):
    tag = synset_pos_tag(pos)
    if tag is None:
        return None
    else:
        sets = wn.synsets(word, tag)
        if sets is []:
            return None
        else:
            return wn.synsets(word, tag)[0]

def path_sim(s1, s2):
    if (s1.pos() == s2.pos()):
        return s1.path_similarity(s2)
    else:
        return 0

# Try using all synsets for each word instead of just one (using wn.synsets('word'))
def sentence_similarity(sentence1, sentence2):

    postag1 = nltk.pos_tag(sentence1)
    postag2 = nltk.pos_tag(sentence2)

    synsets1 = [word_to_synset(*word) for word in postag1]
    synsets2 = [word_to_synset(*word) for word in postag2]

    synsets1 = [word for word in synsets1 if word]
    synsets2 = [word for word in synsets2 if word]

    similarity1 = 0
    num_words1 = 0
    similarity2 = 0
    num_words2 = 0

    for set1 in synsets1:
        top_similarity = np.max([path_sim(set1, set2) for set2 in synsets2])
        if top_similarity is not None:
            similarity1 += top_similarity
            num_words1 += 1

    for set2 in synsets2:
        top_similarity = np.max([path_sim(set2, set1) for set1 in synsets1])
        if top_similarity is not 0:
            similarity2 += top_similarity
            num_words2 += 1

    return ((similarity1 / num_words1) + (similarity2 / num_words2)) / 2


train = code.pdf.get_train_file()
test = code.pdf.get_test_file()

train = transform_caption(train)

print(sentence_similarity(['Hi', 'there', 'this', 'is', 'cool'], ['Hello', 'I', 'am', 'an', 'idiot']))
