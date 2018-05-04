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
import feature_extraction

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

# Function to tokenize a caption
def tokenize(caption):
    words = []

    caption_array = regex.sub('', caption)
    caption_array = ' '.join(word for word in caption_array.split() if word not in stopWords)
    data = caption_array.split(" ")

    for word in data:
        # Make word lower-case and append lemmatized word
        words.append(lemmatizer.lemmatize(word.lower()))

    return words


# Function to add the 'tokenized caption' column, containing the tokenized caption, to a dataframe
def transform_caption(dataframe):
    dataframe['tokenized caption'] = ""
    for i in range(len(dataframe)):
        dataframe.at[i, 'tokenized caption'] = tokenize(str(dataframe.at[i, 'caption']))
    return dataframe


# Function to convert the Penn part of speech tag to the WordNet synset part of speech tag
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


# Function to retrieve the WordNet synset given a word
def word_to_synset(word, pos):
    tag = synset_pos_tag(pos)
    if tag is None:
        return None
    else:
        sets = wn.synsets(word, tag)
        if sets == []:
            return None
        elif sets is None:
            return None
        else:
            return sets[0]


# Function to calculate the path similarity given two synsets
def path_sim(s1, s2):
    # Need two synsets to have same part of speech to calculate path similarity
    if (s1.pos() == s2.pos()):
        return s1.path_similarity(s2)
    else:
        return 0


# Function to calculate the Wu-Palmer similarity given two synsets
def wup_sim(s1, s2):
    # Need two synsets to have same part of speech to calculate Wu-Palmer similarity
    if (s1.pos() == s2.pos()):
        return s1.wup_similarity(s2)
    else:
        return 0


# Function to calculate the Leacock-Chodorow similarity given two synsets
def lch_sim(s1, s2):
    # Need two synsets to have same part of speech to calculate Leacock-Chodorow similarity
    if (s1.pos() == s2.pos()):
        return s1.lch_similarity(s2)
    else:
        return 0


# Try using all synsets for each word instead of just one (using wn.synsets('word'))
# The argument 'f_sim' should be one of [path_sim(), lch_sim(), wup_sim()]
def sentence_similarity(sentence1, sentence2, f_sim):
    postag1 = pos_tag(sentence1)
    postag2 = pos_tag(sentence2)

    synsets1_init = [word_to_synset(*word) for word in postag1]
    synsets2_init = [word_to_synset(*word) for word in postag2]

    synsets1 = [word for word in synsets1_init if word is not None]
    synsets2 = [word for word in synsets2_init if word is not None]

    similarity_main = 0
    num_words_main = 0
    similarity2 = 0
    num_words2 = 0

    if len(synsets1) < len(synsets2):
        for set1 in synsets1:
            sim_array = []
            for set2 in synsets2:
                sim12 = f_sim(set1, set2)
                if sim12 is not None:
                    sim_array.append(sim12)
            top_similarity = np.max(sim_array)
            if top_similarity is not None:
                similarity_main += top_similarity
            num_words_main += 1

        return similarity_main / num_words_main

    elif len(synsets1) > len(synsets2):
        for set2 in synsets2:
            sim_array = []
            for set1 in synsets1:
                sim21 = f_sim(set2, set1)
                if sim21 is not None:
                    sim_array.append(sim21)
            top_similarity = np.max(sim_array)
            if top_similarity is not None:
                similarity_main += top_similarity
            num_words_main += 1

        return similarity_main / num_words_main

    else:
        for set1 in synsets1:
            sim_array = []
            for set2 in synsets2:
                sim12 = f_sim(set1, set2)
                if sim12 is not None:
                    sim_array.append(sim12)
            top_similarity = np.max(sim_array)
            if top_similarity is not None:
                similarity_main += top_similarity
            num_words_main += 1
        for set2 in synsets2:
            sim_array = []
            for set1 in synsets1:
                sim21 = f_sim(set2, set1)
                if sim21 is not None:
                    sim_array.append(sim21)
            top_similarity2 = np.max(sim_array)
            if top_similarity2 is not None:
                similarity2 += top_similarity2
            num_words2 += 1

        return (similarity_main / num_words_main + similarity2 / num_words2) / 2


# The argument 'f_sim' should be one of [path_sim(), lch_sim(), wup_sim()]
def caption_similarity(df, clip_id, f_sim):
    # df = transform_caption(df)
    df['caption path similarity'] = 0.0
    clip_index = df[df['id'] == clip_id].index[0]
    target_clip = df[df['id'] == clip_id]
    df = df.drop([clip_index]).reset_index.drop(columns=['index'])
    df['caption path similarity'] = [sentence_similarity(target_clip.at[clip_index, 'tokenized caption'],
                                        df.at[i, 'tokenized caption']) for i in range(len(df),
                                        f_sim)]
    df = df.sort_values(by=['caption path similarity'], ascending=False).reset_index().drop(columns=['index'])
    return df


# Function to find accuracy with categories as labels and threshold for 1 vs 0. Iterates over every id in the df.
# The argument 'f_sim' should be one of [path_sim(), lch_sim(), wup_sim()]
def calculate_accuracy(df, thresh, top_k, f_sim):
    accuracies = 0
    for id in df['id']:
        sim_df = caption_similarity(df, id, f_sim)
        correct = 0
        for i in range(top_k):
            if sim_df.at[i, 'category'] == df.at[id, 'category']:
                correct += 1
        if correct >= thresh:
            accuracies += 1
    return accuracies / len(df)


pdf = feature_extraction.PandaFrames('similar-staff-picks-challenge-clips.csv')
train = transform_caption(pdf.get_train_file())
test = transform_caption(pdf.get_test_file())

train = caption_similarity(train, 214566929, path_sim())
print(train.head())

# print(sentence_path_similarity(['Hi', 'my', 'name', 'is', 'Nazih', 'and', 'I', 'like', 'to', 'code'],
#                                ['The', 'kid', 'named', 'Jesus', 'ate', 'the', 'apple', 'and', 'loved', 'it']))
# print(sentence_path_similarity(['The', 'kid', 'named', 'Jesus', 'ate', 'the', 'apple', 'and', 'loved', 'it'],
#                                ['Hi', 'my', 'name', 'is', 'Nazih', 'and', 'I', 'like', 'to', 'code']))