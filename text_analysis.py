from ccg_nlpy import remote_pipeline
import nltk
from nltk.stem import WordNetLemmatizer
from nltk.corpus import stopwords
import re
import string
# # UNCOMMENT THE TWO LINES BELOW TO RUN FOR THE FIRST TIME
# nltk.download('wordnet')
# nltk.download('stopwords')
import code

# Initialize WordNetLemmatizer
lemmatizer = WordNetLemmatizer()
# Set of stopwords
stopWords = set(stopwords.words('english'))
# Regex to remove punctuation
regex = re.compile('[%s]' % re.escape(string.punctuation))

# Alternative tokenization schema
# # Tokenization Schema 1
# rgx = re.compile("(\w[\w']*\w|\w)")
# text_list = rgx.findall(text)

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

train = code.pdf.get_train_file()
test = code.pdf.get_test_file()

train = transform_caption(train)
print(train.head())

# pipeline = remote_pipeline.RemotePipeline(server_api='http://austen.cs.illinois.edu:5800/')
#
# doc = pipeline.doc("Hello, how are you. I am doing fine")
# print(doc.get_lemma)
# print(doc.get_pos)