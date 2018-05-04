import numpy as np
import pandas as pd
from urllib.request import urlopen
from bs4 import BeautifulSoup
import urllib.request

def load_whole_file(file_path):
    data = pd.read_csv(file_path, encoding="ISO-8859-1")
    return data


def load_train_and_test_files(file_path):
    data = pd.read_csv(file_path, encoding="ISO-8859-1")
    # categories.set_index('category_id', inplace=True)
    np.random.seed(seed=0)
    indices = np.random.rand(len(data)) < 0.8
    train = data[indices]
    test = data[~indices]
    return [train, test]

def load_clips_categories(file_path):
    clip_categories = pd.read_csv(file_path, encoding="ISO-8859-1")
    return clip_categories


def url_is_alive(url):
    """
    Checks that a given URL is reachable.
    :param url: A URL
    :rtype: bool
    """
    request = urllib.request.Request(url)
    request.get_method = lambda: 'HEAD'

    try:
        urllib.request.urlopen(request)
        return True
    except urllib.request.HTTPError:
        return False


def get_unknown_clip_categories(data_path,categories_path):

    data = load_whole_file(data_path)
    clip_categories = load_clips_categories(categories_path)
    counter  = 0

    for id in data['id']:

        if id not in clip_categories.clip_id.values:
            url = 'https://vimeo.com/' + str(id)
            url_valid = url_is_alive(url)

            if url_valid:  # check if url is valid
               print(id)
               vimeo_webpage = urlopen(url)

               content = vimeo_webpage.read()  # get content from webpage
               soup = BeautifulSoup(content, 'html.parser')
               #print(soup)
               for script in soup("script"):
                   script.extract()
               list_of_scripts = soup.findAll("script")
               print( list_of_scripts)
               #categories = soup.find('div', attrs={'class': "clip-categories"})
               #print(categories)
              # print(categories.get_text(separator=" ", strip=True))
               # counter = counter + 1

    return counter

class PandaFrames(object):
    def __init__(self, filepath):
        self.pandaframes = load_train_and_test_files(filepath)
        self.get_new_captions_for_train_file()
        self.get_new_captions_for_test_file()

    def get_train_file(self):
        train = self.pandaframes[0]
        train = train.reset_index()
        train = train.drop(['Unnamed: 0', 'index'],axis = 1)
        return train

    def get_test_file(self):
        test = self.pandaframes[1]
        test = test.reset_index()
        test = test.drop(['Unnamed: 0', 'index'],axis = 1)
        return test

    def get_new_captions_for_train_file(self):
        train = self.get_train_file()

        for id in train['id']:

            url = 'https://vimeo.com/' + str(id)
            url_valid = url_is_alive(url)

            if url_valid:  # check if url is valid
                vimeo_webpage = urlopen(url)

                content = vimeo_webpage.read()  # get content from webpage
                soup = BeautifulSoup(content, 'html.parser')

                if soup.find('div', attrs={
                    'class': 'clip_details-description description-wrapper iris_desc'}) is not None:
                    # get the video description (all paragraphs instead of first paragraph
                    article_soup = [s.get_text(separator=" ", strip=True) for s in soup.find('div', attrs={
                        'class': 'clip_details-description description-wrapper iris_desc'}).find_all(
                        'p')]

                    index = train.loc[train['id'] == id].index[0]

                    #print(' '.join(article_soup))

                    train.loc[index, 'caption'] = ' '.join(article_soup)  # update the caption of our train dataset

    def get_new_captions_for_test_file(self):
        test = self.get_test_file()

        for id in test['id']:

            url = 'https://vimeo.com/' + str(id)
            url_valid = url_is_alive(url)

            if url_valid:  # check if url is valid
                vimeo_webpage = urlopen(url)

                content = vimeo_webpage.read()  # get content from webpage
                soup = BeautifulSoup(content, 'html.parser')

                if soup.find('div', attrs={
                    'class': 'clip_details-description description-wrapper iris_desc'}) is not None:
                    # get the video description (all paragraphs instead of first paragraph)
                    article_soup = [s.get_text(separator=" ", strip=True) for s in soup.find('div', attrs={
                        'class': 'clip_details-description description-wrapper iris_desc'}).find_all(
                        'p')]

                    index = test.loc[test['id'] == id].index[0]

                    # print(' '.join(article_soup))

                    test.loc[index, 'caption'] = ' '.join(article_soup)  # update the caption of our train dataset



#data  =load_whole_file('similar-staff-picks-challenge-clips.csv')
#print(len(data))

#c_id = load_clips_categories('similar-staff-picks-challenge-clip-categories.csv')
#print (len(c_id))
#get_unknown_clip_categories('similar-staff-picks-challenge-clips.csv','similar-staff-picks-challenge-clip-categories.csv')
