import numpy as np
import pandas as pd


def load_train_and_test_files(file_path):
    categories = pd.read_csv(file_path, encoding="ISO-8859-1")
    # categories.set_index('category_id', inplace=True)
    np.random.seed(seed=0)
    indices = np.random.rand(len(categories)) < 0.8
    train = categories[indices]
    test = categories[~indices]
    return [train, test]


class PandaFrames(object):
    def __init__(self, filepath):
        self.pandaframes = load_train_and_test_files(filepath)

    def get_train_file(self):
        train = self.pandaframes[0]
        train = train.reset_index()
        train = train.drop(['Unnamed: 0', 'index'], axis=1)
        return train

    def get_test_file(self):
        test = self.pandaframes[1]
        test = test.reset_index()
        test = test.drop(['Unnamed: 0', 'index'], axis=1)
        return test


pdf = PandaFrames('similar-staff-picks-challenge-clips.csv')