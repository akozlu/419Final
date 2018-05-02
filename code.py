import numpy as np
import pandas as pd
import math
import matplotlib.pyplot as plt
import seaborn as sb
import sklearn
import matplotlib.pyplot as plt
def load_files(file_path):
    categories = pd.read_csv(file_path)
    categories.set_index('category_id', inplace=True)
    return categories

categories = pd.read_csv('similar-staff-picks-challenge-categories.csv')
categories.set_index('category_id', inplace=True)

print(categories.head())
print(categories.describe())

clip_categories = pd.read_csv('similar-staff-picks-challenge-clip-categories.csv')
print(clip_categories.describe())

def match_categories (nums) :
    num_list = nums.split(', ')
    category_string = ""
    num_list_iter = iter(num_list)
    for num in range(len(num_list)):
        if (next(num_list_iter, None) != None):
            category_string += categories.loc[next(num_list_iter)]['name'] + ', '
        else:
            category_string += categories.loc[next(num_list_iter)]['name']
    return category_string


clip_data = pd.read_csv('similar-staff-picks-challenge-clips.csv', encoding = "ISO-8859-1")
print(clip_data.iloc[:, 1].equals(clip_data.iloc[:, 7]))


# counter = 0
#
# for i in range(len(clip_data)):
#     if (clip_data.iloc[i, 1] == clip_data.iloc[i, 7]):
#         continue
#     else:
#         counter = counter + 1
#
# print(counter)

