#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Sun May  6 20:16:06 2018

@author: christianduffydeabreu
"""
from sklearn.feature_extraction.text import TfidfVectorizer
import feature_extraction as ft
import other_features as ot
import numpy as np
import math
import matplotlib.pyplot as plt
import pickle
import os
import matplotlib.image as mpimg
import pandas as pd
from distance_transformation import *


#Importing Data
all_data = ft.load_whole_file("similar-staff-picks-challenge-clips.csv", "similar-staff-picks-challenge-clip-categories.csv", "similar-staff-picks-challenge-categories.csv")             
data = extract_data_with_thumbid("similar-staff-picks-challenge-clips.csv")

#Removing those without Columns
def remove_empty_categories(data): 
    data = data.reindex(index=data.index[::-1])
    rows_to_delete= []
    for i in range(len(data)):
        if data['main categories'].values[i] == []:
            rows_to_delete.append(i)
    data.drop(data.index[rows_to_delete], inplace = True)
    data.reset_index(inplace = True)
    return(data)
all_data = remove_empty_categories(all_data)
data = remove_empty_categories(data)


images = vectorize_images('vect_images', data)
dir_path = os.path.dirname(os.path.realpath('__file__'))
webppath = dir_path + "/WebP_Files/"
jpgpath = dir_path + "/samples/"

def image_list(color):
    image_list = []
    ''' Saving Images to List '''
    for image in os.listdir(jpgpath):
        if '.DS_Store' in image: #This may sometimes be found in a folder preventing uploads
            continue
        img = mpimg.imread(image)
        if color == "gray":
            image_list.append(rgb2gray(img))
        elif color == "rgb":
            image_list.append(img)
    return(image_list)
os.chdir(jpgpath)
rgb_images = image_list('rgb')

def image_similarity(clip_id, p = 3, all_images = rgb_images, data = data, images = images, mode = 'euclidean' , k = 10, show = False):
    d = []
    test_image_index = np.where(data['id'] == clip_id)[0][0]
    test_image = images[test_image_index]
    
    #Comparing To Every Other Image:
    for idx, item in enumerate(images):
        if idx == test_image_index:
            d.append(0)
            continue
        temp = ot.calculate_distance(test_image, item, mode, p)
        d.append(temp)
    ddf = pd.DataFrame(d, columns = ['Distance'])
    scores_df  = data
    scores_df['Distance'] = d
    top_k= scores_df.sort_values(by=['Distance'])[1:k]
    
    
    
    #Show for plotting
    if show == True:
        indices = top_k.index.get_values()
        for index in indices:
            plt.figure()
            plt.imshow(rgb_images[index])
    plt.figure()
    plt.imshow(rgb_images[np.where(data['id'] == clip_id)[0][0]])
    plt.title('Original Image')
    return(top_k)
#%%
#Example:
clip_id = data['id'][1182]
k = 10
mode1 = 'euclidean'
top_clips = image_similarity(clip_id = clip_id, mode = mode1, k = k, show = False)


    


    


