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

#Importing Data
data = extract_data_with_thumbid("similar-staff-picks-challenge-clips.csv")
images = vectorize_images('vect_images', data)
dir_path = os.path.dirname(os.path.realpath('__file__'))
webppath = dir_path + "/WebP_Files/"
jpgpath = dir_path + "/samples/"

def image_list(color):
    image_list = []
    
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
            
    return(scores_df, top_k)
#%%
#Example:
clip_id = data['id'][2822]
k = 10
mode1 = 'euclidean'
show = True
a, b = image_similarity(clip_id = clip_id, mode = mode1, k = k, show = True)

    


