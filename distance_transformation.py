from sklearn.feature_extraction.text import TfidfVectorizer
import feature_extraction as ft
import other_features as ot
import numpy as np
import math
from sklearn.decomposition import TruncatedSVD
import matplotlib.pyplot as plt
from mpl_toolkits.mplot3d import Axes3D
from text_analysis import transform_caption
from sklearn.manifold import TSNE
import pickle

def extract_data_with_thumbid(path):
    """ Extract data as formatted dataframe"""
    print(1)
    data = ft.load_whole_file(path, "similar-staff-picks-challenge-clip-categories.csv", "similar-staff-picks-challenge-categories.csv")
    print(1)
    data.drop(['Unnamed: 0'], axis=1, inplace=True)
    print(1)
    data = ot.strdate_to_int(data)
    print(1)
    data["thumb_id"] = data["thumbnail"].apply(lambda x: x.split("_")[0].split("/")[-1])
    print(1)
    return data
    
def tf_idf_captions(df):
    """ Return Sparse Matrix with TF IDF for each clip caption"""
    df = transform_caption(df)
    captions = df["caption"].tolist()
    captions = [x if isinstance(x,str) else "" for x in captions]
    tfidf_vect = TfidfVectorizer(stop_words = 'english')
    tfidf_captions = tfidf_vect.fit_transform(captions)   
    return tfidf_captions

def vectorize_images(pickle_file, df):
    """ Return Ordered (consistent with df) Matrix of Vectorized Images"""
    file = open(pickle_file, 'rb')
    images = pickle.load(file)
    
    vect_img_mtx = []
    for i in range(len(df)):
        thumb = str(df["thumb_id"][i])
        for item in images:
            if item[0] == thumb:
                vect_img_mtx += [list(map(float,item[1]))]
                
    return vect_img_mtx

#"""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""

# Truncated SVD

#"""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""

## Truncated SVD
#svd_dp = TruncatedSVD(n_components = 1500)
#test = svd_dp.fit_transform(tfidf_captions)

#"""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""

# Image PCA

#"""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""

# Import Vectorized Images
data = extract_data_with_thumbid("similar-staff-picks-challenge-clips.csv")
images = vectorize_images('vect_images', data)

# TSNE Clustering
#tsne = TSNE(n_components = 2, verbose = 1, perplexity = 40, n_iter = 300)
