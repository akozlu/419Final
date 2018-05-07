from sklearn.feature_extraction.text import TfidfVectorizer
import feature_extraction as ft
import other_features as ot
import numpy as np
import math
from sklearn.decomposition import TruncatedSVD
from sklearn.decomposition import PCA
import matplotlib.pyplot as plt
from mpl_toolkits.mplot3d import Axes3D
from text_analysis import transform_caption
from sklearn.manifold import TSNE
from sklearn.preprocessing import StandardScaler
import pickle

class FeatureSpace(object):
    def __init__(self, data_file, pickled_images):
        print("Extracting Data")
        self.data = extract_data_with_thumbid(data_file)
        print("Extracting Vectorized Images")
        self.images = vectorize_images(pickled_images, self.data)
        
    def generate_text_vector(self, n_components):
        """ Transform Text to (n_clips x n_components) Matrix"""
        print("Generating Text Vector")
        print("        Generating TF IDF Caption Vector")
        tfidf_captions = tf_idf_captions(self.data)
        print("        Performing SVD on Sparse TF IDF Matrix")
        svd_dp = TruncatedSVD(n_components = n_components)
        self.text_vect = svd_dp.fit_transform(tfidf_captions)
        return self.text_vect
        
    def generate_misc_vector(self, features):
        """ Create Matrix of Other Features"""
        print("Generating Other Features Vector")
        misc_vect = [np.array([row[feature] for feature in features]) for idx, row in self.data.iterrows()]
        self.misc_vect = np.array(misc_vect)
        return self.misc_vect
    
    def generate_img_vector(self, n_tsne, pca, n_pca = 10):
        """ Transform Images to (n_clips x n_tsne) Matrix"""
        print("Generating Image Vector")
        images = self.images
        # Perform PCA prior to t-SNE
        if pca:
            print("       Performing PCA prior to t-SNE")
            sc = StandardScaler()
            images = sc.fit_transform(self.images)
            pca_dp = PCA(n_components=n_pca)
            images = pca_dp.fit_transform(images)
        print("       Performing t-SNE")
        # Perform t-SNE
        tsne_clt = TSNE(n_components = n_tsne, verbose = 1, perplexity = 40, n_iter = 300)
        tsne_predict = tsne_clt.fit_transform(images)
        self.img_vect = tsne_predict
        return self.img_vect
    
    def generate_final_vector(self, n_final_tsne, n_pca_text, misc_features, pca_img, n_pca_img,n_tsne_img):
        """ Generates Text, Image, Misc Vectors and Perform Final t-SNE"""
        print("Generating Transformed Feature Space")
        img_vec = self.generate_img_vector(n_tsne = n_tsne_img, pca = pca_img, n_pca = n_pca_img)
        txt_vec = self.generate_text_vector(n_components = n_pca_text)
        misc_vec = self.generate_misc_vector(features = misc_features)
        self.test = zip(txt_vec, misc_vec, img_vec)
        final_vect = [np.concatenate([txt,misc,img]) for txt,misc,img in zip(txt_vec, misc_vec, img_vec)]
        # t-SNE
        print("Performing Final t-SNE")
        tsne_clt = TSNE(n_components = n_final_tsne, verbose = 1, perplexity = 40, n_iter = 300)
        self.final_vect = tsne_clt.fit_transform(final_vect)
        return self.final_vect
        
    def plot_fspace_3d(self, fspace):
        """ Plot any Intermediate or Final Feature Space"""
        """ fspace in ['text', 'misc', 'img', 'final']"""
        if fspace == 'text': plot_vect = self.text_vect
        if fspace == 'misc': plot_vect = self.misc_vect
        if fspace == 'img': plot_vect = self.img_vect
        if fspace == 'final': plot_vect = self.final_vect
        fig = plt.figure()
        ax = fig.add_subplot(111, projection='3d')
        ax.scatter(plot_vect[:,0],plot_vect[:,1],plot_vect[:,2])
        
def extract_data_with_thumbid(path):
    """ Extract data as formatted dataframe"""
    data = ft.load_whole_file(path, "similar-staff-picks-challenge-clip-categories.csv", "similar-staff-picks-challenge-categories.csv")
    data.drop(['Unnamed: 0'], axis=1, inplace=True)
    data = ot.strdate_to_int(data)
    data["thumb_id"] = data["thumbnail"].apply(lambda x: x.split("_")[0].split("/")[-1])
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

#ft = FeatureSpace("similar-staff-picks-challenge-clips.csv", 'vect_images')
