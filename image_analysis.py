# If importing images for the first time, save them as jpegs into folder (no need to right now since we already have them)
# from PIL import Image

# import os
# directory = os.chdir("/Users/christianduffydeabreu/Documents/UPenn/Senior Spring/CIS 519/Project/")
# for thumbnail in os.listdir(directory):
#    if '.DS_Store' in thumbnail:
#        continue
#    im = Image.open(thumbnail).convert("RGB")
#    im.save("/Users/christianduffydeabreu/Documents/UPenn/Senior Spring/CIS 519/Project /thumbnails_jpg/"
#            + thumbnail[:-5] + ".jpg", "jpeg")
# %%Converting pics to gracyscale
import os
import matplotlib.image as mpimg
import numpy as np
import matplotlib.pyplot as plt
import skimage.feature as feature
import csv
import pandas as pd
from scipy.misc import imread
from scipy.linalg import norm
from scipy import sum, average
import numpy as np
from sklearn.cluster import KMeans
import operator
from itertools import *


# %% Saving Images to New Gray Form
def rgb2gray(rgb):
    return np.dot(rgb[..., :3], [0.299, 0.587, 0.114])


gray_images = []
reg_images = []
# Saving Gray Images to List
dir_path = "/Users/christianduffydeabreu/Documents/UPenn/Senior Spring/CIS 519/Project /thumbnails_jpg"
# dir_path = os.path.dirname(os.path.realpath('__file__'))
count = 0
for image in os.listdir(dir_path):
    if '.DS_Store' in image:
        continue
    img = mpimg.imread(image)
    gray_images.append(list(rgb2gray(img)))
    reg_images.append(image)  # names of files
    count += 1
    print(count)

# Saving Gray Images to CSV in Directory --> not working right now
with open('gray_ims.csv', "w") as output:
    writer = csv.writer(output, lineterminator='\n')
    writer.writerows(gray_images)
# %% Reshaping Features for Distance Measure (Slightly Different Processing)
img_converted = []
gray_ims = pd.read_csv('gray.csv')
gray_ims = gray_ims.iloc[:, 1]
for i in range(len(gray_images)):
    image = mpimg.imread(gray_images[i])
    img_converted.append(feature.hog(image))
    print(i)


# %% PCA Min Transform

# %%
###############################################################################


#                            Interacting With Images                          #


###############################################################################
# %% Comparitive Functions
def normalize(arr):
    rng = arr.max() - arr.min()
    amin = arr.min()
    return (arr - amin) * 255 / rng


def compare_images(img1, img2):
    # normalize to compensate for exposure difference, this may be unnecessary
    # consider disabling it
    img1 = normalize(img1)
    img2 = normalize(img2)
    # calculate the difference and its norms
    diff = img1 - img2  # elementwise for scipy arrays
    m_norm = sum(abs(diff))  # Manhattan norm
    z_norm = norm(diff.ravel(), 0)  # Zero norm
    return (m_norm, z_norm)


# %% Running Clustering taking euclidean distance in space
# Clustering
clustermdl = KMeans(n_clusters=90, random_state=40)
clustermdl.fit(gray_images)


# %% Taking Pairwise Distance Between Images
def compare_all(clip_id, images, k):
    m_norm = []
    z_norm = []
    test_image = images.index(clip_id)
    for item in enumerate(images):
        if item == test_image:
            continue
        m_norm, z_norm = compare_images(test_image, item)

    dist_m = pd.DataFrame.from_items(m_norm)
    dist_z = pd.DataFrame.from_items(z_norm)
    frames = [dist_m dist_z]
    scores = pd.concat(frames)
    return top_k, scores



























