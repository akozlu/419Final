from PIL import Image
import matplotlib.image as mpimg

for thumbnail in os.listdir(directory):
    if '.DS_Store' in thumbnail:
        continue
    im = Image.open(thumbnail).convert("RGB")
    im.save("/Users/christianduffydeabreu/Documents/UPenn/Senior Spring/CIS 519/Project /thumbnails_jpg/" + thumbnail[
                                                                                                            :-5] + ".jpg",
            "jpeg")


# %%Converting pics to gracyscale


def rgb2gray(rgb):
    return np.dot(rgb[..., :3], [0.299, 0.587, 0.114])


gray_images = []
for image in os.listdir("/Users/christianduffydeabreu/Documents/UPenn/Senior Spring/CIS 519/Project /thumbnails_jpg/"):
    if '.DS_Store' in image:
        continue
    img = mpimg.imread(image)
    gray_images.append(rgb2gray(img))

# %% Reshaping
import matplotlib.image as mpimg
import skimage.feature as feature

img_converted = []
# for idx, image in enumerate(os.listdir("/Users/christianduffydeabreu/Documents/UPenn/Senior Spring/CIS 519/Project /thumbnails_jpg/")):
for i, image in enumerate(gray_images):
    img_converted.append(feature.hog(image))
    print(i)

# %% PCA Min Transform
from scipy.misc import imread
from scipy.linalg import norm
from scipy import sum, average


# %% Normalizing images
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
import numpy as np
from sklearn.cluster import KMeans
import operator
# from scipy.stats import nanmean
from itertools import *

clustermdl = KMeans(n_clusters=90, random_state=40)
clustermdl.fit(gray_images)
