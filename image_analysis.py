#!/usr/bin/env python3
# -*- coding: utf-8 -*-
'''
What you need in your directory:
    similar-similar-staff-picks-challenge-clips.csv

Running Code Will:
    create two subdirectories: WebP_Files nad JPG_Files containing all images
    from urls scraped.
'''
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
import pandas as pd
import os
import sys

data = pd.read_csv("similar-staff-picks-challenge-clips.csv")
clips = data[['id', 'thumbnail']]
# %% Scraping List of Urls
try:
    from urlparse import urljoin
except ImportError:
    from urllib.parse import urljoin
import requests
from bs4 import BeautifulSoup


class Scraper:
    def __init__(self):
        self.visited = set()
        self.session = requests.Session()
        self.session.headers = {
            "User-Agent": "Mozilla/5.0 (Macintosh; Intel Mac OS X 10_12_5) AppleWebKit/537.36 (KHTML, like Gecko) Chrome/59.0.3071.109 Safari/537.36"}
        requests.packages.urllib3.disable_warnings()  # turn off SSL warnings

    def visit_url(self, url, level):
        print(url)
        if url in self.visited:
            return

        self.visited.add(url)

        content = self.session.get(url, verify=False).content
        soup = BeautifulSoup(content, "lxml")

        for img in soup.select("img[src]"):
            image_url = img["src"]
            if not image_url.startswith(("data:image", "javascript")):
                self.downlzoad_image(urljoin(url, image_url))

        if level > 0:
            for link in soup.select("a[href]"):
                self.visit_url(urljoin(url, link["href"]), level - 1)

    def download_image(self, image_url):
        local_filename = image_url.split('/')[-1].split("?")[0]
        r = self.session.get(image_url, stream=True, verify=False)
        newpath = dir_path + "/WebP_Files/"
        if not os.path.exists(newpath):
            os.makedirs(newpath)
        with open(newpath + local_filename, 'wb') as f:
            for chunk in r.iter_content(chunk_size=1024):
                f.write(chunk)


if __name__ == '__main__':
    dir_path = os.path.dirname(os.path.realpath('__file__'))
    for idx, image in enumerate(clips['thumbnail']):
        scraper = Scraper()
        scraper.visit_url(image, 1)
        scraper.download_image(image)
# %%Importing them all as jpg
from PIL import Image

dir_path = os.path.dirname(os.path.realpath('__file__'))
newpath = dir_path + "/WebP_Files/"
jpgpath = dir_path + "/JPG_Files/"

if not os.path.exists(jpgpath):
    os.makedirs(jpgpath)

os.chdir(newpath)
jpgpath = dir_path + "/JPG_Files/"
for thumbnail in os.listdir(newpath):
    if '.DS_Store' in thumbnail:
        continue
    im = Image.open(thumbnail).convert("RGB")
    im.save(jpgpath + thumbnail[:-5] + ".jpg", "jpeg")
os.chdir(dir_path)
# %%Converting pics to grayscale
os.chdir(jpgpath)
import matplotlib.image as mpimg


def rgb2gray(rgb):
    return np.dot(rgb[..., :3], [0.299, 0.587, 0.114])


gray_images = []
rgb_images = []
for image in os.listdir(jpgpath):
    if '.DS_Store' in image:
        continue
    img = mpimg.imread(image)
    rgb_images.append(img)
    gray_images.append(rgb2gray(img))

# %% Reshaping Image To New Form
import matplotlib.image as mpimg
import skimage.feature as feature

img_converted_gray = []
img_converted_rgb = []
for i, image in enumerate(gray_images):
    img_converted_gray.append(feature.hog(image))
    print(i)
for k, image in enumerate(rgb_images):
    img_converted_rgb.append(feature.hog(image))


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



























