import cv2
import os
from tqdm import tqdm
from skimage.filters import threshold_sauvola
import numpy as np


input_dir = "data/raw/pages/albini1549"
gsc_dir = "/data01/manuscriptrecognition/data/processed/gsc/albini1549"
bzd_dir = "/data01/manuscriptrecognition/data/processed/bzd/albini1549"
ctr_dir = "/data01/manuscriptrecognition/data/processed/ctr/albini1549"

for base_dir in [gsc_dir, bzd_dir, ctr_dir]:
    os.makedirs(base_dir, exist_ok=True)

for filename in sorted(os.listdir(input_dir)):
    if filename.endswith(".png"):
        filepath = os.path.join(input_dir, filename)
        print(filepath)
        #grayscale
        gsc_img = cv2.imread(filepath,cv2.IMREAD_GRAYSCALE)
        if gsc_img is None:
            print(f"Ci sono stati casini leggendo {filepath}")
            continue

        gsc_filename = "gsc_" + filename
        gsc_filepath = os.path.join(gsc_dir, gsc_filename)
        cv2.imwrite(gsc_filepath, gsc_img)

        thresh_sauv = threshold_sauvola(gsc_img, window_size=61, k=0.2)
        bzd_img = (gsc_img > thresh_sauv).astype(np.uint8) * 255
        kernel = np.ones((2, 2), np.uint8)
        bzd_img = cv2.morphologyEx(bzd_img, cv2.MORPH_OPEN, kernel)  # elimina piccoli punti bianchi
        bzd_img = cv2.morphologyEx(bzd_img, cv2.MORPH_CLOSE, kernel) # chiude piccoli buchi nel testo

        bzd_filename = "bzd_" + filename
        bzd_filepath = os.path.join(bzd_dir, bzd_filename)
        cv2.imwrite(bzd_filepath, bzd_img)

        #countour!
        blurred = cv2.GaussianBlur(gsc_img, (5, 5), 0)
        edges = cv2.Canny(blurred, 50, 150)
        edges_inv = cv2.bitwise_not(edges)

        ctr_filename = "ctr_" + filename
        ctr_filepath = os.path.join(ctr_dir, ctr_filename)
        cv2.imwrite(ctr_filepath, edges_inv)
