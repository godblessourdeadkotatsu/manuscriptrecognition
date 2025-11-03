import cv2
import numpy as np
import os

rng = np.random.default_rng(100)
bzd_path = "data/processed/bzd/BerolPhill1516/bzd_page-023.png"  
gsc_path = "data/processed/gsc/BerolPhill1516/gsc_page-023.png"
ctr_path = "data/processed/ctr/BerolPhill1516/ctr_page-023.png"
bzd_dir = "data/patches/bzd"
ctr_dir = "data/patches/ctr"
gsc_dir = "data/patches/gsc"

os.makedirs(bzd_dir, exist_ok=True)
os.makedirs(ctr_dir, exist_ok=True)
os.makedirs(gsc_dir, exist_ok=True)

PATCH_SIZE = 512       # dimensione patch (in pixel)
N_PATCHES = 20         # quante patch estrarre
MIN_BLACK_RATIO = 0.105 # minimo  di segnale in percentuale

img = cv2.imread(bzd_path, cv2.IMREAD_UNCHANGED)
if img is None:
    raise FileNotFoundError(f"casini vari, non sono riuscito a leggere {bzd_path}")

h, w = img.shape
print(f"Immagine caricata: {w}×{h}")

successes = 0
trials = 0


while successes < N_PATCHES and trials < N_PATCHES * 100:
    y = rng.integers(0, h - PATCH_SIZE)
    x = rng.integers(0, w - PATCH_SIZE)
    patch = img[y:y+PATCH_SIZE, x:x+PATCH_SIZE]

    #scarta patch troppo bianche
    black_ratio = (patch < 128).mean()
    if black_ratio < MIN_BLACK_RATIO:
        continue
    successes += 1

    bzd_out = os.path.join(bzd_dir, f"patch_{successes:03d}.png")
    cv2.imwrite(bzd_out, patch)

    #ora dobbiamo usare la stessa patch per tirare fuoir le immagini da grayscale e countour

    gsc_img = cv2.imread(gsc_path, cv2.IMREAD_UNCHANGED)
    patch = gsc_img[y:y+PATCH_SIZE, x:x+PATCH_SIZE]

    gsc_out = os.path.join(gsc_dir, f"patch_{successes:03d}.png")
    cv2.imwrite(gsc_out, patch)

    ctr_img = cv2.imread(ctr_path, cv2.IMREAD_UNCHANGED)
    patch = ctr_img[y:y+PATCH_SIZE, x:x+PATCH_SIZE]

    ctr_out = os.path.join(ctr_dir, f"patch_{successes:03d}.png")
    cv2.imwrite(ctr_out, patch)
    

print(f"Estratte {successes} patch salvate")