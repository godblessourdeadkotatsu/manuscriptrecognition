import cv2
import os
from tqdm import tqdm
from skimage.filters import threshold_sauvola
import numpy as np

input_dir = "data/raw/pages"
gsc_dir = "data/processed/gsc"
bzd_dir = "data/processed/bzd"
ctr_dir = "data/processed/ctr"

for base_dir in [gsc_dir, bzd_dir, ctr_dir]:
    os.makedirs(base_dir, exist_ok=True)

# Cammina in tutte le sottocartelle
for root, dirs, files in os.walk(input_dir):
    # Calcola il percorso relativo rispetto alla root
    rel_path = os.path.relpath(root, input_dir)
    # normalizza: se sei nella root stessa, usa stringa vuota (evita '.' come cartella)
    if rel_path == ".":
        rel_sub = ""
    else:
        rel_sub = rel_path

    # Crea le sottocartelle corrispondenti (se rel_sub == "" allora crea la root stessa)
    for target_root in [gsc_dir, bzd_dir, ctr_dir]:
        target_dir = os.path.join(target_root, rel_sub) if rel_sub else target_root
        os.makedirs(target_dir, exist_ok=True)

    # Prendi solo i file immagine (case-insensitive)
    images = sorted([f for f in files if f.lower().endswith(".png")])

    if not images:
        continue

    print(f"\n Processing folder: {rel_path} ({len(images)} images)")

        # Elabora le immagini
    for name in tqdm(images, desc=f"lavorando su {rel_path}"):
        path = os.path.join(root, name)
        try:
            gsc_img = cv2.imread(path, cv2.IMREAD_GRAYSCALE)
            if gsc_img is None:
                print(f"casini vari leggendo {path}")
                continue

            # Build output paths that preserve sottocartella
            gsc_out_dir = os.path.join(gsc_dir, rel_sub) if rel_sub else gsc_dir
            bzd_out_dir = os.path.join(bzd_dir, rel_sub) if rel_sub else bzd_dir
            ctr_out_dir = os.path.join(ctr_dir, rel_sub) if rel_sub else ctr_dir

            # Save grayscale
            gsc_filename = "gsc_" + name
            gsc_filepath = os.path.join(gsc_out_dir, gsc_filename)
            cv2.imwrite(gsc_filepath, gsc_img)

            # Save binarized (Sauvola)
            try:
                thresh_sauv = threshold_sauvola(gsc_img, window_size=61, k=0.2)
                bzd_img = (gsc_img > thresh_sauv).astype(np.uint8) * 255
                kernel = np.ones((2, 2), np.uint8)
                bzd_img = cv2.morphologyEx(bzd_img, cv2.MORPH_OPEN, kernel)
                bzd_img = cv2.morphologyEx(bzd_img, cv2.MORPH_CLOSE, kernel)
            except Exception as e:
                print(f"casini vari durante bzd su {path}: {e}")
                continue

            bzd_filename = "bzd_" + name
            bzd_filepath = os.path.join(bzd_out_dir, bzd_filename)
            cv2.imwrite(bzd_filepath, bzd_img)

            # Save contour (edges)
            blurred = cv2.GaussianBlur(gsc_img, (5, 5), 0)
            edges = cv2.Canny(blurred, 50, 150)
            edges_inv = cv2.bitwise_not(edges)

            ctr_filename = "ctr_" + name
            ctr_filepath = os.path.join(ctr_out_dir, ctr_filename)
            cv2.imwrite(ctr_filepath, edges_inv)

        except Exception as e:
            print(f"casini vari su {path}: {e}")
            continue

