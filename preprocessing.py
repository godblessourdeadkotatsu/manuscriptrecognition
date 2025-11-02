import cv2
import os
from tqdm import tqdm


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

    # Crea le sottocartelle corrispondenti
    for target_root in [gsc_dir, bzd_dir, ctr_dir]:
        os.makedirs(os.path.join(target_root, rel_path), exist_ok=True)

    # Prendi solo i file immagine
    images = sorted([f for f in files if f.endswith(".png")])

    if not images:
        continue

    print(f"\n Processing folder: {rel_path} ({len(images)} images)")

    # Elabora le immagini
    for name in tqdm(images, desc=f"STO LAVORANDO su {rel_path}"):
        path = os.path.join(root, name)

        gsc_path = os.path.join(gsc_dir, rel_path, name)
        bzd_path = os.path.join(bzd_dir, rel_path, name)
        ctr_path = os.path.join(ctr_dir, rel_path, name)

        gsc_img = cv2.imread(path, cv2.IMREAD_GRAYSCALE)
        if gsc_img is None:
            print(f" Errore leggendo {path}")
            continue

        # Save grayscale
        cv2.imwrite(gsc_path, gsc_img)

        # Save binarized (Otsu)
        _, bzd_img = cv2.threshold(
            gsc_img, 0, 255,
            cv2.THRESH_BINARY + cv2.THRESH_OTSU
        )
        cv2.imwrite(bzd_path, bzd_img)

        # Save contour (edges)
        blurred = cv2.GaussianBlur(gsc_img, (5, 5), 0)
        edges = cv2.Canny(blurred, 50, 150)
        edges_inv = cv2.bitwise_not(edges)
        cv2.imwrite(ctr_path, edges_inv)