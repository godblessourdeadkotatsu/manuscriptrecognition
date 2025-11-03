import cv2
import numpy as np
import os
from tqdm import tqdm
from concurrent.futures import ProcessPoolExecutor


processed_read_base = "data/processed"
out_dir = "data/patches"
bzd_save_base = "data/patches/bzd"
ctr_save_base = "data/patches/ctr"
gsc_save_base = "data/patches/gsc"

os.makedirs(bzd_save_base, exist_ok=True)
os.makedirs(ctr_save_base, exist_ok=True)
os.makedirs(gsc_save_base, exist_ok=True)

PATCH_SIZE = 512       #dimensione patch (in pixel)
N_PATCHES = 20         #quante patch estrarre
MIN_BLACK_RATIO = 0.105 # minimo  di segnale in percentuale

#ok da qui in avanti il codice è parecchio incasinato (ma va'?), occhio ai commenti

def patchmaker(args):

    rng = np.random.default_rng(seed)

    folder, file = args

    #percorsi di lettura
    bzd_read_file = os.path.join(processed_read_base, "bzd", folder, file)

    gsc_filename = file.replace("bzd_", "gsc_") #atroce
    gsc_read_file = os.path.join(processed_read_base, "gsc", folder, gsc_filename)

    ctr_filename = file.replace("bzd_", "ctr_") #atroce
    ctr_read_file = os.path.join(processed_read_base, "ctr", folder, ctr_filename)

    #percorsi di scrittura
    bzd_save_path = os.path.join(bzd_save_base, folder)
    gsc_save_path = os.path.join(gsc_save_base, folder)
    ctr_save_path = os.path.join(ctr_save_base, folder)

    #eventualmente fare cartelle...
    os.makedirs(bzd_save_path, exist_ok=True)
    os.makedirs(gsc_save_path, exist_ok=True)
    os.makedirs(ctr_save_path, exist_ok=True)

    #caricare immagini
    bzd_img = cv2.imread(bzd_read_file, cv2.IMREAD_UNCHANGED)
    gsc_img = cv2.imread(gsc_read_file, cv2.IMREAD_UNCHANGED)
    ctr_img = cv2.imread(ctr_read_file, cv2.IMREAD_UNCHANGED)

    if bzd_img is None or gsc_img is None or ctr_img is None:
        return f"casini con i file"
    
    #carica dimensioni e inizializza i contatori
    h, w = bzd_img.shape
    successes = 0
    trials = 0

    #inizia a estrarre patches dall'immagine bzd
    while successes < N_PATCHES and trials < N_PATCHES * 100:
        
        #aggiorna ORA il contatore di trials se non vuoi essere un cretino (lo sei comunque)
        trials += 1

        #scelgo i numeri random
        y = rng.integers(0, h - PATCH_SIZE)
        x = rng.integers(0, w - PATCH_SIZE)
        patch = bzd_img[y:y+PATCH_SIZE, x:x+PATCH_SIZE]

        #scarto patch troppo bianche
        black_ratio = (patch < 128).mean()
        if black_ratio < MIN_BLACK_RATIO:
            continue

        #scrivo la patch se è buona
        basename = os.path.splitext(file)[0] #identificatore
        bzd_out = os.path.join(bzd_save_path, f"{basename}patch_{successes+1:03d}.png")
        cv2.imwrite(bzd_out, patch)

        #ora devo usare la stessa patch per tirare fuoir le immagini da grayscale e countour

        patch = gsc_img[y:y+PATCH_SIZE, x:x+PATCH_SIZE]
        gsc_out = os.path.join(gsc_save_path, f"{basename}patch_{successes+1:03d}.png")
        cv2.imwrite(gsc_out, patch)

        patch = ctr_img[y:y+PATCH_SIZE, x:x+PATCH_SIZE]

        ctr_out = os.path.join(ctr_save_path, f"{basename}patch_{successes+1:03d}.png")
        cv2.imwrite(ctr_out, patch)

        successes += 1

    return f"{folder}/{file}: {successes} patch"
    

#preparo una lista di tuple file/sottocartella così da avere tutti i task separati
tasks = []
for i, folder in enumerate(sorted(os.listdir(os.path.join(processed_read_base, "bzd")))):
    folder_path = os.path.join(processed_read_base, "bzd", folder)
    for j, file in enumerate(os.listdir(folder_path)):
        if file.endswith(".png"):
            seed =100 + i * 1000 + j
            tasks.append((folder, file))

#parallelizzo, che dio ce la mandi buona
with ProcessPoolExecutor(max_workers=os.cpu_count()) as executor:
    results = list(tqdm(executor.map(patchmaker, tasks),
                        total=len(tasks), desc="Estrazione patch"))

print("\n".join(results)) 
