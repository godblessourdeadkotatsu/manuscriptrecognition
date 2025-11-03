import cv2
import numpy as np
import os
from tqdm import tqdm
from concurrent.futures import ProcessPoolExecutor

rng = np.random.default_rng(100)

processed_path_base = "data/processed"
bzd_path_base = "data/processed/bzd"
out_dir = "data/patches"
bzd_patch_base = "data/patches/bzd"
ctr_patch_base = "data/patches/ctr"
gsc_patch_base = "data/patches/gsc"

os.makedirs(bzd_patch_base, exist_ok=True)
os.makedirs(ctr_patch_base, exist_ok=True)
os.makedirs(gsc_patch_base, exist_ok=True)

PATCH_SIZE = 512       #dimensione patch (in pixel)
N_PATCHES = 20         #quante patch estrarre
MIN_BLACK_RATIO = 0.105 # minimo  di segnale in percentuale

#ok da qui in avanti il codice è parecchio incasinato, occhio ai commenti

#ciclo su tutte le sottocartelle di processed/bzd, che corrispondono per costruzione alle sottocartelle di pages
for folder in sorted(os.listdir(bzd_path_base)):

    #creo cartelle per gli output output
    bzd_dir = os.path.join(out_dir,"bzd",folder)
    gsc_dir = os.path.join(out_dir,"gsc",folder)
    ctr_dir = os.path.join(out_dir,"ctr",folder)
    os.makedirs(bzd_dir, exist_ok=True)
    os.makedirs(ctr_dir, exist_ok=True)
    os.makedirs(gsc_dir, exist_ok=True)

    #ciclo sui file della sottocartella corrente
    for file in tqdm(os.listdir(os.path.join(bzd_path_base, folder)), desc="manoscritto "):

        #seleziono il percorso del file corrente
        bzd_path = os.path.join(bzd_path_base,folder,file)

        #leggo il file
        img = cv2.imread(bzd_path, cv2.IMREAD_UNCHANGED)
        if img is None:
            raise FileNotFoundError(f"casini vari, non sono riuscito a leggere {bzd_path}")

        h, w = img.shape
        #contatori per evitare di cercare all'infinito patch valide
        successes = 0
        trials = 0

        while successes < N_PATCHES and trials < N_PATCHES * 100:
            #scelgo due numeri a caso, saranno le coordinate della mia patch
            y = rng.integers(0, h - PATCH_SIZE)
            x = rng.integers(0, w - PATCH_SIZE)
            patch = img[y:y+PATCH_SIZE, x:x+PATCH_SIZE]

            #se non aggiorno qui trials (prima di controllare se la patch è valida) sono meritatamente un cretino
            trials += 1
            
            #scarto patch troppo bianche
            black_ratio = (patch < 128).mean()
            if black_ratio < MIN_BLACK_RATIO:
                continue

            #scrivo i file, pregando che i percorsi siano giusti
            bzd_out = os.path.join(bzd_dir, f"patch_{successes+1:03d}.png")
            cv2.imwrite(bzd_out, patch)

            #ora devo usare la stessa patch per tirare fuoir le immagini da grayscale e countour

            gsc_filename = file.replace("bzd_", "gsc_") #atroce
            gsc_path = os.path.join(processed_path_base, "gsc",folder,gsc_filename)
            gsc_img = cv2.imread(gsc_path, cv2.IMREAD_UNCHANGED)
            if img is None:
                raise FileNotFoundError(f"casini vari, non sono riuscito a leggere {gsc_path}")

            patch = gsc_img[y:y+PATCH_SIZE, x:x+PATCH_SIZE]

            gsc_out = os.path.join(gsc_dir, f"patch_{successes+1:03d}.png")
            cv2.imwrite(gsc_out, patch)

            ctr_filename = file.replace("bzd_", "ctr_") #atroce
            ctr_path = os.path.join(processed_path_base, "ctr",folder,ctr_filename)
            ctr_img = cv2.imread(ctr_path, cv2.IMREAD_UNCHANGED)
            patch = ctr_img[y:y+PATCH_SIZE, x:x+PATCH_SIZE]

            ctr_out = os.path.join(ctr_dir, f"patch_{successes+1:03d}.png")
            cv2.imwrite(ctr_out, patch)

            successes += 1

        print(f"Estratte {successes} patch salvate nelle varie sottocartelle")

#e ora bisogna parallelizzare per forza...
