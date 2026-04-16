
import os
import shutil
from huggingface_hub import HfFileSystem, hf_hub_download
from dotenv import load_dotenv
load_dotenv()


TOKEN = os.getenv("Token")
REPO_ID = "neilrigaud/hagrid-subset"

mappa_classi = {
    "fist": "fist",
    "one": "index",
    "palm": "open_palm",
    "two_up": "two_fingers"
}

cartella_base = "dataset/train"
limite = 150  

def scarica_file():
    fs = HfFileSystem(token=TOKEN)
    
    for classe_hf, tua_cartella in mappa_classi.items():
        print(f"Cerco file per: {classe_hf}...")
        percorso = f"datasets/{REPO_ID}/data/train/{classe_hf}"
        
        try:
            # Legge solo i nomi dei file in quella specifica cartella
            tutti_file = fs.ls(percorso)
            immagini = [f for f in tutti_file if f["name"].endswith((".jpg", ".png"))]
            immagini = immagini[:limite]
            
            if not immagini:
                print(f"Nessun file trovato per {classe_hf}")
                continue
                
            out_dir = os.path.join(cartella_base, tua_cartella)
            os.makedirs(out_dir, exist_ok=True)
            
            for i, file_info in enumerate(immagini):
                nome_file = file_info["name"].split(f"{REPO_ID}/")[1]
                
                cache_path = hf_hub_download(
                    repo_id=REPO_ID,
                    repo_type="dataset",
                    filename=nome_file,
                    token=TOKEN
                )
                
                destinazione = os.path.join(out_dir, f"hagrid_{classe_hf}_{i+1}.jpg")
                shutil.copy(cache_path, destinazione)
                
            print(f"Ok, {len(immagini)} immagini salvate in '{tua_cartella}'")
                
        except Exception as e:
            print(f"Errore con {classe_hf}: {e}")

if __name__ == "__main__":
    scarica_file()