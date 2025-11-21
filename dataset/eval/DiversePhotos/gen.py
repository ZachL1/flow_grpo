
'''
Steps for reproducing “DiversePhotos×1”:
1. Download SPAQ, KONIQ, and LIVEChallenge datasets.
2. Gather images whose file names are mentioned in the following 12 listings.
3. Center-crop all images from SPAQ and KONIQ datasets to 512 × 512 resolution.
4. Resize (bicubic) all images from LIVEChallenge dataset (from 500 × 500) to 512 × 512 resolution.

(SPAQ, low resolution as dominating degradation, with other degradations): 00019, 00025, 00033, 00109, 00192, 00226, 00251, 00381, 00414, 00559, 00561, 00585, 00743, 03973, 04085, 04136, 04270, 04317, 04334, 06682.
(SPAQ, motion blur as dominating degradation, with other degradations): 00043, 00075, 00121, 00161, 00175, 00178, 00236, 01868, 03513, 04089, 04272, 04380, 06341, 06863, 10388, 10391, 10495.
(SPAQ, defocus blur as dominating degradation, with other degradations): 00125, 00212, 00282, 04379, 06727, 09121.
(SPAQ, noise as dominating degradation, with other degradations): 00077, 00086, 00096, 00143, 00187, 00199, 00292, 00365, 00450, 04337, 04345, 06485, 06703, 07121, 07162, 07394, 07494, 07866, 07903, 08108, 09682.
(KONIQ, low resolution as dominating degradation, with other degradations): 1755366250, 187640892, 2096424103, 2443117568, 2596393826, 2704811, 2836089223, 2956548148, 3015139450, 3435545140, 3551648026, 4378419360, 527633229, 86243803.
(KONIQ, motion blur as dominating degradation, with other degradations): 2367261033, 3147416579, 331406867, 62480371.
(KONIQ, defocus blur as dominating degradation, with other degradations): 1306193020, 1315889745, 155711788, 1807195948, 206294085, 2166503846, 2214729676, 23371433, 2360058082, 2950983139, 3149433848, 324339500, 427196028, 518080817.
(KONIQ, noise as dominating degradation, with other degradations): 1317678723, 1987196687, 218457399, 2593384818, 2837843986, 2867718050, 3727572481, 4410900135.
(LIVEChallenge, low resolution as dominating degradation, with other degradations): 110, 723, 760, 805, 819, 875.
(LIVEChallenge, motion blur as dominating degradation, with other degradations): 1017, 104, 1156, 12, 154, 239, 270, 283, 29, 429, 458, 460, 468, 659, 663, 700, 732, 810, 856.
(LIVEChallenge, defocus blur as dominating degradation, with other degradations): 337, 550, 592, 698, 713, 714, 717, 731, 737, 750, 751, 787, 788, 855, 862, 873, 874, 876, 884, 887.
(LIVEChallenge, noise as dominating degradation, with other degradations): 1001, 1011, 1024, 1037, 1055, 1079, 1098, 1149, 370, 443, 5.
'''

import os
from PIL import Image
import shutil

# Base paths
BASE_DIR = os.path.dirname(os.path.dirname(os.path.abspath(__file__))) # data/
SPAQ_DIR = os.path.join(BASE_DIR, 'SPAQ', 'TestImage')
KONIQ_DIR = os.path.join(BASE_DIR, 'koniq10k', '1024x768')
LIVE_DIR = os.path.join(BASE_DIR, 'LIVEChallenge', 'Images')

OUTPUT_DIR_X1 = os.path.join(os.path.dirname(os.path.abspath(__file__)), 'x1')
OUTPUT_DIR_X4 = os.path.join(os.path.dirname(os.path.abspath(__file__)), 'x4')

# Ensure output directories exist
os.makedirs(OUTPUT_DIR_X1, exist_ok=True)
os.makedirs(OUTPUT_DIR_X4, exist_ok=True)

# Image Lists
spaq_lists = [
    # low resolution
    ["00019", "00025", "00033", "00109", "00192", "00226", "00251", "00381", "00414", "00559", "00561", "00585", "00743", "03973", "04085", "04136", "04270", "04317", "04334", "06682"],
    # motion blur
    ["00043", "00075", "00121", "00161", "00175", "00178", "00236", "01868", "03513", "04089", "04272", "04380", "06341", "06863", "10388", "10391", "10495"],
    # defocus blur
    ["00125", "00212", "00282", "04379", "06727", "09121"],
    # noise
    ["00077", "00086", "00096", "00143", "00187", "00199", "00292", "00365", "00450", "04337", "04345", "06485", "06703", "07121", "07162", "07394", "07494", "07866", "07903", "08108", "09682"]
]

koniq_lists = [
    # low resolution
    ["1755366250", "187640892", "2096424103", "2443117568", "2596393826", "2704811", "2836089223", "2956548148", "3015139450", "3435545140", "3551648026", "4378419360", "527633229", "86243803"],
    # motion blur
    ["2367261033", "3147416579", "331406867", "62480371"],
    # defocus blur
    ["1306193020", "1315889745", "155711788", "1807195948", "206294085", "2166503846", "2214729676", "23371433", "2360058082", "2950983139", "3149433848", "324339500", "427196028", "518080817"],
    # noise
    ["1317678723", "1987196687", "218457399", "2593384818", "2837843986", "2867718050", "3727572481", "4410900135"]
]

live_lists = [
    # low resolution
    ["110", "723", "760", "805", "819", "875"],
    # motion blur
    ["1017", "104", "1156", "12", "154", "239", "270", "283", "29", "429", "458", "460", "468", "659", "663", "700", "732", "810", "856"],
    # defocus blur
    ["337", "550", "592", "698", "713", "714", "717", "731", "737", "750", "751", "787", "788", "855", "862", "873", "874", "876", "884", "887"],
    # noise
    ["1001", "1011", "1024", "1037", "1055", "1079", "1098", "1149", "370", "443", "5"]
]

def center_crop(img, size):
    w, h = img.size
    left = (w - size[0]) / 2
    top = (h - size[1]) / 2
    right = (w + size[0]) / 2
    bottom = (h + size[1]) / 2
    return img.crop((left, top, right, bottom))

def process_spaq_koniq(ids, src_dir, prefix='', ext='.jpg'):
    for img_id in ids:
        # Handle cases where extension might be missing or different
        filename = f"{img_id}{ext}"
        src_path = os.path.join(src_dir, filename)
        
        if not os.path.exists(src_path):
            print(f"Warning: {src_path} not found.")
            continue
            
        try:
            with Image.open(src_path) as img:
                # Center crop to 512x512
                img_x1 = center_crop(img, (512, 512))
                
                # Save x1
                x1_path = os.path.join(OUTPUT_DIR_X1, f"{prefix}_{filename}")
                img_x1.save(x1_path)
                print(f"Saved {x1_path}")
                
        except Exception as e:
            print(f"Error processing {filename}: {e}")

def process_live(ids, src_dir):
    for img_id in ids:
        # Try common extensions
        src_path = None
        filename = None
        for ext in ['.bmp', '.JPG', '.jpg']:
            temp_path = os.path.join(src_dir, f"{img_id}{ext}")
            if os.path.exists(temp_path):
                src_path = temp_path
                filename = f"livec_{img_id}{ext}"
                break
        
        if src_path is None:
            print(f"Warning: Image {img_id} not found in {src_dir}")
            continue
            
        try:
            with Image.open(src_path) as img:
                # Resize to 512x512
                img_x1 = img.resize((512, 512), Image.BICUBIC)
                
                # Save x1
                x1_path = os.path.join(OUTPUT_DIR_X1, filename)
                img_x1.save(x1_path)
                print(f"Saved {x1_path}")
                
        except Exception as e:
            print(f"Error processing {filename}: {e}")

def generate_x4():
    # Iterate over all images in x1 directory
    for filename in os.listdir(OUTPUT_DIR_X1):
        if filename.lower().endswith(('.png', '.jpg', '.jpeg', '.bmp')):
            src_path = os.path.join(OUTPUT_DIR_X1, filename)
            try:
                with Image.open(src_path) as img:
                    # Center crop to 128x128
                    img_x4 = center_crop(img, (128, 128))
                    
                    # Save x4
                    x4_path = os.path.join(OUTPUT_DIR_X4, filename)
                    img_x4.save(x4_path)
                    print(f"Saved {x4_path}")
            except Exception as e:
                print(f"Error processing x4 for {filename}: {e}")

if __name__ == "__main__":
    # Flatten lists
    spaq_all = [item for sublist in spaq_lists for item in sublist]
    koniq_all = [item for sublist in koniq_lists for item in sublist]
    live_all = [item for sublist in live_lists for item in sublist]

    print("Processing SPAQ...")
    process_spaq_koniq(spaq_all, SPAQ_DIR, 'spaq')
    
    print("Processing KONIQ...")
    process_spaq_koniq(koniq_all, KONIQ_DIR, 'koniq')
    
    print("Processing LIVEChallenge...")
    process_live(live_all, LIVE_DIR)
    
    print("Generating x4...")
    generate_x4()
    
    print("Done.")
