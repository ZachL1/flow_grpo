import os
import json
import glob

def generate_metadata():
    output_file = "dataset/eval/test_metadata.jsonl"
    
    with open(output_file, 'w') as f:
        # 1. DiversePhotos (DP)
        dp_path = "dataset/eval/DiversePhotos/x1"
        if os.path.exists(dp_path):
            for filename in sorted(os.listdir(dp_path)):
                if filename.lower().endswith(('.png', '.jpg', '.jpeg', '.bmp')):
                    entry = {
                        "tag": "restoration",
                        "source": "DPx1",
                        "lq_image": os.path.join(dp_path, filename).replace("dataset/eval/", "")
                    }
                    f.write(json.dumps(entry) + '\n')
        
        dp_path = "dataset/eval/DiversePhotos/x4"
        if os.path.exists(dp_path):
            for filename in sorted(os.listdir(dp_path)):
                if filename.lower().endswith(('.png', '.jpg', '.jpeg', '.bmp')):
                    entry = {
                        "tag": "restoration",
                        "source": "DPx4",
                        "lq_image": os.path.join(dp_path, filename).replace("dataset/eval/", "")
                    }
                    f.write(json.dumps(entry) + '\n')
        
        # 2. RealPhoto60 (Real60)
        real60_path = "dataset/eval/RealPhoto60/LQ"
        if os.path.exists(real60_path):
            for filename in sorted(os.listdir(real60_path)):
                if filename.lower().endswith(('.png', '.jpg', '.jpeg')):
                    entry = {
                        "tag": "restoration",
                        "source": "Real60",
                        "lq_image": os.path.join(real60_path, filename).replace("dataset/eval/", "")
                    }
                    f.write(json.dumps(entry) + '\n')
        
        # 3. RealSR
        realsr_base = "dataset/eval/RealSR(V3)"
        if os.path.exists(realsr_base):
            # Iterate over cameras (Canon, Nikon)
            cameras = [d for d in os.listdir(realsr_base) if os.path.isdir(os.path.join(realsr_base, d))]
            for camera in sorted(cameras):
                test_path = os.path.join(realsr_base, camera, "Test", "4")
                if os.path.exists(test_path):
                    files = sorted(os.listdir(test_path))
                    for filename in files:
                        if filename.endswith("_LR4.png"):
                            # Construct paths
                            lq_path = os.path.join(test_path, filename)
                            hq_filename = filename.replace("_LR4.png", "_HR.png")
                            hq_path = os.path.join(test_path, hq_filename)
                            
                            # Check if HQ exists
                            if os.path.exists(hq_path):
                                entry = {
                                    "tag": "restoration",
                                    "source": "RealSR",
                                    "lq_image": lq_path.replace("dataset/eval/", ""),
                                    "hq_image": hq_path.replace("dataset/eval/", "")
                                }
                                f.write(json.dumps(entry) + '\n')

if __name__ == "__main__":
    generate_metadata()
