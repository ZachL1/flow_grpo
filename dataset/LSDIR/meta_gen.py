# Generate metadata for LSDIR dataset
# example:
# {"tag": "restoration", "image": "generated_images/image_9.jpg"}

import os
import json
from tqdm import tqdm

def generate_metadata(dataset_path):
    metadata = []
    for root, dirs, files in os.walk(os.path.join(dataset_path, 'HQ')):
        for file in tqdm(files, desc="Generating metadata"):
            if file.endswith('.jpg') or file.endswith('.png'):
                metadata.append({'tag': 'restoration', 'image': os.path.join(root.replace(dataset_path + '/', ''), file)})
    with open(os.path.join(dataset_path, 'metadata.jsonl'), 'w') as f:
        for item in metadata:
            f.write(json.dumps(item) + '\n')

    # split metadata into train and test
    train_metadata = metadata[:int(len(metadata) * 0.999)]
    test_metadata = metadata[int(len(metadata) * 0.999):]
    print(f"Train metadata: {len(train_metadata)}, Test metadata: {len(test_metadata)}")
    with open(os.path.join(dataset_path, 'train_metadata.jsonl'), 'w') as f:
        for item in train_metadata:
            f.write(json.dumps(item) + '\n')
    with open(os.path.join(dataset_path, 'test_metadata.jsonl'), 'w') as f:
        for item in test_metadata:
            f.write(json.dumps(item) + '\n')

if __name__ == "__main__":
    generate_metadata(os.path.dirname(__file__))