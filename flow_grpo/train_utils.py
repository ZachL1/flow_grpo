import os
import json
from PIL import Image
from torch.utils.data import Dataset, Sampler
import torch
from torchvision.transforms.functional import to_pil_image, center_crop
import random
import numpy as np
from realesrgan.realesrgan import RealESRGANDataset
from realesrgan.batch_transform import RealESRGANBatchTransform
from realesrgan import data_config


SYSTEM_PROMPT = (
    "A conversation between User and Assistant. The user asks a question, and the Assistant solves it. The assistant "
    "first thinks about the reasoning process in the mind and then provides the user with the answer. The reasoning "
    "process and answer are enclosed within <think> </think> and <answer> </answer> tags, respectively, i.e., "
    "<think> reasoning process here </think><answer> answer here </answer>"
)
SCORE_QUESTION_PROMPT = 'What is your overall rating on the quality of this picture? The rating should be a float between 1 and 5, rounded to two decimal places, with 1 representing very poor quality and 5 representing excellent quality. Return the final answer in JSON format with the following keys: "rating": The score.'
RESTORE_PROMPT = 'Restore this image to high quality.'

class RealESRGANPromptImageDataset(Dataset):
    def __init__(self, dataset, split='train'):
        # if split != 'train':
        #     raise ValueError(f"RealESRGANPromptImageDataset only supports train split, got {split}")

        self.file_path = os.path.join(dataset, f'{split}_metadata.jsonl')
        with open(self.file_path, 'r', encoding='utf-8') as f:
            self.metadatas = [json.loads(line) for line in f]
            
            self.prompts = [item.get('prompt', RESTORE_PROMPT) for item in self.metadatas]
            self.image_paths = [item['image'] for item in self.metadatas]
        
        file_meta = [{'image_path': os.path.join(dataset, image_path), 'prompt': prompt} for image_path, prompt in zip(self.image_paths, self.prompts)]
        self.dataset = RealESRGANDataset(file_meta=file_meta, **data_config['dataset']['params'])
        self.batch_transform = RealESRGANBatchTransform(**data_config['batch_transform']['params'])
    
    def __len__(self):
        return len(self.dataset) * 100
    
    def __getitem__(self, idx):
        # set random seed
        self.dataset.set_seed(idx)
        self.batch_transform.set_seed(idx)

        idx = idx % len(self.dataset)
        item = self.dataset[idx]
        assert item['filename'].endswith(self.image_paths[idx]), f"filename {item['filename']} does not match image path {self.image_paths[idx]}"

        trans_need_keys = ['hq', 'kernel1', 'kernel2', 'sinc_kernel']
        for k in trans_need_keys:
            item[k] = item[k].unsqueeze(0) # add batch dimension
        item = self.batch_transform(item)

        return {
            "prompt": item["txt"],
            "metadata": self.metadatas[idx],
            "prompt_with_image_path": f"{self.prompts[idx]}_{self.image_paths[idx]}",
            "image": to_pil_image(item["LQ"].squeeze(0)),
            "image_target": to_pil_image(item["GT"].squeeze(0)),
        }
    
    @staticmethod
    def collate_fn(examples):
        prompts = [example["prompt"] for example in examples]
        metadatas = [example["metadata"] for example in examples]
        prompt_with_image_paths = [example["prompt_with_image_path"] for example in examples]
        images = [example["image"] for example in examples]
        image_targets = [example["image_target"] for example in examples]
        return prompts, metadatas, images, prompt_with_image_paths, image_targets

class EvalPromptImageDataset(Dataset):
    def __init__(self, dataset, split='test'):
        if split != 'test':
            raise ValueError(f"EvalPromptImageDataset only supports test split, got {split}")
        
        self.dataset = dataset
        self.file_path = os.path.join(dataset, f'{split}_metadata.jsonl')
        with open(self.file_path, 'r', encoding='utf-8') as f:
            self.metadatas = [json.loads(line) for line in f]
            self.prompts = [item.get('prompt', RESTORE_PROMPT) for item in self.metadatas]
        
    def __len__(self):
        return len(self.prompts)
    
    def __getitem__(self, idx):
        item = {
            "prompt": self.prompts[idx],
            "metadata": self.metadatas[idx]
        }
        image_path = self.metadatas[idx]['lq_image']
        item["prompt_with_image_path"] = f"{self.prompts[idx]}_{image_path}"
        image = Image.open(os.path.join(self.dataset, image_path)).convert('RGB')
        item["image"] = image
        if 'hq_image' in self.metadatas[idx]:
            hq_image_path = self.metadatas[idx]['hq_image']
            hq_image = Image.open(os.path.join(self.dataset, hq_image_path)).convert('RGB')
            item["image_target"] = hq_image
        else:
            item["image_target"] = None
        
        for k in ["image", "image_target"]:
            if item[k] is None:
                continue
            # center crop item[k]
            item[k] = center_crop(item[k], min(item[k].size))
        
            while min(*item[k].size) >= 2 * 512:
                item[k] = item[k].resize(
                    tuple(x // 2 for x in item[k].size), resample=Image.BOX
                )
            item[k] = item[k].resize((512, 512), resample=Image.BICUBIC)

        return item

    @staticmethod
    def collate_fn(examples):
        prompts = [example["prompt"] for example in examples]
        metadatas = [example["metadata"] for example in examples]
        prompt_with_image_paths = [example["prompt_with_image_path"] for example in examples]
        images = [example["image"] for example in examples]
        image_targets = [example["image_target"] for example in examples]
        return prompts, metadatas, images, prompt_with_image_paths, image_targets

class GenevalPromptImageDataset(Dataset):
    def __init__(self, dataset, split='train'):
        self.dataset = dataset
        self.file_path = os.path.join(dataset, f'{split}_metadata.jsonl')
        with open(self.file_path, 'r', encoding='utf-8') as f:
            self.metadatas = [json.loads(line) for line in f]
            self.prompts = [item['prompt'] for item in self.metadatas]
        
    def __len__(self):
        return len(self.prompts)
    
    def __getitem__(self, idx):
        item = {
            "prompt": self.prompts[idx],
            "metadata": self.metadatas[idx]
        }
        # Assuming 'image' in metadata contains a path to the image file
        image_path = self.metadatas[idx]['image']
        item["prompt_with_image_path"] = f"{self.prompts[idx]}_{image_path}"
        image = Image.open(os.path.join(self.dataset, image_path)).convert('RGB')
        item["image"] = image
        return item

    @staticmethod
    def collate_fn(examples):
        prompts = [example["prompt"] for example in examples]
        metadatas = [example["metadata"] for example in examples]
        images = [example["image"] for example in examples]
        prompt_with_image_paths = [example["prompt_with_image_path"] for example in examples]
        return prompts, metadatas, images, prompt_with_image_paths



class DistributedKRepeatSampler(Sampler):
    def __init__(self, dataset, batch_size, k, num_replicas, rank, seed=0):
        self.dataset = dataset
        self.batch_size = batch_size  # Batch size per replica
        self.k = k                    # Number of repetitions per sample
        self.num_replicas = num_replicas  # Total number of replicas
        self.rank = rank              # Current replica rank
        self.seed = seed              # Random seed for synchronization
        
        # Compute the number of unique samples needed per iteration
        self.total_samples = self.num_replicas * self.batch_size
        assert self.total_samples % self.k == 0, f"k can not divide n*b, k{k}-num_replicas{num_replicas}-batch_size{batch_size}"
        self.m = self.total_samples // self.k  # Number of unique samples
        self.epoch = 0

    def __iter__(self):
        while True:
            # Generate a deterministic random sequence to ensure all replicas are synchronized
            g = torch.Generator()
            g.manual_seed(self.seed + self.epoch)
            
            # Randomly select m unique samples
            indices = torch.randperm(len(self.dataset), generator=g)[:self.m].tolist()
            
            # Repeat each sample k times to generate n*b total samples
            repeated_indices = [idx for idx in indices for _ in range(self.k)]
            
            # Shuffle to ensure uniform distribution
            shuffled_indices = torch.randperm(len(repeated_indices), generator=g).tolist()
            shuffled_samples = [repeated_indices[i] for i in shuffled_indices]
            
            # Split samples to each replica
            per_card_samples = []
            for i in range(self.num_replicas):
                start = i * self.batch_size
                end = start + self.batch_size
                per_card_samples.append(shuffled_samples[start:end])
            
            # Return current replica's sample indices
            yield per_card_samples[self.rank]
    
    def set_epoch(self, epoch):
        self.epoch = epoch  # Used to synchronize random state across epochs