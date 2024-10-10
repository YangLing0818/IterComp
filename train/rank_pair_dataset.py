import os
import json
import math
import torch
from torch.utils.data import Dataset
from config.utils import *
from config.options import *
from PIL import Image
from torchvision.transforms import Compose, Resize, CenterCrop, ToTensor, Normalize
from tqdm import tqdm
from transformers import BertTokenizer
import clip

try:
    from torchvision.transforms import InterpolationMode
    BICUBIC = InterpolationMode.BICUBIC
except ImportError:
    BICUBIC = Image.BICUBIC

def _convert_image_to_rgb(image):
    return image.convert("RGB")

def _transform(n_px):
    return Compose([
        Resize(n_px, interpolation=BICUBIC),
        CenterCrop(n_px),
        _convert_image_to_rgb,
        ToTensor(),
        Normalize((0.48145466, 0.4578275, 0.40821073), (0.26862954, 0.26130258, 0.27577711)),
    ])

def init_tokenizer():
    tokenizer = BertTokenizer.from_pretrained('bert-base-uncased')
    tokenizer.add_special_tokens({'bos_token':'[DEC]'})
    tokenizer.add_special_tokens({'additional_special_tokens':['[ENC]']})       
    tokenizer.enc_token_id = tokenizer.additional_special_tokens_ids[0]  
    return tokenizer

class rank_pair_dataset(Dataset):
    def __init__(self, dataset):
        self.preprocess = _transform(config['BLIP']['image_size'])
        self.tokenizer = init_tokenizer()
        self.model_gallery = config["model_gallery"]
        
        if opts.load_pair_store:
            self.dataset_path = os.path.join(config['pair_store_base'], f"{dataset}.pth")
            self.data = torch.load(self.dataset_path)
        else:
            self.dataset_path = os.path.join(config['data_base'], f"{dataset}.json")
            with open(self.dataset_path, "r") as f:
                self.data = json.load(f)
            self.data = self.make_data()
        
        self.iters_per_epoch = int(math.ceil(len(self.data)*1.0/opts.batch_size))

    def __getitem__(self, index):
        return self.data[index]

    def __len__(self):
        return len(self.data)
    
    def store_dataset(self, dataset):
        makedir(config['pair_store_base'])
        torch.save(self.data, os.path.join(config['pair_store_base'], f"{dataset}.pth"))
    
    def make_data(self):
        data_items = []
        
        bar = tqdm(range(len(self.data)), desc=f'making dataset: ')
        for item in self.data:
            dict_item = {}
            img_set = []    
            for i in range(len(self.model_gallery)):
                pil_image = Image.open(item[f"image_{self.model_gallery[i]}"])
                image = self.preprocess(pil_image)
                img_set.append(image)
            text_input = self.tokenizer(item["prompt"], padding='max_length', truncation=True, max_length=35, return_tensors="pt")
            ranking = [int(char) for char in item["rank"]]
            for i in range(len(ranking)-1):
                for j in range(i+1, len(ranking)):
                    dict_item = {}
                    dict_item['clip_text'] = clip.tokenize(item["prompt"], truncate=True)
                    dict_item['text_ids'] = text_input.input_ids
                    dict_item['text_mask'] = text_input.attention_mask
                    dict_item['img_better'] = img_set[ranking[i]-1]
                    dict_item['img_worse'] = img_set[ranking[j]-1]
                    data_items.append(dict_item)
            bar.update(1)
        return data_items
    
    def make_data_iterative(self, iteration):
        data_items = []
        
        bar = tqdm(range(len(self.data)), desc=f'making dataset: ')
        for item in self.data:
            dict_item = {}
            img_set = []    
            for i in range(len(self.model_gallery)):
                pil_image = Image.open(item[f"image_{self.model_gallery[i]}"])
                image = self.preprocess(pil_image)
                img_set.append(image)
            text_input = self.tokenizer(item["prompt"], padding='max_length', truncation=True, max_length=35, return_tensors="pt")
            ranking = [int(char) for char in item["rank"]]
            for i in range(len(ranking)-1):
                for j in range(i+1, len(ranking)):
                    if ranking[i] == iteration - 1  or ranking[j] == iteration - 1 or ranking[i] == iteration or ranking[j] == iteration:
                        dict_item = {}
                        dict_item['clip_text'] = clip.tokenize(item["prompt"], truncate=True)
                        dict_item['text_ids'] = text_input.input_ids
                        dict_item['text_mask'] = text_input.attention_mask
                        dict_item['img_better'] = img_set[ranking[i]-1]
                        dict_item['img_worse'] = img_set[ranking[j]-1]
                        data_items.append(dict_item)
            bar.update(1)
        return data_items
