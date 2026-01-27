import torch
from torch.utils.data import Dataset
from pycocotools.coco import COCO
from PIL import Image
import nltk
import os
import random

class CocoDatasetClass(Dataset):
    def __init__(self, root, json_path, vocab, transform=None, max_samples=None):
        self.root = root
        self.coco = COCO(json_path)
        self.vocab = vocab

        self.ids = self.coco.getImgIds()
        
        self.transform = transform
        if max_samples:
            self.ids = self.ids[:max_samples]

    def __len__(self):
        return len(self.ids)

    def __getitem__(self, index):
        vocab = self.vocab
        img_id = self.ids[index]
        
        img_info = self.coco.loadImgs(img_id)[0]
        img_path = os.path.join(self.root, img_info['file_name'])

        try:
            image = Image.open(img_path).convert("RGB")
        except:
            # skip corrupted image
            return self.__getitem__((index + 1) % len(self.ids))
        
        if self.transform:
            image = self.transform(image)

        ann_ids = self.coco.getAnnIds(imgIds=img_id)
        anns = self.coco.loadAnns(ann_ids)
        caption = random.choice(anns)['caption'].lower()
        
        tokens = nltk.tokenize.word_tokenize(caption)

        caption_tokens = nltk.tokenize.word_tokenize(str(caption).lower())
        caption_indices = [self.vocab.word2idx["<start>"]]
        caption_indices += [self.vocab.word2idx.get(token, self.vocab.word2idx["<unk>"]) for token in caption_tokens]
        caption_indices.append(self.vocab.word2idx["<end>"])

        caption_tensor = torch.tensor(caption_indices)

        return image, caption_tensor
    