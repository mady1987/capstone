import torch
from torch.utils.data import Dataset
from pycocotools.coco import COCO
from PIL import Image
import nltk


class CocoDataset(Dataset):
    def __init__(self, root, json_path, vocab, transform=None, max_samples=None):
        self.root = root
        self.coco = COCO(json_path)
        self.ids = self.coco.getImgIds()
        self.vocab = vocab
        self.transform = transform
        if max_samples:
            self.ids = self.ids[:max_samples]

    def __len__(self):
        return len(self.ids)

    def __getitem__(self, idx):
        vocab = self.vocab
        img_id = self.ids[idx]
        
        ann_ids = self.coco.getAnnIds(imgIds=img_id)
        caption = self.coco.loadAnns(ann_ids)[0]["caption"]

        path = self.coco.loadImgs(img_id)[0]['file_name']
        image = Image.open(f"{self.root}/{path}").convert("RGB")

        if self.transform is not None:
            image = self.transform(image)

        # Convert caption to tensor of word indices
        caption_tokens = nltk.tokenize.word_tokenize(str(caption).lower())
        caption_indices = [self.vocab.word2idx["<start>"]]
        caption_indices += [self.vocab.word2idx.get(token, self.vocab.word2idx["<unk>"]) for token in caption_tokens]
        caption_indices.append(self.vocab.word2idx["<end>"])

        caption_tensor = torch.tensor(caption_indices)

        return image, caption_tensor
    
    