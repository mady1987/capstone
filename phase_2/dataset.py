import torch
from torch.utils.data import Dataset
from pycocotools.coco import COCO
from PIL import Image
import nltk


class CocoDataset(Dataset):
    def __init__(self, root, json_path, vocab, transform=None, max_samples=5000):
        self.root = root
        self.coco = COCO(json_path)
        self.ids = list(self.coco.anns.keys())[:max_samples]
        self.vocab = vocab
        self.transform = transform

    def __len__(self):
        return len(self.ids)

    def __getitem__(self, idx):
        ann_id = self.ids[idx]
        caption_data = self.coco.anns[ann_id]
        img_id = caption_data['image_id']
        caption = caption_data['caption']

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
    
    