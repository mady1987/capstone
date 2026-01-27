import torch
from torch.utils.data import Dataset
from PIL import Image
import nltk
import os
import random
import csv

class FlickrDataset(Dataset):
    def __init__(self, root, captions_path, vocab, transform=None, max_samples=None):
        self.root = root
        self.vocab = vocab
        self.transform = transform

        image_captions = {}
        with open(captions_path, "r", encoding="utf-8") as f:
            reader = csv.reader(f)
            next(reader)  # skip header: image,caption

            for row in reader:
                if len(row) < 2:
                    continue
                img_name, caption = row
                if img_name not in image_captions:
                        image_captions[img_name] = []
                image_captions[img_name].append(caption)
        # Load captions
        self.image_captions = image_captions

        self.image_ids = list(self.image_captions.keys())

        if max_samples:
            self.image_ids = self.image_ids[:max_samples]

    def __len__(self):
        return len(self.image_ids)

    def __getitem__(self, index):
        vocab = self.vocab
        img_name = self.image_ids[index]
        img_path = os.path.join(self.root, img_name)

        try:
            image = Image.open(img_path).convert("RGB")
        except:
            # Skip corrupted image
            return self.__getitem__((index + 1) % len(self.image_ids))

        if self.transform:
            image = self.transform(image)

        # Randomly select one caption per image
        caption = random.choice(self.image_captions[img_name])

        tokens = nltk.tokenize.word_tokenize(caption)

        caption_indices = [vocab.word2idx["<start>"]]
        caption_indices += [
            vocab.word2idx.get(token, vocab.word2idx["<unk>"])
            for token in tokens
        ]
        caption_indices.append(vocab.word2idx["<end>"])

        caption_tensor = torch.tensor(caption_indices)

        return image, caption_tensor, img_name