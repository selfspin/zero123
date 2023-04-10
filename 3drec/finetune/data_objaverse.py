import PIL
from torch.utils.data import dataset
import torch
import torch.nn.functional as F
from torch.utils.data import DataLoader
from torchvision import transforms
from datasets import load_dataset, load_from_disk
from tqdm import tqdm
from io import BytesIO
import requests
from PIL import Image
import os.path
from torch import nn
import numpy as np
from transformers import Blip2Processor, Blip2ForConditionalGeneration


class DiffusionLoss(nn.Module):
    def __init__(self):
        super().__init__()

    def forward(self, sigma, y_pred, y):
        return torch.mean(torch.pow(y_pred - y, 2) / sigma)


class Objaverse(dataset.Dataset):
    def __init__(self, picture_size=256, train=True):
        super().__init__()
        self.train = train
        self.path = '/EXT_DISK/datasets/objaverse_animal_render/'

        print('Loading BLIP2 model ...')
        self.processor = Blip2Processor.from_pretrained("Salesforce/blip2-opt-2.7b")
        self.model = Blip2ForConditionalGeneration.from_pretrained("Salesforce/blip2-opt-2.7b",
                                                                   torch_dtype=torch.float16)
        self.model.cuda()
        print('BLIP2 loaded!')

        # train预处理
        self.train_transforms = transforms.Compose([
            transforms.Resize([picture_size]),
            transforms.RandomCrop([picture_size, picture_size]),
            # transforms.RandomHorizontalFlip(),
            transforms.ToTensor()
        ])

        # test预处理
        self.test_transforms = transforms.Compose([
            transforms.Resize([picture_size]),
            transforms.CenterCrop([picture_size, picture_size]),
            transforms.ToTensor()
        ])

        self.UID_list = os.listdir(self.path)

    @torch.no_grad()
    def get_prompt(self, uid):
        img_path = os.path.join(self.path, uid, 'image', '000_img_0001.png')
        img = Image.open(img_path)
        img = img.convert('RGB')
        # 注意区分预处理
        if self.train:
            img = self.train_transforms(img)
        else:
            img = self.test_transforms(img)

        inputs = self.processor(img, return_tensors="pt").to('cuda', torch.float16)
        generated_ids = self.model.generate(**inputs, max_new_tokens=20)
        generated_text = self.processor.batch_decode(generated_ids, skip_special_tokens=True)[0].strip()
        return generated_text

    def __getitem__(self, index):
        uid = self.UID_list[index]

        cam_len_type = np.random.choice([0, 1, 2])
        train_id, refer_id = np.random.choice(range(14), 2) + cam_len_type * 14

        cam_positions = np.load(os.path.join(self.path, uid, 'cam', 'information.npy'), allow_pickle=True)

        train_img_path = os.path.join(self.path, uid, 'image', f'{"%03d" % train_id}_img_0001.png')
        train_img = Image.open(train_img_path)
        train_img = train_img.convert('RGB')
        # 注意区分预处理
        if self.train:
            train_img = self.train_transforms(train_img)
        else:
            train_img = self.test_transforms(train_img)

        train_cam_position = np.array(cam_positions[train_id]['cam_position'])

        refer_img_path = os.path.join(self.path, uid, 'image', f'{"%03d" % refer_id}_img_0001.png')
        refer_img = Image.open(refer_img_path)
        refer_img = refer_img.convert('RGB')
        # 注意区分预处理
        if self.train:
            refer_img = self.train_transforms(refer_img)
        else:
            refer_img = self.test_transforms(refer_img)

        refer_cam_position = np.array(cam_positions[refer_id]['cam_position'])

        return train_img, train_cam_position, refer_img, refer_cam_position, self.get_prompt(uid)

    def __len__(self):
        return len(self.UID_list)
