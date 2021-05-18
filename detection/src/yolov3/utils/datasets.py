import glob
import random
import os
import numpy as np
from PIL import Image
import torch
import torch.nn.functional as F

from detection.src.yolov3.utils.utils import horisontal_flip
from torch.utils.data import Dataset
import torchvision.transforms as transforms


def pad_to_square(img, pad_value):
    c, h, w = img.shape
    dim_diff = np.abs(h - w)
    # (upper / left) padding and (lower / right) padding
    pad1, pad2 = dim_diff // 2, dim_diff - dim_diff // 2
    # Determine padding
    pad = (0, 0, pad1, pad2) if h <= w else (pad1, pad2, 0, 0)
    # Add padding
    img = F.pad(img, pad, "constant", value=pad_value)

    return img, pad


def resize(image, size):
    image = F.interpolate(image.unsqueeze(0), size=size, mode="nearest").squeeze(0)
    return image


def random_resize(images, min_size=288, max_size=448):
    new_size = random.sample(list(range(min_size, max_size + 1, 32)), 1)[0]
    images = F.interpolate(images, size=new_size, mode="nearest")
    return images


class ImageFolder(Dataset):
    def __init__(self, folder_path, img_size=416): #data_manager.py에서 (path_to_data_file, img_size=self.image_size)를 인자로 받아옴
        self.files = sorted(glob.glob("%s/*.*" % folder_path)) # 이미지 파일 목록 모두 불러와 소팅(경로포함) 즉, 모든 이미지들의 경로포함 파일이름을 소팅함
        self.img_size = img_size #이미지 크기 초기화

    def __getitem__(self, index): ''' a=ImageFolder() 같이 객체 호출 시, 그 속의 변수에 인덱싱 접근을 위해선 a.files[:]같은 표기가 필요한데, 
                                      __getitem__(특별 매소드) 선언 시 일반적인 인덱싱  방법으로 변수에 접근할 수 있다'''
       
        img_path = self.files[index % len(self.files)] #  __getitem__덕분에 이렇게 바로 인덱싱이 가능하다.-> 주어진
        # Extract image as PyTorch tensor
        img = transforms.ToTensor()(Image.open(img_path)) # indexing
        # Pad to square resolution
        img, _ = pad_to_square(img, 0)
        # Resize
        img = resize(img, self.img_size)

        return img_path, img

    def __len__(self):
        return len(self.files)


class ListDataset(Dataset):
    def __init__(self, list_path, img_size=416, augment=True, multiscale=True, normalized_labels=True):
        with open(list_path, "r") as file:
            self.img_files = file.readlines()

        self.label_files = [
            path.replace("images", "labels").replace(".png", ".txt").replace(".jpg", ".txt")
            for path in self.img_files
        ]
        self.img_size = img_size
        self.max_objects = 100
        self.augment = augment
        self.multiscale = multiscale
        self.normalized_labels = normalized_labels
        self.min_size = self.img_size - 3 * 32
        self.max_size = self.img_size + 3 * 32
        self.batch_count = 0

    def __getitem__(self, index):
        """
        Returns an element of the dataset if index>=0. If index<0, the caller expects information about the labels
        considered in a sampled episode. In this case, this returns the expected label in the first element of the
        tuple, and shallow tensors in the two other elements.
        Args:
            index (int): selects one element of the dataset
        Returns:
            Tuple[str, torch.Tensor, torch.Tensor]: path to the image, image data, and target
            of shape (number_of_boxes_in_image, 6)
        """

        # ---------
        #  Image
        # ---------

        if index < 0:
            return str(-(int(index)+1)), torch.zeros((3,4,4)), torch.zeros((1, 6))

        img_path = self.img_files[index % len(self.img_files)].rstrip()

        # Extract image as PyTorch tensor
        img = transforms.ToTensor()(Image.open(img_path).convert('RGB'))

        # Handle images with less than three channels
        if len(img.shape) != 3:
            img = img.unsqueeze(0)
            img = img.expand((3, img.shape[1:]))

        _, h, w = img.shape
        h_factor, w_factor = (h, w) if self.normalized_labels else (1, 1)
        # Pad to square resolution
        img, pad = pad_to_square(img, 0)
        _, padded_h, padded_w = img.shape

        # ---------
        #  Label
        # ---------

        label_path = self.label_files[index % len(self.img_files)].rstrip()

        targets = None
        if os.path.exists(label_path):
            boxes = torch.from_numpy(np.loadtxt(label_path).reshape(-1, 5))
            # Extract coordinates for unpadded + unscaled image
            x1 = w_factor * (boxes[:, 1] - boxes[:, 3] / 2)
            y1 = h_factor * (boxes[:, 2] - boxes[:, 4] / 2)
            x2 = w_factor * (boxes[:, 1] + boxes[:, 3] / 2)
            y2 = h_factor * (boxes[:, 2] + boxes[:, 4] / 2)
            # Adjust for added padding
            x1 += pad[0]
            y1 += pad[2]
            x2 += pad[1]
            y2 += pad[3]
            # Returns (x, y, w, h)
            boxes[:, 1] = ((x1 + x2) / 2) / padded_w
            boxes[:, 2] = ((y1 + y2) / 2) / padded_h
            boxes[:, 3] *= w_factor / padded_w
            boxes[:, 4] *= h_factor / padded_h

            targets = torch.zeros((len(boxes), 6))
            targets[:, 1:] = boxes

        # Apply augmentations
        if self.augment:
            if np.random.random() < 0.5:
                img, targets = horisontal_flip(img, targets)

        return img_path, img, targets

    def collate_fn_episodic(self, batch):
        """
        Merges a list of samples to form an episode
        Args:
            batch (list): contains the elements sampled from the datasets
        Returns:
            Tuple[Tuple, torch.Tensor, torch.Tensor, torch.Tensor]: respectively contains 0. the paths to sampled
            images ; 1. the sampled images ; 2. targets of the sampled images and 3. the sampled labels
        """
        # Remove lines containing data about the labels
        labels = []
        for index in range(len(batch)):
            if batch[index][0].isdigit():
                labels.append(int(batch[index][0]))
            else:
                begin_index = index
                break
        paths, imgs, all_targets = list(zip(*batch[begin_index:]))
        # Remove empty placeholder targets
        all_targets = [boxes for boxes in all_targets if boxes is not None]
        # Remove boxes that don't match labels
        targets = []
        for boxes in all_targets:
            targets.append(torch.cat([box.view((1, 6)) for box in boxes if int(box[1]) in labels]))
        # Add sample index to targets
        for i, boxes in enumerate(targets):
            boxes[:, 0] = i
        targets = torch.cat(targets, 0)
        # Selects new image size every tenth batch
        if self.multiscale and self.batch_count % 10 == 0:
            self.img_size = random.choice(range(self.min_size, self.max_size + 1, 32))
        # Resize images to input shape
        imgs = torch.stack([resize(img, self.img_size) for img in imgs])
        self.batch_count += 1
        return paths, imgs, targets, torch.tensor(labels, dtype=torch.int32)

    def collate_fn(self, batch):
        """
        Merges a list of samples to form a batch
        Args:
            batch (list): contains the elements sampled from the datasets
        Returns:
            Tuple[Tuple, torch.Tensor, torch.Tensor, torch.Tensor]: respectively contains 0. the paths to sampled
            images ; 1. the sampled images ; 2. targets of the sampled images and 3. the sampled labels
        """
        paths, imgs, targets = list(zip(*batch))
        # Remove empty placeholder targets
        targets = [boxes for boxes in targets if boxes is not None]
        # Add sample index to targets
        for i, boxes in enumerate(targets):
            boxes[:, 0] = i
        targets = torch.cat(targets, 0)
        # Selects new image size every tenth batch
        if self.multiscale and self.batch_count % 10 == 0:
            self.img_size = random.choice(range(self.min_size, self.max_size + 1, 32))
        # Resize images to input shape
        imgs = torch.stack([resize(img, self.img_size) for img in imgs])
        self.batch_count += 1
        return paths, imgs, targets

    def __len__(self):
        return len(self.img_files)
