from typing import List, Union, Dict, Optional
from omegaconf import ListConfig, DictConfig
import blobfile as bf
import random
from PIL import Image, UnidentifiedImageError
import pytorch_lightning as pl
import torchvision
from torch.utils.data import DataLoader, Dataset
from torchvision import transforms as T
from ..util import instantiate_from_config


# https://github.com/openai/guided-diffusion/blob/main/guided_diffusion/image_datasets.py#L70
def _list_image_files_recursively(data_dir):
    results = []
    for entry in sorted(bf.listdir(data_dir)):
        full_path = bf.join(data_dir, entry)
        ext = entry.split(".")[-1]
        if "." in entry and ext.lower() in ["jpg", "jpeg", "png", "gif"]:
            results.append(full_path)
        elif bf.isdir(full_path):
            results.extend(_list_image_files_recursively(full_path))
    return results


class BaseDataset(Dataset):
    def __init__(self):
        super().__init__()
        self.shuffle = False

    def _convert_to_rgb(self, img):
        return img.convert("RGB") if img.mode != "RGB" else img

    def random_sample(self):
        return self.__getitem__(random.randint(0, self.__len__() - 1))

    def sequential_sample(self, ind):
        if ind >= self.__len__() - 1:
            return self.__getitem__(0)
        return self.__getitem__(ind + 1)

    def skip_sample(self, ind):
        if self.shuffle:
            return self.random_sample()
        return self.sequential_sample(ind=ind)
    
    
class ImageDataset(BaseDataset):
    def __init__(
        self,
        image_paths,
        transforms,
    ):
        super().__init__()
        self.local_images = image_paths
        self.transforms = transforms

    def __len__(self):
        return len(self.local_images)

    def __getitem__(self, idx):
        path = self.local_images[idx]
        try:
            with bf.BlobFile(path, "rb") as f:
                pil_image = Image.open(f)
                pil_image.load()
            pil_image = pil_image.convert("RGB")
        except (UnidentifiedImageError, OSError) as corrupt_image_exceptions:
            print(f"An exception occurred trying to load file {path}.")
            print(f"Skipping index {idx}")
            return self.skip_sample(idx)
        image = self.transforms(pil_image)
        return {"jpg": image}    


class ImageLoader(pl.LightningDataModule):
    def __init__(self, 
                 batch_size, 
                 image_dir: str, 
                 test_ratio: Optional[float] = None, 
                 transforms: Union[Union[Dict, DictConfig], ListConfig] = [T.ToTensor(), T.Lambda(lambda x: x * 2.0 - 1.0)],
                 rescaled: bool = False,
                 num_workers=0, 
                 prefetch_factor=2, 
                 shuffle=True
                 ):
        super().__init__()
        trainimages = _list_image_files_recursively(image_dir)
        random.shuffle(trainimages)
        chained_transforms = []
        if isinstance(transforms, (DictConfig, Dict)):
            transforms = [transforms]
        for trf in transforms:
            trf = instantiate_from_config(trf)
            chained_transforms.append(trf)
        if rescaled:
            chained_transforms.append(T.Lambda(lambda x: x * 2.0 - 1.0))
        transform = T.Compose(chained_transforms)
        testimages = None
        if test_ratio is not None:
            total_size = len(trainimages)
            test_size = int(total_size * test_ratio)
            testimages = trainimages[:test_size]
            trainimages = trainimages[test_size:]

        self.batch_size = batch_size
        self.num_workers = num_workers
        self.prefetch_factor = prefetch_factor if num_workers > 0 else 0
        self.shuffle = shuffle
        self.train_dataset = ImageDataset(trainimages, transform)
        self.test_dataset = ImageDataset(testimages, transform) if testimages is not None else None

    def prepare_data(self):
        pass

    def train_dataloader(self):
        return DataLoader(
            self.train_dataset,
            batch_size=self.batch_size,
            shuffle=self.shuffle,
            num_workers=self.num_workers,
            prefetch_factor=self.prefetch_factor,
        )

    def test_dataloader(self):
        return DataLoader(
            self.test_dataset,
            batch_size=self.batch_size,
            shuffle=False,
            num_workers=self.num_workers,
        )

    def val_dataloader(self):
        return DataLoader(
            self.test_dataset,
            batch_size=self.batch_size,
            shuffle=False,
            num_workers=self.num_workers,
        )

