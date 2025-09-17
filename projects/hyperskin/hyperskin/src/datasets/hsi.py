import PIL
import PIL.Image
import numpy as np
import pandas as pd
import torch
from torch.utils.data import Dataset
from pathlib import Path
from scipy.io import loadmat

DATA_DIR = Path(__file__).parent.parent.parent.parent / "data" / "hsi"
DF_PATH = DATA_DIR / "hsi.csv"

labels_map = {
    "dysplastic_nevi": 0,
    "melanoma": 1,
    "solar_lentigo": 2,
    "interepidermal_carcinoma": 3,
    "nevi": 4,
    "seborrheic_keratosis": 5,
    "melanocytic_lesion": 6
}

others_labels_map = {
    "IEC": "interepidermal_carcinoma",
    "Lentigo": "solar_lentigo",
    "Nv": "nevi",
    "SK": "seborrheic_keratosis",
    "Melt": "melanocytic_lesion"
}

class HSIDataset(Dataset):
    def __init__(self, split='train', transform=None):
        assert split in ['train', 'val', 'test'], "split must be 'train', 'val', or 'test'"
        self.split = split
        self.transform = transform
        
        if DF_PATH.exists():
            self.df = pd.read_csv(DF_PATH)
        else:
            self.df = self.create_df()
            self.df.to_csv(DF_PATH, index=False)
    
    def create_df(self):
        dysplastic_nevi_path = DATA_DIR / "DNCube"    
        melanoma_path = DATA_DIR / "MMCube"
        other_lesions_path = DATA_DIR / "OtherCube"

        data = []

        for lesion_type, path in [("dysplastic_nevi", dysplastic_nevi_path),
                                  ("melanoma", melanoma_path)]:
            for file in path.glob("*.mat"):
                data.append({"file_path": str(file), "label": lesion_type})
        
        for file in other_lesions_path.glob("*.mat"):
            filename = file.stem
            for short_label, full_label in others_labels_map.items():
                if short_label in filename:
                    data.append({"file_path": str(file), "label": full_label})
                    break

        return pd.DataFrame(data)
    
    def __getitem__(self, index):
        
        label = labels_map[self.df.iloc[index]['label']]

        image = loadmat(self.df.iloc[index]['file_path']).popitem()[-1]
        
        if self.transform:
            image = self.transform(image)
        
        # convert to tensor
        image = torch.tensor(image, dtype=torch.float32)
        label = torch.tensor(label, dtype=torch.float32)

        return image, label

    def __len__(self):
        return len(self.df)
    

if __name__ == "__main__":
    dataset = HSIDataset(split='train')
    print(f"Dataset size: {len(dataset)}")
    image, label = dataset[0]
    print(f"Image shape: {image.shape}, Label: {label}")

    image = image.permute(2, 0, 1)
    # use matplotlib to visualize the first band of the first image
    import matplotlib.pyplot as plt
    plt.imshow(image[0, :, :], cmap='gray')
    plt.title(f"Label: {label.item()}")
    plt.show()