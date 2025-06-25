
from torch.utils.data import Dataset

from torch.utils.data import Dataset
from torchvision import transforms
from pathlib import Path
import torch
from PIL import Image

class MalwareDataset(Dataset):
    def __init__(self, data):
        self.samples=data["samples"]
        self.targets=data["targets"]
    def __len__(self):
        return len(self.targets)
    def __getitem__(self, index):
        x=self.samples[index]
        y=self.targets[index]
        return {"sample":x,"target":y}
    

class MalwareBytesDataset(Dataset):
    def __init__(self, root_dir, transform=None, max_len=57086):
        self.root_dir = Path(root_dir)
        # self.transform = transform  # optional, e.g. padding, truncation
        self.transform = transform if transform else transforms.ToTensor()
        self.samples = []
        self.class_to_idx = {}
        self.class_to_count={}
        for class_idx, class_dir in enumerate(sorted(self.root_dir.iterdir())):
            if class_dir.is_dir():
                self.class_to_idx[class_dir.name] = class_idx
                count=0
                for img_path in class_dir.rglob("*"):  # catch all files
                    if count>10: break
                    if img_path.is_file():
                        self.samples.append((img_path, class_idx))
                        count+=1
                self.class_to_count[class_dir.name]=count
        # print("Class to Count")
        # print(self.class_to_count)
        # assert False
        
        # Compute class weights: inverse frequency
        # num_classes = len(label_counts)
        # class_counts = torch.tensor([label_counts[i] for i in range(num_classes)], dtype=torch.float)

        

        self.max_len = max_len  # for padding/truncating byte sequences
    
    # Util Method for class imbalance
    def get_class_weights(self):
        # util method : to solve the class imbalance problem
        class_counts=torch.tensor(list(self.class_to_count.values()),dtype=torch.float)
        class_weights = 1.0 / class_counts
        class_weights = class_weights / class_weights.sum()  # Normalize
        return class_weights
    
    def __len__(self):
        return len(self.samples)
        # return 5000

    def __getitem__(self, idx):
        file_path, label = self.samples[idx]
        
        # Read file as bytes
        with open(file_path, "rb") as f:
            byte_data = f.read()

        # Convert bytes to list of integers (0–255)
        byte_array = list(byte_data)

        
        # Pad or truncate to fixed length
        if len(byte_array) < self.max_len:
            byte_array += [-1] * (self.max_len - len(byte_array))
        else:
            byte_array = byte_array[:self.max_len]

        # Convert to tensor
        byte_tensor = torch.tensor(byte_array, dtype=torch.long)

        # if self.transform:
            # byte_tensor = self.transform(byte_tensor)

        return {"sample":byte_tensor,"target":label}

    # def __getitem__(self, idx):
    #     img_path, label = self.samples[idx]
    #     image = Image.open(img_path).convert("RGB")
    #     return {"sample":self.transform(image),"target":label}       