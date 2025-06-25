from torch.utils.data import DataLoader
from torch import nn
from torch.optim import Adam
import torch

from module.custom_dataset import MalwareDataset
from module.byteformer import ByteFormer
from module.trainer import SimpleTrainer

'''
agenda behind this project :
    - build a prototypical framework in similar spirit to https://www.crowdstrike.com/en-us/blog/byte-back-next-gen-malware-classification/ 

important concepts/ideas covered:
    - Byteformer 
    - sliding window attention, dilated sliding window
    - strided window attention (think like a combination of transformer+CNN)



steps:
    - basic model building
    - generate some toy dataset
    - training loop (with weights and bias) + sanity check
    - 

'''

BATCH_SIZE=64
NUM_OF_CLASSES=43
DEVICE=torch.device("cuda" if torch.cuda.is_available() else "cpu")



TRAIN_ROOT_DIR="/home/ubuntu/Toy-Proto/malicious_detection/malnet/data/malnet-images-tiny/train"
TEST_ROOT_DIR="/home/ubuntu/Toy-Proto/malicious_detection/malnet/data/malnet-images-tiny/test"
VAL_ROOT_DIR="/home/ubuntu/Toy-Proto/malicious_detection/malnet/data/malnet-images-tiny/val" # not used


if __name__=="__main__":

    # which layer to subsample(ie do strided convolution) , False means no subsample  and True means subsample
    downsample_map=[True, False] + ([False, False] * 1)+ ([False, False] * 1)+[False, False]+[False, False] + [False, False]
    
    model=ByteFormer(opts=None,num_classes=NUM_OF_CLASSES,downsample_map=downsample_map)
    data=model.dummy_input_and_label(1000) # generating some dummy dataset to mimick the actual dataset 
    
    dataset_obj=MalwareDataset(data)
    dataloader_obj=DataLoader(dataset_obj,batch_size=BATCH_SIZE)
    
    loss_fn = nn.CrossEntropyLoss()
    
    trainer = SimpleTrainer(
        model=model,
        loss_fn=loss_fn,
        dataloader=dataloader_obj,
        val_dataloader=None,
        project_name="byteformer-prototype-experiment",
        run_name="subsample-last-layer0"
    )
    trainer.train(epochs=50)
    
    
    
    
