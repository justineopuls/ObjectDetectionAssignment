import gdown
import tarfile
import shutil
import os.path
import os
import numpy as np
import torch
import torch.utils.data
import pandas as pd
from PIL import Image


class Drinks(torch.utils.data.Dataset):
    def __init__(self, root, csv_file, transforms=None):
        self.root = root
        self.transforms = transforms
        
        self.drinks_frame = pd.read_csv(csv_file)

        self.frame = self.drinks_frame['frame']
        self.imgs = np.unique(self.frame)

        self.image_name = np.unique(self.frame)
        self.num_objs = len(self.image_name)

        # print(num_objs)
        self.boxes = {}
        self.area = {}
        self.labels = {}
        self.iscrowd = {}

        for key in self.image_name:
          self.boxes[key] = []
          self.area[key] = []
          self.labels[key] = []
          self.iscrowd[key] = []
          
        for value in self.drinks_frame.values:
          xmin = value[1]
          xmax = value[2]
          ymin = value[3]
          ymax = value[4]
          area_temp = ((xmax - xmin) * (ymax - ymin))
          label_temp = value[5]

          self.boxes[value[0]].append([xmin, ymin, xmax, ymax])
          self.area[value[0]].append(area_temp)
          self.labels[value[0]].append(label_temp)
          self.iscrowd[value[0]].append(0)

        for key in self.image_name:
          self.boxes[key] = torch.from_numpy(np.array(self.boxes[key], dtype=np.float32))
          self.area[key] = torch.from_numpy(np.array(self.area[key], dtype=np.float32))
          self.labels[key] = torch.from_numpy(np.array(self.labels[key], dtype=np.int64))
          self.iscrowd[key] = torch.from_numpy(np.array(self.iscrowd[key], dtype=np.int64))
        
        # for key in self.image_name:
        #   self.boxes[key] = torch.as_tensor(self.boxes, dtype = torch.float32)
        #   self.area[key] = torch.as_tensor(self.area, dtype = torch.float32)
        #   self.labels[key] = torch.as_tensor(self.labels, dtype = torch.int64)
        #   self.iscrowd[key] = torch.as_tensor(self.iscrowd, dtype = torch.int64)

    def __getitem__(self, idx):

        # load images ad masks
        img_path = os.path.join(self.root, self.imgs[idx])
        img = Image.open(img_path).convert("RGB")
        img_name = self.imgs[idx]

        target = {}
        target["boxes"] = self.boxes[img_name]
        target["labels"] = self.labels[img_name]
        target["image_id"] = torch.tensor([idx])
        target["area"] = self.area[img_name]
        target["iscrowd"] = self.iscrowd[img_name]

        # print(target)

        if self.transforms is not None:
            img, target = self.transforms(img, target)

        return img, target

    def __len__(self):
        return len(self.imgs)


import torchvision
from torchvision.models.detection.faster_rcnn import FastRCNNPredictor
from torchvision.models.detection.mask_rcnn import MaskRCNNPredictor
      
def get_instance_segmentation_model(num_classes):
    # load an instance segmentation model pre-trained on COCO
    model = torchvision.models.detection.fasterrcnn_resnet50_fpn(pretrained=True)

    # get the number of input features for the classifier
    in_features = model.roi_heads.box_predictor.cls_score.in_features
    # replace the pre-trained head with a new one
    model.roi_heads.box_predictor = FastRCNNPredictor(in_features, num_classes)

    return model

from engine import train_one_epoch, evaluate
import utils
import transforms as T


def get_transform(train):
    transforms = []
    # converts the image, a PIL image, into a PyTorch Tensor
    transforms.append(T.ToTensor())
    if train:
        # during training, randomly flip the training images
        # and ground-truth for data augmentation
        transforms.append(T.RandomHorizontalFlip(0.5))
    return T.Compose(transforms)

def main():
    if os.path.exists('drinks.tar.gz'):
        print('File already exists.')
    else:
        url = 'https://drive.google.com/file/d/1AdMbVK110IKLG7wJKhga2N2fitV1bVPA/view'
        output = 'drinks.tar.gz'
        gdown.download(url = url, output = output, quiet = False, fuzzy = True)
        tar = tarfile.open(output)
        tar.extractall()
        tar.close()

    if not os.path.exists('labels_train.csv'):
        shutil.move('./drinks/labels_train.csv', os.getcwd())
    if not os.path.exists('labels_test.csv'):
        shutil.move('./drinks/labels_test.csv', os.getcwd())
        
    device = torch.device('cuda') if torch.cuda.is_available() else torch.device('cpu')

    # our dataset has two classes only - background and person
    num_classes = 4
    
    # use our dataset and defined transformations
    dataset_train = Drinks('drinks', 'labels_train.csv', get_transform(train=True))
    dataset_test = Drinks('drinks', 'labels_test.csv', get_transform(train=False))

    # define training and validation data loaders
    # define training and validation data loaders
    data_loader = torch.utils.data.DataLoader(
        dataset_train, batch_size=2, shuffle=True, num_workers=4,
        collate_fn=utils.collate_fn)

    data_loader_test = torch.utils.data.DataLoader(
        dataset_test, batch_size=1, shuffle=False, num_workers=4,
        collate_fn=utils.collate_fn)

    # get the model using our helper function
    model = get_instance_segmentation_model(num_classes)
    
    # move model to the right device
    model.to(device)

    # construct an optimizer
    params = [p for p in model.parameters() if p.requires_grad]
    optimizer = torch.optim.SGD(params, lr=0.005,
                                momentum=0.9, weight_decay=0.0005)
    # and a learning rate scheduler
    lr_scheduler = torch.optim.lr_scheduler.StepLR(optimizer,
                                                   step_size=3,
                                                   gamma=0.1)

    # let's train it for 10 epochs
    num_epochs = 10

    for epoch in range(num_epochs):
        # train for one epoch, printing every 10 iterations
        train_one_epoch(model, optimizer, data_loader, device, epoch, print_freq=10)
        # update the learning rate
        lr_scheduler.step()
        # evaluate on the test dataset
        evaluate(model, data_loader_test, device=device)

        model_name = 'model_{}.pth'.format(epoch)
        torch.save(model.state_dict(), model_name)
        

    print("That's it!")
    
if __name__ == "__main__":
    main()