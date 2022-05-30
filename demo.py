import os
from cv2 import VideoWriter_fourcc
import numpy as np
import torch
import cv2
import imutils

def infer_frame(frame, og_frame, model):
    classes = { 0: 'background', 1: 'Summit Drinking Water', 2: 'Coca-Cola', 3: 'Pineapple Juice' }

    with torch.inference_mode():
        detected = model(frame)[0]
        
        for idx, box in enumerate(detected['boxes']):
            confidence = detected['scores'][idx]

            if confidence > 0.8:
                label_idx = int(detected['labels'][idx])
                box = box.detach().cpu().numpy()
                xmin, ymin, xmax, ymax = box.astype('int')

                label = classes[label_idx]
                if label_idx == 1:
                    color = (255, 0, 0)
                elif label_idx == 2:
                    color = (0, 0 ,255)
                elif label_idx == 3:
                    color = (0, 255, 0)
                cv2.rectangle(og_frame, (xmin, ymin), (xmax,ymax), color, 2)
                cv2.putText(og_frame, label, (xmin, ymin), cv2.FONT_HERSHEY_DUPLEX, 0.5, color, 2)

    return og_frame

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
    device = torch.device('cuda') if torch.cuda.is_available() else torch.device('cpu')

    num_classes = 4

    #Load model
    model = get_instance_segmentation_model(num_classes)
    ckpt = torch.load('model_9.pth', map_location=device)
    model.load_state_dict(ckpt)

    model.to(device)
    model.eval()
    
    vid_capture = cv2.VideoCapture('vid.mp4')

    if (vid_capture.isOpened() == False):
        print("Error opening the video file")
    else:
        # Get frame rate information

        fps = int(vid_capture.get(5))
        print("Frame Rate : ",fps,"frames per second")	

        # Get frame count
        frame_count = vid_capture.get(7)
        print("Frame count : ", frame_count)

    # Video Writer for 30 sec demo vid
    write = cv2.VideoWriter('shorter_demo.mp4', VideoWriter_fourcc(*'mp4v'), 24, (640, 480)) 
    frames = 0

    while(vid_capture.isOpened()):
        # vCapture.read() methods returns a tuple, first element is a bool 
        # and the second is frame

        ret, frame = vid_capture.read()
        if ret == True:
            frame = imutils.resize(frame, width=640, height=480)
            og = frame.copy()

            # Convert frame to tensor
            frame = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)          # cv2 reads images as BGR, so convert to RGB first
            frame = frame.transpose((2, 0, 1))                      # H W C -> C H W
            frame = np.expand_dims(frame, axis=0)                   # add dim for batch size
            frame = frame / 255                                     # normalize [0,1]
            frame = torch.tensor(frame, dtype=torch.float32, device=device)

            frame_show = infer_frame(frame, og, model)

            write.write(frame_show)
            frames+=1

            k = cv2.waitKey(20)
            # 113 is ASCII code for q key
            if k == 113:
                break
        else:
            break

    vid_capture.release()
    write.release()
    cv2.destroyAllWindows()
    
    
if __name__ == "__main__":
    main()