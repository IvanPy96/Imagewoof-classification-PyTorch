# -*- coding: utf-8 -*-

import torch
import torch.nn as nn
import torch.nn.functional as F
from torchvision import models, transforms

import os
from sys import argv
import shutil
import requests

from PIL import Image
import matplotlib.pyplot as plt 

script, file_link = argv      

response = requests.get(file_link, stream=True)
with open('img_test.jpg', 'wb') as out_file:
    shutil.copyfileobj(response.raw, out_file)
del response
    

test_transforms = transforms.Compose([transforms.CenterCrop(320),
                         transforms.ToTensor(),
                         transforms.Normalize(mean=[0.485, 0.456, 0.406], 
                                              std=[0.229, 0.224, 0.225])]
                                     )

device = torch.device('cpu')
model = models.resnet50(pretrained=False)
num_ftrs = model.fc.in_features
model.fc = nn.Linear(num_ftrs, 10)
model.load_state_dict(torch.load('/Users/mordovets.i/imagewoof_ResNet50.pth', map_location=device))
model.eval()

class_names = ['Shih-Tzu', 'Rhodesian ridgeback', 'Beagle', 'English foxhound', 
           'Border terrier', 'Australian terrier', 'Golden retriever', 
           'Old English sheepdog', 'Samoyed', 'Dingo']

fig = plt.figure(figsize=(20,10))

with torch.no_grad():

        img = Image.open('img_test.jpg')
        img_t = test_transforms(img).unsqueeze(0)
        img_t = img_t.to(device)

        outputs = model(img_t)
        _, preds = torch.max(outputs, 1)
        prob = F.softmax(outputs, dim=1)
        top_p, top_class = prob.topk(1, dim = 1)

        plt.title('Порода: {}\n Уверенность модели: {}%'.format(class_names[int(preds.cpu().numpy())], 100*round(top_p.detach().cpu().numpy().tolist()[0][0],2)))
        plt.imshow(img)
        plt.show()
