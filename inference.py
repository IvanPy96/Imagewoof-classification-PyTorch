# -*- coding: utf-8 -*-

import torch
import torch.nn as nn
import torch.nn.functional as F
from torchvision import datasets, models, transforms

import os
from sys import argv
import shutil
import requests

from PIL import Image
import matplotlib.pyplot as plt 
%matplotlib inline
%config InlineBackend.figure_format = 'retina'

file_link = argv

# Commented out IPython magic to ensure Python compatibility.
%mkdir test_data

urls = [
        r'file_link'
        ]

for ii, url in enumerate(urls):
    response = requests.get(url, stream=True)
    with open('test_data/img_test_' + str(ii) + '.jpg', 'wb') as out_file:
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
model.load_state_dict(torch.load('https://drive.google.com/file/d/1umw8zS6LUk3ntVbHHRzRvXBrpZcdxYrZ/view?usp=sharing', map_location=device))
model.eval()

class_names = ['Shih-Tzu', 'Rhodesian ridgeback', 'Beagle', 'English foxhound', 
           'Border terrier', 'Australian terrier', 'Golden retriever', 
           'Old English sheepdog', 'Samoyed', 'Dingo']

fig = plt.figure(figsize=(20,10))

with torch.no_grad():

    for ii, file_name in enumerate(os.listdir( './test_data' )):
        img = Image.open( './test_data' + '/' + file_name)
        img_t = test_transforms(img).unsqueeze(0)
        img_t = img_t.to(device)

        outputs = model(img_t)
        _, preds = torch.max(outputs, 1)
        prob = F.softmax(outputs, dim=1)
        top_p, top_class = prob.topk(1, dim = 1)

        ax = plt.subplot(len(os.listdir( '/content/test_data' )),2, ii+1)
        ax.axis('off')
        ax.set_title('Порода: {}\n Уверенность модели: {}%'.format(class_names[int(preds.cpu().numpy())], 100*round(top_p.detach().cpu().numpy().tolist()[0][0],2)))
        plt.imshow(img)