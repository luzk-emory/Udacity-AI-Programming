import numpy as np
import torch
from torch import nn
from torch import optim
from torchvision import datasets, transforms, models
from PIL import Image
import json
from collections import OrderedDict
from time import time
import argparse

# Creates Argument Parser object named parser
parser = argparse.ArgumentParser()

parser.add_argument('image_path', type = str, 
                    help = 'Provide the path to a singe image (required)')
parser.add_argument('save_path', type = str, 
                    help = 'Provide the path to the file of the trained model (required)')

parser.add_argument('--category_names', type = str, #default = 'cat_to_name.json',
                    help = 'Use a mapping of categories to real names')
parser.add_argument('--top_k', type = int, default = 5,
                    help = 'Return top K most likely classes. Default value is 5')
# GPU
parser.add_argument('--gpu', action='store_true',
                    help = "Add to activate CUDA")

args_in = parser.parse_args()



if args_in.gpu:
    device = torch.device("cuda")
    print("****** CUDA activated ****************************")
else:
    device = torch.device("cpu")


### ------------------------------------------------------------
###                         Load the checkpoint 
### ------------------------------------------------------------

def load_checkpoint(filepath):
    checkpoint = torch.load(filepath)
    
    if checkpoint['arch'] == 'densenet121':
        model = models.densenet121(pretrained=True)
    elif checkpoint['arch'] == 'vgg13':
        model = models.vgg13(pretrained = True)
    else:
        raise ValueError('Model arch error.')

    for param in model.parameters():
        param.requires_grad = False
        
    model.classifier = checkpoint['classifier']
    model.load_state_dict(checkpoint['model_state_dict'])
    model.class_to_idx = checkpoint['class_to_idx']
    
    
    return model


checkpoint_path = args_in.save_path
model = load_checkpoint(checkpoint_path)
model.to(device);

### ------------------------------------------------------------




def process_image(image):
    ''' Scales, crops, and normalizes a PIL image for a PyTorch model,
        returns an Numpy array
    '''
    
    # TODO: Process a PIL image for use in a PyTorch model
    img = Image.open(image) 
    transform = transforms.Compose([transforms.Resize(255),
                                    transforms.CenterCrop(224),
                                    transforms.ToTensor(),
                                    transforms.Normalize([0.485, 0.456, 0.406], 
                                                         [0.229, 0.224, 0.225])])
    img = np.array(transform(img))

    return img





def predict(image_path, model, topk=5):
    ''' Predict the class (or classes) of an image using a trained deep learning model.
    '''
    
    # TODO: Implement the code to predict the class from an image file
    image = process_image(image_path)
    image = torch.from_numpy(image).type(torch.FloatTensor)
    image = image.unsqueeze(0) 
    image = image.to(device)
    
    # to make sure no drop; otherwise, the output is random
    model.eval()
    
    with torch.no_grad ():
        output = model.forward(image)
        
    output_prob = torch.exp(output)
    
    probs, indeces = output_prob.topk(topk)
    probs   =   probs.to('cpu').numpy().tolist()[0]
    indeces = indeces.to('cpu').numpy().tolist()[0]
    
    mapping = {val: key for key, val in model.class_to_idx.items()}
    classes = [mapping[item] for item in indeces]
    
    return probs, classes





### ------------------------------------------------------------
###                         Prediction 
### ------------------------------------------------------------

image_path = args_in.image_path
top_k      = args_in.top_k

probs, classes = predict(image_path, model, topk=top_k)



### ------------------------------------------------------------
###                         label mapping
### ------------------------------------------------------------

# print(args_in.category_names)

if args_in.category_names:
    with open('cat_to_name.json', 'r') as f:
        cat_to_name = json.load(f)
    names = [cat_to_name[key] for key in classes]
    print("Class name:")
    print(names)
### ------------------------------------------------------------

print("Class number:")
print(classes)
print("Probability (%):")
for idx, item in enumerate(probs):
    probs[idx] = round(item*100, 2)
print(probs)


# command line usage: 
# python predict.py flowers/test/1/image_06743.jpg checkpoint.pth --gpu
# python predict.py flowers/test/1/image_06743.jpg checkpoint.pth --category_names cat_to_name.json --top_k 10
# python predict.py flowers/test/1/image_06743.jpg checkpoint.pth --category_names cat_to_name.json --top_k 10 --gpu


