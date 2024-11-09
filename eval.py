import os
import time
import argparse
import torch
from thop import profile
from net.net import net
from torchvision import transforms
from torch.utils.data import DataLoader
from PIL import Image
from utils import *

# Argument parser setup
parser = argparse.ArgumentParser(description='PairLIE')
parser.add_argument('--input_image_path', type=str, required=True, help='Path to the input image')
parser.add_argument('--gpu_mode', type=bool, default=True, help='Use GPU if available')
parser.add_argument('--model', default='weights/PairLIE.pth', help='Path to the pretrained model')
parser.add_argument('--output_folder', type=str, default='results/', help='Folder to save the output image')
opt = parser.parse_args()

# Ensure CUDA is used if available and enabled
device = torch.device('cuda' if torch.cuda.is_available() and opt.gpu_mode else 'cpu')

# Load and preprocess the input image
def load_image(image_path, rgb_range=1):
    image = Image.open(image_path).convert('RGB')
    transform = transforms.Compose([
        transforms.ToTensor(),
        transforms.Lambda(lambda x: x * rgb_range)  # Scale pixel values to the specified range
    ])
    return transform(image).unsqueeze(0)  # Add batch dimension

print('===> Loading the input image')
input_image = load_image(opt.input_image_path).to(device)

print('===> Building model')
model = net().to(device)
model.load_state_dict(torch.load(opt.model, map_location=device))
print('Pre-trained model is loaded.')

# Evaluation function
def eval():
    torch.set_grad_enabled(False)
    model.eval()
    print('\nEvaluation:')
    
    with torch.no_grad():
        input_tensor = input_image
        output_name = os.path.basename(opt.input_image_path)
        
        L, R, X = model(input_tensor)
        D = input_tensor - X
        I = torch.pow(L, 0.2) * R  # Adjust as needed

        # Save the output image
        output_path = os.path.join(opt.output_folder, 'output_' + output_name)
        save_image(I, output_path)
        print(f'Output saved to {output_path}')

# Run the evaluation
eval()
