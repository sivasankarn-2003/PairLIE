import os
import torch
import argparse
from torchvision import transforms
from torch.utils.data import DataLoader
from PIL import Image
from net.net import net
from utils import *

# Set GPU device if applicable
os.environ['CUDA_VISIBLE_DEVICES'] = '3'

parser = argparse.ArgumentParser(description='PairLIE')
parser.add_argument('--input_image_path', type=str, required=True, help='Path to the input image')
parser.add_argument('--gpu_mode', type=bool, default=True)
parser.add_argument('--model', default='weights/PairLIE.pth', help='Pretrained base model')  
parser.add_argument('--output_folder', type=str, default='results/')
opt = parser.parse_args()

# Ensure output folder exists
os.makedirs(opt.output_folder, exist_ok=True)

print('===> Loading image')
def load_image(image_path):
    transform = transforms.Compose([
        transforms.ToTensor(),
    ])
    image = Image.open(image_path).convert('RGB')
    image = transform(image).unsqueeze(0)  # Add batch dimension
    return image, os.path.basename(image_path)

input_image, image_name = load_image(opt.input_image_path)
print(f'Loaded image: {image_name}')

print('===> Building model')
model = net().cuda() if opt.gpu_mode else net()
model.load_state_dict(torch.load(opt.model, map_location='cuda' if opt.gpu_mode else 'cpu'))
print('Pre-trained model is loaded.')

def eval():
    torch.set_grad_enabled(False)
    model.eval()
    print('\nEvaluation:')
    
    input_image_tensor = input_image.cuda() if opt.gpu_mode else input_image
    with torch.no_grad():
        L, R, X = model(input_image_tensor)
        D = input_image_tensor - X
        I = torch.pow(L, 0.2) * R  # Adjust the exponent as needed (e.g., 0.2, LOL=0.14)
        
        # Save the processed image
        output_path = os.path.join(opt.output_folder, f'output_{image_name}')
        save_image(I, output_path)  # Implement this utility function if not present
        
        print(f'Output saved to {output_path}')

if __name__ == "__main__":
    eval()
