import argparse
import torch
import matplotlib.pyplot as plt

from torch import nn
from torch.utils.data import DataLoader, Dataset

from src.dataset import get_data
from src.model import MNISTModel


parser = argparse.ArgumentParser()
parser.add_argument('--device', default='cpu', type=str)
parser.add_argument('--path', default='./best.pth', type=str)
args = parser.parse_args()


def predict(test_data: Dataset, model:nn.Module, device: str):
    model.eval()
    
    image = test_data[0][0]
    plt.imshow(image.permute(1, 2, 0))
    plt.savefig("output_image.png")
    image = image.to(device)
    image = image.unsqueeze(0)
    target = test_data[0][1]
    target = torch.tensor(target)
    
    with torch.inference_mode():
        pred = model(image)
        predicted = pred[0].argmax(0)
        actual = target
        print(f"Predicted : {predicted}, Actual : {actual}")
        

def test(device):
    num_classes = 10
    
    data_dir = 'data'
    _, test_data = get_data(data_dir)
    
    model = MNISTModel(input_shape=1, num_classes=num_classes).to(args.device)
    model.load_state_dict(torch.load(args.path))
    
    predict(test_data, model, args.device)
    
    
if __name__ == "__main__":
    test(device=args.device)
