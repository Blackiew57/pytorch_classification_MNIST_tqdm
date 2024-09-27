import torch
import argparse

from torch.utils.data import DataLoader
from torchvision import transforms
from torch import nn
from torch.optim import Optimizer
from tqdm.auto import tqdm
from time import sleep

from src.model import MNISTModel
from src.dataset import get_data
from helper_functions import accuracy_fn

parser = argparse.ArgumentParser()
parser.add_argument("--device", default='cpu', type=str)
parser.add_argument("--batch", default='64', type=int)
parser.add_argument("--epoch", default="20", type=int)
parser.add_argument("--lr", default="1e-3", type=float)
parser.add_argument('--path', default='./best.pth', type=str)
args = parser.parse_args()


def train(device: str):
    batch_size = args.batch
    epochs = args.epoch
    lr = args.lr
    best_acc = 0
    
    data_dir = 'data'
    train_data, test_data = get_data(data_dir)
    
    train_loader = DataLoader(train_data, batch_size=batch_size, shuffle=True)
    test_loader = DataLoader(test_data, batch_size=batch_size, shuffle=False)
    
    model = MNISTModel(input_shape= 1, num_classes=10)
    model.to(args.device)
    
    loss_fn = nn.CrossEntropyLoss()
    optimizer = torch.optim.SGD(model.parameters(), lr=lr)
    
    for epoch in range(epochs):
        model.train()
        train_loss = 0.0
        
        with tqdm(train_loader, desc=f"Epoch {epoch+1}/{epochs}", unit="batch") as tepoch:
            for batch, (X, y) in enumerate(tepoch):
                X, y = X.to(args.device), y.to(args.device)
                
                optimizer.zero_grad()
                y_pred = model(X)
                loss = loss_fn(y_pred, y)
                loss.backward()
                optimizer.step()
                
                train_loss += loss.item()
                tepoch.set_postfix(loss=loss.item())
        
        train_loss /= len(train_loader)
        
        model.eval()
        test_loss, test_acc = 0.0, 0.0
        
        with torch.inference_mode():
            for X, y in test_loader:
                X, y = X.to(args.device), y.to(args.device)
                test_pred = model(X)
                
                test_loss += loss_fn(test_pred, y).item()
                test_acc += accuracy_fn(y_true=y, y_pred=test_pred.argmax(dim=1))
        
        test_loss /= len(test_loader)
        test_acc /= len(test_loader)
        
        print(f"\nTrain loss: {train_loss:.5f} | Test loss: {test_loss:.5f}, Test accuracy: {test_acc:.2f}%\n")    

        if test_acc > best_acc:
            best_acc = test_acc
            torch.save(model.state_dict(), args.path)
            print(f"Model saved at {args.path}, best accuracy: {best_acc:.2f}%\n")
            
            
if __name__ == "__main__":
    train(device=args.device)

