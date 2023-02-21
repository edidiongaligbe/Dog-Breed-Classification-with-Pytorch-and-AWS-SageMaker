#TODO: Import your dependencies.
#For instance, below are some dependencies you might need if you are using Pytorch
import os
import numpy as np
import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import DataLoader
import torchvision
import torchvision.models as models
import torchvision.transforms as transforms
import torchvision.datasets as datasets

import argparse

def test(model, test_loader):
    model.eval()
    print('Testing started')
    correct = 0
    with torch.no_grad():
        for data, target in test_loader:
            data = data.view(data.shape[0], -1)
            output = model(data)
            pred = output.argmax(dim=1, keepdim=True)  
            correct += pred.eq(target.view_as(pred)).sum().item()

    print(f'Test set: Accuracy: {correct}/{len(test_loader.dataset)} = {100*(correct/len(test_loader.dataset))}%)')
    print('Testing completed')
    

def train(model, train_loader, criterion, optimizer, epoch):
    model.train()
    print('Training started')
    for e in range(epoch):
        running_loss=0
        correct=0
        for step, (data, target) in enumerate(train_loader):
            data = data.view(data.shape[0], -1)
            optimizer.zero_grad()
            pred = model(data)
            loss = criterion(pred, target)
            running_loss+=loss
            loss.backward()
            optimizer.step()
            pred=pred.argmax(dim=1, keepdim=True)
            correct += pred.eq(target.view_as(pred)).sum().item()
            dataset_length = len(train_loader.dataset)
            running_samples = (step + 1) * len(data)
            proportion = 0.4 # we will use 40% of the dataset
            if running_samples>(proportion*dataset_length):
                break
        print(f"Epoch {e}: Loss {running_loss/len(train_loader.dataset)}, Accuracy {100*(correct/len(train_loader.dataset))}%")
        print('Training Completed')
    
def net():
    model = models.resnet50(pretrained=True)
    #Freeze the convolutional layer
    for param in model.parameters():
        param.requires_grad = False   
    
    #Add a fully connected layer at the end of the pretrained model.
    #Number of output class is 133, to match the number of labels in our dog breed dataset
    num_features=model.fc.in_features
    model.fc = nn.Sequential(
        nn.Dropout(p=0.2, inplace=True),
        nn.Linear(num_features, 133),
        nn.LogSoftmax(dim=1))
    return model

def create_data_loaders(data, batch_size):
    '''
    This is an optional function that you may or may not need to implement
    depending on whether you need to use data loaders or not
    '''
    pass

def main(args):
    # The images are not of the same size.
    # Transform all the training images to have the same size.
    train_transform = transforms.Compose([
        transforms.Resize((224, 224)),
        transforms.RandomHorizontalFlip(p=0.5),
        transforms.RandomVerticalFlip(p=0.5),
        transforms.GaussianBlur(kernel_size=(5, 9), sigma=(0.1, 5)),
        transforms.RandomRotation(degrees=(30, 70)),
        transforms.ToTensor(),
        transforms.Normalize(
            mean=[0.485, 0.456, 0.406],
            std=[0.229, 0.224, 0.225]
        )
    ])

    # Transform all the testing images to have the same size.
    test_transform = transforms.Compose([
        transforms.Resize((224, 224)),
        transforms.ToTensor(),
        transforms.Normalize(
            mean=[0.485, 0.456, 0.406],
            std=[0.229, 0.224, 0.225]
        )
    ])
    # Load training and test data
    train_dataset = datasets.ImageFolder(
        root=args.train_path,
        transform=train_transform
    )
    test_dataset = datasets.ImageFolder(
        root=args.test_path,
        transform=test_transform
    )
    train_loader = DataLoader(
        train_dataset, batch_size=args.batch_size, shuffle=True,
        num_workers=2, pin_memory=True
    )
    test_loader = DataLoader(
        test_dataset, batch_size=args.batch_size, shuffle=False,
        num_workers=2, pin_memory=True
    )

    device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
    print(f"Running on Device {device}")

    # Initialize a model by calling the net function
    model=net()
    model=model.to(device)

    #Create loss function and optimizer
    loss_criterion = nn.CrossEntropyLoss()
    optimizer = optim.SGD(model.parameters(), lr=args.lr)

    #Call the train function to start training your model    
    model=train(model, train_loader, loss_criterion, optimizer, args.num_epoch)
    print(model)

    #Test the model to see its accuracy
    test(model, test_loader, loss_criterion)

    #Save the trained model
    torch.save(model.state_dict(), os.path.join(args.model_dir, "model.pth"))

if __name__=='__main__':
    parser=argparse.ArgumentParser()

    #Specify all the hyperparameters you need to use to train your model.
    parser.add_argument("--lr", type=float, default=0.01, help="Learning rate")
    parser.add_argument('--batch_size', type=int, default=64)
    parser.add_argument('--num_epoch', type=int, default=10)

    # Data, model, and output directories. Passed by sagemaker with default to os env variables
    #parser.add_argument('-o','--output-data-dir', type=str, default=os.environ['SM_OUTPUT_DATA_DIR'])
    parser.add_argument('-m','--model_dir', type=str, default=os.environ['SM_MODEL_DIR'])
    parser.add_argument('-tr','--train_path', type=str, default=os.environ['SM_CHANNEL_TRAIN'])
    parser.add_argument('-te','--test_path', type=str, default=os.environ['SM_CHANNEL_TEST'])
    args=parser.parse_args()
    print(args)
    
    main(args)
