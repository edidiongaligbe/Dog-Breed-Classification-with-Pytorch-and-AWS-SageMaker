# TODO: Import your dependencies.
# For instance, below are some dependencies you might need if you are using Pytorch
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


def test(model, test_loader, criterion, device):
    model.eval()
    print('Testing started')

    running_corrects = 0
    running_loss = 0
    for inputs, labels in test_loader:
        inputs = inputs.to(device)
        labels = labels.to(device)
        outputs = model(inputs)
        loss = criterion(outputs, labels)
        _, preds = torch.max(outputs, 1)
        running_loss += loss.item() * inputs.size(0)  # calculate running loss
        running_corrects += torch.sum(preds == labels.data)  # calculate running corrects

    total_loss = running_loss // len(test_loader)
    total_acc = running_corrects.double() // len(test_loader)

    print("\nTest set: Average loss: {:.4f}, Accuracy: {}\n".format(total_loss, total_acc))
    print('Testing completed')


def train(model, train_loader, validation_loader, criterion, optimizer, device):
    epochs = 40
    loss_counter = 0
    best_loss = 1e6
    image_dataset = {'train': train_loader, 'valid': validation_loader}

    for epoch in range(epochs):
        print('Training and validation started')
        for phase in ['train', 'valid']:
            if phase == 'train':
                model.train()
            else:
                model.eval()
            running_loss = 0.0
            running_corrects = 0

            for inputs, labels in image_dataset[phase]:
                inputs = inputs.to(device)
                labels = labels.to(device)
                outputs = model(inputs)
                loss = criterion(outputs, labels)

                if phase == 'train':
                    optimizer.zero_grad()
                    loss.backward()
                    optimizer.step()

                _, preds = torch.max(outputs, 1)
                running_loss += loss.item() * inputs.size(0)
                running_corrects += torch.sum(preds == labels.data)

            epoch_loss = running_loss // len(image_dataset[phase])
            epoch_acc = running_corrects // len(image_dataset[phase])

            if phase == 'valid':
                if epoch_loss < best_loss:
                    best_loss = epoch_loss
                else:
                    loss_counter += 1

            print('{} loss: {:.4f}, acc: {:.4f}, best loss: {:.4f}'.format(phase, epoch_acc, best_loss))

        if loss_counter == 1:
            break
        if epoch == 0:
            break
    print('Training and Validation Completed')
    return model


def net():
    model = models.resnet50(pretrained=True)
    # Freeze the convolutional layer
    for param in model.parameters():
        param.requires_grad = False

        # Add a fully connected layer at the end of the pretrained model.
    # Number of output class is 133, to match the number of labels in our dog breed dataset
    num_features = model.fc.in_features
    model.fc = nn.Sequential(
        nn.Dropout(p=0.2, inplace=True),
        nn.Linear(num_features, 133),
        nn.LogSoftmax(dim=1))
    return model


def create_data_loaders(data, batch_size):
    train_path = os.path.join(data, 'train')
    test_path = os.path.join(data, 'test')
    validation_path = os.path.join(data, 'valid')
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
    train_dataset = datasets.ImageFolder(root=train_path, transform=train_transform)
    train_loader = DataLoader(train_dataset, batch_size=batch_size, shuffle=True)

    test_dataset = datasets.ImageFolder(root=test_path, transform=test_transform)
    test_loader = DataLoader(test_dataset, batch_size=batch_size)

    validation_dataset = datasets.ImageFolder(root=validation_path, transform=test_transform)
    validation_loader = DataLoader(validation_dataset, batch_size=batch_size)

    return train_loader, test_loader, validation_loader


def main(args):
    device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
    print(f"Running on Device {device}")

    # Initialize a model by calling the net function
    model = net()
    model = model.to(device)

    # Create loss function and optimizer
    loss_criterion = nn.CrossEntropyLoss()
    optimizer = optim.SGD(model.parameters(), lr=args.lr)

    train_loader, test_loader, validation_loader = create_data_loaders(args.data_path, args.batch_size)

    # Call the train function to start training your model    
    model = train(model, train_loader, validation_loader, loss_criterion, optimizer, device)
    print(model)

    # Test the model to see its accuracy
    test(model, test_loader, loss_criterion, device)

    # Save the trained model
    torch.save(model.state_dict(), os.path.join(args.model_dir, "model.pth"))


if __name__ == '__main__':
    parser = argparse.ArgumentParser()

    # Specify all the hyperparameters you need to use to train your model.
    parser.add_argument("--lr", type=float, default=0.01, help="Learning rate")
    parser.add_argument('--batch_size', type=int, default=64)

    # Data, model, and output directories. Passed by sagemaker with default to os env variables
    # parser.add_argument('-o','--output-data-dir', type=str, default=os.environ['SM_OUTPUT_DATA_DIR'])
    parser.add_argument('-m', '--model_dir', type=str, default=os.environ['SM_MODEL_DIR'])
    parser.add_argument('-tr', '--data_path', type=str, default=os.environ['SM_CHANNEL_TRAIN'])
    args = parser.parse_args()
    print(args)

    main(args)
