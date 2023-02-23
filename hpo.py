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


from PIL import ImageFile
ImageFile.LOAD_TRUNCATED_IMAGES = True
import argparse




def test(model, test_loader, device):
    model.eval()
    print('Testing started')

    correct = 0
    total = 0
    with torch.no_grad():
        for inputs, labels in test_loader:
            inputs, labels = inputs.to(device), labels.to(device)
            outputs = model(inputs)
            _, predicted = torch.max(outputs.data, 1)
            total += labels.size(0)
            correct += (predicted == labels).sum().item()

    print('Accuracy of the network on the test images: %d %%' % (100 * correct / total))


def train(model, train_loader, loss_fn, optimizer, device):
    epochs = 40
    model.train()
    
    for epoch in range(epochs):
        print('Training Started')
        print('Length of data: ', len(train_loader.dataset))
        running_loss = 0.0
        
        for i, (inputs, labels) in enumerate(train_loader):
            inputs, labels = inputs.to(device), labels.to(device)
            optimizer.zero_grad()
            outputs = model(inputs)
            loss = loss_fn(outputs, labels)
            loss.backward()
            optimizer.step()
            running_loss += loss.item()
            
            if i % 100 == 99:
                print('[%d, %5d] loss: %.3f' % (epoch + 1, i + 1, running_loss / 100))
                running_loss = 0.0

            #dataset_len = len(train_loader.dataset)
            #running_samples = (i + 1) * len(inputs)
            #proportion = 0.2
            #if running_samples > (proportion * dataset_len):
            #    break

    print('Training Completed')
    return model


def net():
    model = models.resnet50(pretrained=True)
    # Freeze the convolutional layer
    for param in model.parameters():
        param.requires_grad = False

    # Add a fully connected layer at the end of the pretrained model.
    # Number of output class is 133, to match the number of labels in our dog breed dataset
    num_features = model.fc.in_features
    model.fc = nn.Linear(num_features, 133)
    return model


def create_data_loaders(data, batch_size):
    train_path = os.path.join(data, 'train')
    test_path = os.path.join(data, 'test')
    #validation_path = os.path.join(data, 'valid')
    # The images are not of the same size.
    # Transform all the training images to have the same size.
    train_transform = transforms.Compose([
        transforms.Resize((224, 224)),
        transforms.RandomHorizontalFlip(),
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

    return train_loader, test_loader 


def main(args):
    device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
    print(f"Running on Device {device}")

    # Initialize a model by calling the net function
    model = net()
    model = model.to(device)

    # Create loss function and optimizer
    loss_fn = nn.CrossEntropyLoss()
    optimizer = optim.SGD(model.parameters(), lr=args.lr)

    train_loader, test_loader = create_data_loaders(args.data_path, args.batch_size)

    # Call the train function to start training your model    
    model = train(model, train_loader, loss_fn, optimizer, device)
    print(model)

    # Test the model to see its accuracy
    test(model, test_loader, device)

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
