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
import smdebug.pytorch as smd

from PIL import ImageFile
ImageFile.LOAD_TRUNCATED_IMAGES = True
import argparse

def test(model, test_loader, criterion, device, hook):
    model.eval()
    hook.set_mode(smd.modes.EVAL)
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


def train(model, train_loader, validation_loader, criterion, optimizer, device, hook):
    epochs=2
    best_loss=1e6
    image_dataset={'train':train_loader, 'valid':validation_loader}
    loss_counter=0
    
    for epoch in range(epochs):
        for phase in ['train', 'valid']:
            print(f"Epoch {epoch}, Phase {phase}")
            if phase=='train':
                model.train()
                hook.set_mode(smd.modes.TRAIN)
            else:
                model.eval()
                hook.set_mode(smd.modes.EVAL)
            running_loss = 0.0
            running_corrects = 0
            running_samples=0

            for step, (inputs, labels) in enumerate(image_dataset[phase]):
                inputs=inputs.to(device)
                labels=labels.to(device)
                outputs = model(inputs)
                loss = criterion(outputs, labels)

                if phase=='train':
                    optimizer.zero_grad()
                    loss.backward()
                    optimizer.step()

                _, preds = torch.max(outputs, 1)
                running_loss += loss.item() * inputs.size(0)
                running_corrects += torch.sum(preds == labels.data).item()
                running_samples+=len(inputs)
                if running_samples % 2000  == 0:
                    accuracy = running_corrects/running_samples
                    print("Images [{}/{} ({:.0f}%)] Loss: {:.2f} Accuracy: {}/{} ({:.2f}%)".format(
                            running_samples,
                            len(image_dataset[phase].dataset),
                            100.0 * (running_samples / len(image_dataset[phase].dataset)),
                            loss.item(),
                            running_corrects,
                            running_samples,
                            100.0*accuracy,
                        )
                    )                
                
                if running_samples>(0.2*len(image_dataset[phase].dataset)):
                    break

            epoch_loss = running_loss / running_samples
            
            if phase=='valid':
                if epoch_loss<best_loss:
                    best_loss=epoch_loss
                else:
                    loss_counter+=1

        if loss_counter==1:
            break
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
    validation_path = os.path.join(data, 'valid')
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

    validation_dataset = datasets.ImageFolder(root=validation_path, transform=test_transform)
    validation_loader = DataLoader(validation_dataset, batch_size=batch_size)

    return train_loader, test_loader, validation_loader 


def main(args):
    device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
    print(f"Running on Device {device}")

    # Initialize a model by calling the net function
    model = net()
    model = model.to(device)
    
    hook = smd.Hook.create_from_json_file()
    hook.register_module(model)

    # Create loss function and optimizer
    loss_fn = nn.CrossEntropyLoss()
    optimizer = optim.SGD(model.parameters(), lr=args.lr)

    train_loader, test_loader, validation_loader = create_data_loaders(args.data_path, args.batch_size)

    # Call the train function to start training your model    
    model = train(model, train_loader, validation_loader, loss_fn, optimizer, device, hook)
    print(model)

    # Test the model to see its accuracy
    test(model, test_loader, loss_fn, device, hook)

    # Save the trained model
    torch.save(model.state_dict(), os.path.join(args.model_dir, "model.pth"))


if __name__ == '__main__':
    parser = argparse.ArgumentParser()

    # Specify all the hyperparameters you need to use to train your model.
    parser.add_argument("--lr", type=float, default=0.01, help="Learning rate")
    parser.add_argument('--batch_size', type=int, default=64)

    # Data, model, and output directories. Passed by sagemaker with default to os env variables
    parser.add_argument('-o','--output-data-dir', type=str, default=os.environ['SM_OUTPUT_DATA_DIR'])
    parser.add_argument('-m', '--model_dir', type=str, default=os.environ['SM_MODEL_DIR'])
    parser.add_argument('-tr', '--data_path', type=str, default=os.environ['SM_CHANNEL_TRAIN'])
    args = parser.parse_args()
    print(args)

    main(args)
