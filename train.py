import argparse
import torch
import json
from collections import OrderedDict
from torch import nn
from torch import optim
from torchvision import datasets, transforms, models

with open('cat_to_name.json', 'r') as f:
        cat_to_name = json.load(f)

def arg_parser():
    parser = argparse.ArgumentParser(description='train.py')
    parser.add_argument('--arch', dest='arch', action='store', default='vgg16',
                        help='You can choose from vgg11, vgg13, vgg16, and vgg19.',type = str)
    parser.add_argument('--save_dir', dest='save_dir', action='store', default='./checkpoint.pth', type = str)
    parser.add_argument('--learning_rate', dest='learning_rate', action='store', default=0.001, type = float)
    parser.add_argument('--hidden_units', dest='hidden_units', action='store', default=512, type=int)
    parser.add_argument('--epochs', dest='epochs', action='store', default=5, type=int)
    parser.add_argument('--gpu', dest='gpu', action='store', default='gpu', type = str)

    args = parser.parse_args()

    return args

def check_gpu(gpu_arg):
    if not gpu_arg:
        return torch.device('cpu')

    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

    return device



def load_model(state_dict='vgg16'):
    model_options = {
        'vgg11': models.vgg11,
        'vgg13': models.vgg13,
        'vgg16': models.vgg16,
        'vgg19': models.vgg19
    }

    get_model = model_options.get(state_dict, None)

    if not get_model:
        print('Please choose a valid model option: {}'.format(','.join(model_options.keys())))

    loaded_model = get_model(pretrained=True)

    # Freeze parameters so we don't backdrop through them
    for param in loaded_model.parameters():
        param.requires_grad = False

    return loaded_model



def load_classifier():

  loaded_classifier = nn.Sequential(OrderedDict([
  ('fc1', nn.Linear(25088, 4096)),
  ('relu1', nn.ReLU()),
  ('dropout1', nn.Dropout(0.2)),
  ('fc2', nn.Linear(4096, len(cat_to_name))),
  ('output', nn.LogSoftmax(dim=1))]))

  return loaded_classifier



def validation(model, loader, device):
    total = 0
    correct = 0

    with torch.no_grad():
        model.eval()

        for data in loader:
            images, labels = data
            images, labels = images.to(device), labels.to(device)
            # Get probabilities
            outputs = model(images)
            # Calculate predictions
            _, predicted = torch.max(outputs.data, 1)
            # Total number of images
            total += labels.size(0)
            # Total number of correct predictions
            correct += (predicted == labels).sum().item()

    print(f"Model accuracy on test set: {round((100 * correct)/total, 1)}%")



def model_trainer(model, optimizer, trainloader, testloader, criterion, device):
    # Learning method

    args = arg_parser()

    epochs = args.epochs
    steps = 0
    running_loss = 0
    print_every = 25

    print("Let's start training...\n")

    for epoch in range(epochs):
        for inputs, labels in trainloader:
            steps += 1
            # Move input and label tensors to the default device
            inputs, labels = inputs.to(device), labels.to(device)

            optimizer.zero_grad()  # Zeroes out the gradients

            # Forward and backward passes through the model
            logps = model.forward(inputs)
            loss = criterion(logps, labels)
            loss.backward()
            optimizer.step()

            running_loss += loss.item()

            if steps % print_every == 0:
                test_loss = 0
                accuracy = 0
                model.eval()  # Turns off dropout
                with torch.no_grad():
                    for inputs, labels in testloader:
                        inputs, labels = inputs.to(device), labels.to(device)
                        logps = model.forward(inputs)
                        batch_loss = criterion(logps, labels)

                        test_loss += batch_loss.item()

                        # Calculate accuracy
                        ps = torch.exp(logps) # Calculates actual probabilities
                        top_p, top_class = ps.topk(1, dim=1)
                        equals = top_class == labels.view(*top_class.shape)
                        accuracy += torch.mean(equals.type(torch.FloatTensor)).item()

                print(f"Epoch {epoch+1}/{epochs}.. "
                      f"Train loss: {running_loss/print_every:.3f}.. "
                      f"Test loss: {test_loss/len(testloader):.3f}.. "
                      f"Test accuracy: {100*accuracy/len(testloader):.3f}%")
                running_loss = 0
                model.train()

    print('\nTraining is complete!')

    return model



# TODO: Save the checkpoint
def save_checkpoint(model, optimizer, train_data):

    args = arg_parser()
    model.class_to_idx = train_data.class_to_idx
    checkpoint = {'state_dict': model.state_dict(),
                  'class_to_idx': model.class_to_idx}

    torch.save(checkpoint, args.save_dir)

    print('Checkpoint saved.')


def main():

    # Get Keyword Args for Training
    args = arg_parser()

    with open('cat_to_name.json', 'r') as f:
        cat_to_name = json.load(f)

    # Create data directories
    data_dir = 'flowers'
    train_dir = data_dir + '/train'
    valid_dir = data_dir + '/valid'
    test_dir = data_dir + '/test'

    # Pass transforms in, then create trainloader
    train_transforms = transforms.Compose([transforms.RandomRotation(30),
                                       transforms.RandomResizedCrop(224),
                                       transforms.RandomHorizontalFlip(),
                                       transforms.ToTensor(),
                                       transforms.Normalize([0.485, 0.456, 0.406],
                                                            [0.229, 0.224, 0.225])])

    test_transforms = transforms.Compose([transforms.Resize(255),
                                      transforms.CenterCrop(224),
                                      transforms.ToTensor(),
                                      transforms.Normalize([0.485, 0.456, 0.406],
                                                           [0.229, 0.224, 0.225])])

    val_transforms = transforms.Compose([transforms.Resize(255),
                                      transforms.CenterCrop(224),
                                      transforms.ToTensor(),
                                      transforms.Normalize([0.485, 0.456, 0.406],
                                                           [0.229, 0.224, 0.225])])

    # Load the datasets with ImageFolder
    train_data = datasets.ImageFolder(data_dir + '/train', transform=train_transforms)
    test_data = datasets.ImageFolder(data_dir + '/test', transform=test_transforms)
    val_data = datasets.ImageFolder(data_dir + '/valid', transform=val_transforms)

    # Using the image datasets and the trainforms, define t
    trainloader = torch.utils.data.DataLoader(train_data, batch_size=50, shuffle=True)
    testloader = torch.utils.data.DataLoader(test_data, batch_size=50)
    valloader = torch.utils.data.DataLoader(val_data, batch_size=50)

    model = load_model(state_dict=args.arch)

    model.classifier = load_classifier()

    device = check_gpu(gpu_arg=args.gpu)

    criterion = nn.NLLLoss()

    optimizer = optim.Adam(model.classifier.parameters(), lr=args.learning_rate)

    model.to(device);

    model_trainer(model, optimizer, trainloader, testloader, criterion, device)

    validation(model, valloader, device)

    save_checkpoint(model, optimizer, train_data)

if __name__ == '__main__': main()
