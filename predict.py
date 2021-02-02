import argparse
import json
import torch
import numpy as np
import itertools as it
from torch import optim
from train import check_gpu, load_model, load_classifier
from torchvision import datasets, transforms, models
from PIL import Image

def arg_parser():
    parser = argparse.ArgumentParser(description='predict.py')
    parser.add_argument('--image', help='Point to image file.', default='flowers/test/49/image_06213.jpg', type=str)
    parser.add_argument('--checkpoint', help='Point to checkpoint file.', default='checkpoint.pth', type=str)
    parser.add_argument('--top_k', help='Indicate a top K number.', default=5, type=int)
    parser.add_argument('--category_names', dest='category_names', action='store', default='cat_to_name.json', type=str)
    parser.add_argument('--gpu', action='store', dest='gpu', default='gpu', type=str)

    args = parser.parse_args()

    return args


def load_checkpoint(filepath, device):

    args = arg_parser()

    # Load the file
    loaded_checkpoint = torch.load(filepath)

    model = load_model()

    for param in model.parameters():
        param.requires_grad = False

    model.classifier = load_classifier()

    # Load from checkpoint
    model.load_state_dict(loaded_checkpoint['state_dict'])
    model.class_to_idx = loaded_checkpoint['class_to_idx']

    optimizer = optim.Adam(model.classifier.parameters(), lr=0.001)

    model.to(device);

    return model


def process_image(image):
    ''' Scales, crops, and normalizes a PIL image for a PyTorch model,
        returns an Numpy array
    '''
    # TODO: Process a PIL image for use in a PyTorch model
    # Convert the image to PIL
    input_image = Image.open(image)

    # Transform the image to the right size and demensions, convert to tensor, and normalize
    transform = transforms.Compose([transforms.Resize(256),
                                    transforms.CenterCrop(224),
                                    transforms.ToTensor(),
                                    transforms.Normalize([0.485, 0.456, 0.406],
                                                         [0.229, 0.224, 0.225])])

    input_image = transform(input_image)

    return input_image



def predict(image_path, model_path, device, topk=5):
    ''' Predict the class (or classes) of an image using a trained deep learning model.
    '''
    # Implement the code to predict the class from an image file
    # Load the saved model

    args = arg_parser()

    with open(args.category_names, 'r') as f:
        cat_to_name = json.load(f)

    model = load_checkpoint(model_path, device)

    # Preprocess the image
    image = process_image(image_path)

    # Convert the image to a tensor
    image_tensor = torch.from_numpy(np.asarray(image)).type(torch.FloatTensor).to(device)
    img_add_dim = image_tensor.unsqueeze_(0)

    # Send image forward through the loaded model
    model.eval()
    with torch.no_grad():
        output = model.forward(img_add_dim)

    # Calculate probabilities
    probs = torch.exp(output)

    # Find the top 5 results
    top_probs = probs.topk(topk)[0]
    top_idx = probs.topk(topk)[1]

    # Convert top probabilities and outputs to respective lists
    top_probs_list = np.array(top_probs)[0]
    top_idx_list = np.array(top_idx[0])

    # Load index and class mappings
    class_to_idx = model.class_to_idx

    # Convert to classes
    idx_to_class = {val: key for key, val in class_to_idx.items()}

    # Create a list of classes from index of classes
    top_classes = []
    for i in top_idx_list:
        top_classes += [idx_to_class[i]]

    # Create a list of flower names from list of classes
    top_flowers = []
    for c in top_classes:
        top_flowers.append(cat_to_name[str(c)])

    return top_probs, top_flowers


def print_probabilities(img_path, model_path, device):

    probs, flowers = predict(img_path, model_path, device, topk=5)

    # Convert probabilities tensor to a list
    probs_list = probs.tolist()

    # Flatten the probabilities list
    flat_probs = list(it.chain.from_iterable(probs_list))

    # Print out the flowers and probabilities
    for f, p in zip(flowers, flat_probs):
        print("%s: %5.3f%%" % (f, p*100))



def main():

    args = arg_parser()

    device = check_gpu(gpu_arg=args.gpu)

    with open(args.category_names, 'r') as f:
        cat_to_name = json.load(f)

    img_path = args.image
    model_path = args.checkpoint

    print_probabilities(img_path, model_path, device)



if __name__ == '__main__': main()
