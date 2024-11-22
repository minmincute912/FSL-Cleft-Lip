import argparse
import torch
import os
from PIL import Image
from torchvision import transforms
from torch.utils.data import DataLoader
from few_shot.datasets import CleftLipDataset
from few_shot.models import get_few_shot_encoder
from few_shot.core import prepare_nshot_task
from few_shot.proto import proto_net_episode, proto_net_episode_for_inference
from config import PATH

# Device configuration
assert torch.cuda.is_available()
device = torch.device('cuda')

def load_model(checkpoint_path, num_input_channels=3):
    """ Load a trained Network Model from checkpoint.
    
    Args:
        checkpoint_path: Path to the saved model checkpoint.
        num_input_channels: Number of input channels (e.g., 3 for RGB images).

    Returns: 
        Trained model.
    """
    model = get_few_shot_encoder(num_input_channels)
    checkpoint = torch.load(checkpoint_path, map_location=device, weights_only=True)
    model.load_state_dict(checkpoint)
    model.to(device)
    model.eval()
    
    return model

def load_support_from_training_data(support_dir):
    """
    Load support set images and labels from training data directory.

    Args:
        support_dir: Path to training data directory (e.g., images_background or images_evaluation).

    Returns:
        support_images: List of PIL images.
        support_labels: List of class labels.
    """
    support_images = []
    support_labels = []
    id_to_class_name = {}

    for class_label, group_name in enumerate(sorted(os.listdir(support_dir))):
        group_path = os.path.join(support_dir, group_name)
        if not os.path.isdir(group_path):
            continue
        id_to_class_name[class_label] = group_name

        for img_name in os.listdir(group_path):
            img_path = os.path.join(group_path, img_name)
            support_images.append(Image.open(img_path).convert("RGB"))
            support_labels.append(class_label)

    return support_images, support_labels, id_to_class_name

def prepare_data(support_images, support_labels, query_images):
    """ Prepares support set and query set tensors for inference.

    Args:
        support_images: List of PIL images for the support set
        support_labels: List of class labels for the support set
        query_images: List of PIL images for the query set

    Returns:
        support_tensors, support_labels, query_tensors: Preprocessed tensors for inference.
    """
    transform = transforms.Compose([
        transforms.Resize((224, 224)),
        transforms.ToTensor(),
        transforms.Normalize(mean=[0.5], std=[0.5])
    ])
    
    support_tensors = torch.stack([transform(img) for img in support_images]).to(device)
    support_labels = torch.tensor(support_labels, dtype=torch.long).to(device)
    query_tensors = torch.stack([transform(img) for img in query_images]).to(device)
    
    return support_tensors, support_labels, query_tensors


def infer(model, support_tensors, support_labels, query_tensors, distance='l2'):
    """
    Perform inference using the Prototypical Network.
    """
    with torch.no_grad():
        predictions = proto_net_episode_for_inference(
        model=model,
        optimiser=None,
        loss_fn=None,  
        x=(support_tensors, support_labels, query_tensors),
        y=None,
        train=False,
        n_shot=support_tensors.size(0) // len(torch.unique(support_labels)),
        k_way=len(torch.unique(support_labels)),
        q_queries=query_tensors.size(0),
        distance=distance
)
    return predictions

if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument('--support_dir', required=True, help="Path to the support set directory")
    parser.add_argument('--query_dir', required=True, help="Path to the query set directory")
    parser.add_argument('--checkpoint', required=True, help="Path to the trained model checkpoint")
    parser.add_argument('--distance', default='l2', help="Distance metric to use ('l2' or 'cosine')")
    args = parser.parse_args()

    # Load the trained model
    model = load_model(args.checkpoint)

    # Load support set from training data
    support_images, support_labels, id_to_class_name = load_support_from_training_data(args.support_dir)

    # Load query set
    query_images = []
    for img_name in os.listdir(args.query_dir):
        img_path = os.path.join(args.query_dir, img_name)
        query_images.append(Image.open(img_path).convert("RGB"))

    # Prepare data for inference
    support_tensors, support_labels, query_tensors = prepare_data(support_images, support_labels, query_images)

    # Perform inference
    predictions = infer(model, support_tensors, support_labels, query_tensors, distance=args.distance)

    # Output predictions with class names
    print("Predicted classes for query images:")
    for i, prediction in enumerate(predictions):
        predicted_class_id = prediction.argmax(dim=-1).item()  
        predicted_class_name = id_to_class_name[predicted_class_id]
        print(f"Query Image {i + 1}: Class {predicted_class_name}")
