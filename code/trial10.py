import argparse
import json
import matplotlib.pyplot as plt
import numpy as np
import os
from omegaconf import OmegaConf
import pandas as pd
import torch
from tqdm import tqdm
import wandb
from PIL import Image
import torch.nn.functional as F
import random
import torchvision.transforms as transforms
import torchvision.transforms.functional as TF
# import torchsummary

from archs.stable_diffusion.resnet import collect_dims
from archs.diffusion_extractor import DiffusionExtractor
from archs.aggregation_network import ObjectDetectionModel
from loss import bounding_box_distance_iou_loss

# Define paths
original_images_folder = './sampled_data/images/train2017/'
sample_train_annotations_path = './sampled_data/annotations/instances_sample_train2017.json'
original_image_size = (640, 480)

def get_rescale_size(config):
    output_size = (config["output_resolution"], config["output_resolution"])
    if "load_resolution" in config:
        load_size = (config["load_resolution"], config["load_resolution"])
    else:
        load_size = output_size
    return output_size, load_size

# Define a function to store the data
def store_data(image_id, step, pred_boxes, target_boxes, loss):
    data = {
        "image_id": image_id,
        "step": step,
        "predicted_boxes": pred_boxes,
        "target_boxes": target_boxes,
        "loss": loss
    }
    with open('data.json', 'a') as json_file:
        json.dump(data, json_file)
        json_file.write('\n')  # Add newline for each entry

    
def scale_boxes(bboxes, image_size):
    """
    Scale the predicted bounding boxes to match the target box format.
    
    Args:
        bboxes (torch.Tensor): Predicted bounding boxes with shape (N, 4) or (N, M, 4).
        image_size (tuple): Size of the original image (width, height).

    Returns:
        torch.Tensor: Scaled bounding boxes.
    """
    image_width, image_height = image_size
    bboxes[:, 0] *= image_width  # x1
    bboxes[:, 1] *= image_height  # y1
    bboxes[:, 2] *= image_width  # x2
    bboxes[:, 3] *= image_height  # y2
    print("Bounding Box Coordinates:", bboxes)
    return bboxes

def log_aggregation_network(aggregation_network, config):
    mixing_weights = torch.nn.functional.softmax(aggregation_network.mixing_weights)
    num_layers = len(aggregation_network.feature_dims)
    num_timesteps = len(aggregation_network.save_timestep)
    save_timestep = aggregation_network.save_timestep
    if config["diffusion_mode"] == "inversion":
        save_timestep = save_timestep[::-1]
    fig, ax = plt.subplots()
    ax.imshow(mixing_weights.view((num_timesteps, num_layers)).T.detach().cpu().numpy())
    ax.set_ylabel("Layer")
    ax.set_yticks(range(num_layers))
    ax.set_yticklabels(range(1, num_layers+1))
    ax.set_xlabel("Timestep")
    ax.set_xticklabels(save_timestep)
    ax.set_xticks(range(num_timesteps))
    wandb.log({f"mixing_weights": plt})

# Define get image path
def get_image_path(image_id, folder_path):
    # Assuming image_id is an integer
    image_file_name = f"{image_id:012d}.jpg"  # Assuming COCO image IDs are 12 digits
    return os.path.join(folder_path, image_file_name)

def process_image(image_pil, target_size=(512, 512), mean=(0.485, 0.456, 0.406), std=(0.229, 0.224, 0.225)):
    """
    Preprocesses an image for object detection.

    Args:
        image_pil (PIL.Image): PIL image.
        target_size (tuple, optional): Target size for resizing. Defaults to (256, 256).
        mean (tuple, optional): Mean values for normalization. Defaults to ImageNet mean.
        std (tuple, optional): Standard deviation values for normalization. Defaults to ImageNet std.

    Returns:
        tuple: Processed image tensor and original PIL image.
    """
    print("in process image")
    # Resize with aspect ratio preservation
    image_pil = TF.resize(image_pil, target_size)

    # Pad the image
    pad_left = max((target_size[0] - image_pil.size[0]) // 2, 0)
    pad_right = max(target_size[0] - image_pil.size[0] - pad_left, 0)
    pad_top = max((target_size[1] - image_pil.size[1]) // 2, 0)
    pad_bottom = max(target_size[1] - image_pil.size[1] - pad_top, 0)
    image_pil = TF.pad(image_pil, (pad_left, pad_top, pad_right, pad_bottom))

    # Convert to tensor and normalize using transforms.ToTensor()
    transform = transforms.Compose([
        transforms.Resize(target_size),  # Resize before conversion
        transforms.ToTensor(),
        transforms.Normalize(mean=mean, std=std)
    ])
    image_tensor = transform(image_pil)
    image_tensor = image_tensor.half()

    return image_tensor, image_pil

def load_coco_data(image_id, annotations, images_folder, device):
    # Load COCO annotation
    image_path = get_image_path(image_id, images_folder)
    image_pil = Image.open(image_path).convert("RGB")
    image_tensor, image_pil = process_image(image_pil)

    # Move image tensor to the specified device
    image_tensor = image_tensor.to(device)

    # Filter annotations based on the image ID
    image_annotations = [ann for ann in annotations if ann['image_id'] == image_id]

    # Convert bounding box to tensor
    target_boxes = []
    if image_annotations:
        annotation = image_annotations[0]  # Get the first annotation
        bbox = annotation["bbox"]  # [x, y, width, height]
        x1, y1, width, height = bbox
        x2, y2 = x1 + width, y1 + height
        target_boxes.append([x1, y1, x2, y2])

    print("image tensor:", image_tensor.shape, "target_boxes", target_boxes)
    
    return image_tensor, target_boxes


#Define get hyperfeats
def get_object_detection_feats(diffusion_extractor, aggregation_network, imgs):
    with torch.no_grad(), torch.autocast("cuda"):
        print("in get object detection feats")
        # Forward pass through diffusion extractor
        feats, _ = diffusion_extractor.forward(imgs.unsqueeze(0))  # Add batch dimension
        print("feats", feats.shape)
        # Reshape feature maps for aggregation network
        batch_size, num_timesteps, channels, width, height = feats.shape
        reshaped_feats = feats.float().view(batch_size, -1, width, height)
        print(reshaped_feats.shape)

        # Forward pass through aggregation network
        bboxes = aggregation_network(reshaped_feats)
        print(bboxes, bboxes.shape)

        rescaled_bboxes = scale_boxes(bboxes,original_image_size)
        rescaled_bboxes.requires_grad = True
        # Prepare hyperfeats for each image in the batch
        # img_hyperfeats = [hyperfeats[i][None, ...] for i in range(batch_size)]
        # print("img_hyperfeats: ", type(img_hyperfeats), len(img_hyperfeats))
        # return img_hyperfeats
        return rescaled_bboxes

def bounding_box_loss(pred_bboxes, target_boxes):

    # Bounding box regression loss using smooth L1 loss
    bbox_loss = F.smooth_l1_loss(pred_bboxes, target_boxes, reduction='mean')

    return bbox_loss

def load_models(config_path):
    config = OmegaConf.load(config_path)
    config = OmegaConf.to_container(config, resolve=True)
    device = config.get("device", "cuda")
    diffusion_extractor = DiffusionExtractor(config, device)
    dims = config.get("dims")
    print(dims)
    if dims is None:
        dims = collect_dims(diffusion_extractor.unet, idxs=diffusion_extractor.idxs)
        print(dims)
    if config.get("flip_timesteps", False):
        config["save_timestep"] = config["save_timestep"][::-1]
    aggregation_network = ObjectDetectionModel(
            feature_dims=dims,
            device=device,
            save_timestep=config["save_timestep"],
            num_timesteps=config["num_timesteps"]
    )
    return config, diffusion_extractor, aggregation_network

def save_model(config, aggregation_network, optimizer, step):
    results_folder = os.path.join(config['results_folder'], wandb.run.name)
    os.makedirs(results_folder, exist_ok=True)
    filename = f"checkpoint_step_{step:08d}.pt"
    torch.save({
        "step": step,
        "config": config,
        "model_state_dict": aggregation_network.state_dict(),
        "optimizer_state_dict": optimizer.state_dict()
    }, os.path.join(results_folder, filename))
    print(f"Model checkpoint saved at: {os.path.join(results_folder, filename)}")


def train(config, diffusion_extractor, aggregation_network, optimizer, train_anns):
    device = config.get("device", "cuda")
    # output_size, load_size = get_rescale_size(config)
    # np.random.seed(0)

    losses = []  # List to store losses

    for epoch in range(config["max_epochs"]):
        # Shuffle the train_anns list in place
        random.shuffle(train_anns["images"])
        # Take a subset of train_anns based on max_steps_per_epoch
        epoch_train_anns = train_anns["images"][:config["max_steps_per_epoch"]]
        for i, ann in enumerate(epoch_train_anns):
            step = epoch * config["max_steps_per_epoch"] + i
            optimizer.zero_grad()
            print(i, ann)
            image_id = ann["id"]
            annotations = train_anns["annotations"]
            # Load image and target bounding boxes
            image_tensor, target_boxes = load_coco_data(image_id, annotations, original_images_folder, device)
            print(image_tensor.shape, len(target_boxes))

            # Get object detection features
            pred_bboxes = get_object_detection_feats(diffusion_extractor, aggregation_network, image_tensor)
            print(pred_bboxes)

            target_boxes = torch.tensor(target_boxes, dtype=torch.float32, device=pred_bboxes.device)
            # Flatten the predictions and targets
            pred_bboxes = pred_bboxes.view(-1, 4)  # Shape: (B * num_anchors * H * W, 4)
            target_boxes = target_boxes.view(-1, 4)  # Shape: (B * num_anchors * H * W, 4)

            # # Calculate loss
            loss = bounding_box_loss(pred_bboxes, target_boxes)
            print(loss)

            store_data(image_id, step, pred_bboxes.tolist(), target_boxes.tolist(), loss.item())

            # # Backpropagation
            optimizer.zero_grad()
            print(loss.backward())
            optimizer.step()

            losses.append(loss.item())

            wandb.log({"train/loss": loss.item()}, step=step)
            # if step > 0 and config["val_every_n_steps"] > 0 and step % config["val_every_n_steps"] == 0:
            #     with torch.no_grad():
            #         log_aggregation_network(aggregation_network, config)
            #         save_model(config, aggregation_network, optimizer, step)

    print(losses)

def main(args):
    torch.manual_seed(0)
    np.random.seed(0)
    random.seed(0)
    config, diffusion_extractor, aggregation_network = load_models(args.config_path)
    print(aggregation_network)

    # torchsummary.summary(aggregation_network)
    # print("Done with loading config")

    wandb.init(project=config["wandb_project"], name=config["wandb_run"])
    wandb.run.name = f"{str(wandb.run.id)}_{wandb.run.name}"
    parameter_groups = [
            {"params": aggregation_network.backbone.mixing_weights, "lr": config["lr"]},
            {"params": aggregation_network.backbone.bottleneck_layers.parameters(), "lr": config["lr"]},
            {"params": aggregation_network.baseModel.parameters(), "lr": config["lr"]},
            {"params": aggregation_network.regressor.parameters(), "lr": config["lr"]}
        ]
    optimizer = torch.optim.AdamW(parameter_groups, weight_decay=config["weight_decay"])
    print("Done with Optimizer")

    if config.get("train_path"):
        # assert config["batch_size"] == 2, "The loss computation compute_clip_loss assumes batch_size=2."
        train_json = json.load(open(config["train_path"]))
        # train_anns = train_json["images"]

        # print("Type of train_anns:", type(train_anns))
        # print("Length of train_anns:", len(train_anns))
        # print("Sample annotation:", train_anns[0])  # Print a sample annotation to verify its structure

        train(config, diffusion_extractor, aggregation_network, optimizer, train_json)
    else:
        print("In valpath else")

if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="")
    parser.add_argument("--config_path", type=str, help="Path to yaml config file", default="configs/train.yaml")
    args = parser.parse_args()
    main(args)