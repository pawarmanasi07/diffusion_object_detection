import torch
import numpy as np
from sklearn.metrics import average_precision_score
from sklearn.metrics import precision_recall_fscore_support

def calculate_iou(boxes1, boxes2):
    """
    Calculate IoU (Intersection over Union) between two sets of bounding boxes.

    Args:
        boxes1 (torch.Tensor): Predicted bounding boxes with shape (N, 4) or (N, M, 4).
        boxes2 (torch.Tensor): Target bounding boxes with shape (N, 4) or (N, M, 4).

    Returns:
        torch.Tensor: IoU values with shape (N,) or (N, M).
    """
    # Compute intersection coordinates
    x1 = torch.max(boxes1[:, 0], boxes2[:, 0])
    y1 = torch.max(boxes1[:, 1], boxes2[:, 1])
    x2 = torch.min(boxes1[:, 2], boxes2[:, 2])
    y2 = torch.min(boxes1[:, 3], boxes2[:, 3])

    # Compute intersection area
    intersection = torch.clamp(x2 - x1, min=0) * torch.clamp(y2 - y1, min=0)

    # Compute union area
    area1 = (boxes1[:, 2] - boxes1[:, 0]) * (boxes1[:, 3] - boxes1[:, 1])
    area2 = (boxes2[:, 2] - boxes2[:, 0]) * (boxes2[:, 3] - boxes2[:, 1])
    union = area1 + area2 - intersection

    # Compute IoU
    iou = intersection / union

    return iou


def calculate_distance(boxes1, boxes2):
    """
    Calculate the normalized distance between the center points of two sets of bounding boxes.

    Args:
        boxes1 (torch.Tensor): Predicted bounding boxes with shape (N, 4) or (N, M, 4).
        boxes2 (torch.Tensor): Target bounding boxes with shape (N, 4) or (N, M, 4).

    Returns:
        torch.Tensor: Normalized distance values with shape (N,) or (N, M).
    """
    # Compute center coordinates of boxes
    center_x1 = (boxes1[:, 0] + boxes1[:, 2]) / 2
    center_y1 = (boxes1[:, 1] + boxes1[:, 3]) / 2
    center_x2 = (boxes2[:, 0] + boxes2[:, 2]) / 2
    center_y2 = (boxes2[:, 1] + boxes2[:, 3]) / 2

    # Compute distance between center points
    distance = torch.sqrt((center_x1 - center_x2) ** 2 + (center_y1 - center_y2) ** 2)

    # Normalize distance by dividing with image diagonal
    image_diagonal = torch.sqrt((boxes1[:, 2] - boxes1[:, 0]) ** 2 + (boxes1[:, 3] - boxes1[:, 1]) ** 2)
    normalized_distance = distance / image_diagonal

    return normalized_distance


def bounding_box_distance_iou_loss(pred_bboxes, target_boxes, lambda_distance=0.5):
    """
    Compute Distance IoU loss between predicted and target bounding boxes.

    Args:
        pred_bboxes (torch.Tensor): Predicted bounding boxes with shape (N, 4) or (N, M, 4).
        target_boxes (torch.Tensor): Target bounding boxes with shape (N, 4) or (N, M, 4).
        lambda_distance (float): Weighting factor for the distance term. Defaults to 0.5.

    Returns:
        torch.Tensor: Distance IoU loss value.
    """
    target_boxes = torch.tensor(target_boxes, dtype=torch.float32, device=pred_bboxes.device)
    # Flatten the predictions and targets
    pred_bboxes = pred_bboxes.view(-1, 4)  # Shape: (B * num_anchors * H * W, 4)
    target_boxes = target_boxes.view(-1, 4)  # Shape: (B * num_anchors * H * W, 4)
    
    iou = calculate_iou(pred_bboxes, target_boxes)
    distance = calculate_distance(pred_bboxes, target_boxes)

    # Compute Distance IoU loss
    distance_iou_loss = 1 - iou + lambda_distance * distance

    return distance_iou_loss.mean()


def calculate_iou(boxes1, boxes2):
    """
    Calculate IoU (Intersection over Union) between two sets of bounding boxes.

    Args:
        boxes1 (torch.Tensor): Predicted bounding boxes with shape (N, 4) or (N, M, 4).
        boxes2 (torch.Tensor): Target bounding boxes with shape (N, 4) or (N, M, 4).

    Returns:
        torch.Tensor: IoU values with shape (N,) or (N, M).
    """
    # Compute intersection coordinates
    x1 = torch.max(boxes1[:, 0], boxes2[:, 0])
    y1 = torch.max(boxes1[:, 1], boxes2[:, 1])
    x2 = torch.min(boxes1[:, 2], boxes2[:, 2])
    y2 = torch.min(boxes1[:, 3], boxes2[:, 3])

    # Compute intersection area
    intersection = torch.clamp(x2 - x1, min=0) * torch.clamp(y2 - y1, min=0)

    # Compute union area
    area1 = (boxes1[:, 2] - boxes1[:, 0]) * (boxes1[:, 3] - boxes1[:, 1])
    area2 = (boxes2[:, 2] - boxes2[:, 0]) * (boxes2[:, 3] - boxes2[:, 1])
    union = area1 + area2 - intersection

    # Compute IoU
    iou = intersection / union

    return iou


def bounding_box_iou_loss(pred_bboxes, target_boxes):
    """
    Compute IoU (Intersection over Union) loss between predicted and target bounding boxes.

    Args:
        pred_bboxes (torch.Tensor): Predicted bounding boxes with shape (N, 4) or (N, M, 4).
        target_boxes (torch.Tensor): Target bounding boxes with shape (N, 4) or (N, M, 4).

    Returns:
        torch.Tensor: IoU loss value.
    """
    target_boxes = torch.tensor(target_boxes, dtype=torch.float32, device=pred_bboxes.device)
    # Flatten the predictions and targets
    pred_bboxes = pred_bboxes.view(-1, 4)  # Shape: (B * num_anchors * H * W, 4)
    target_boxes = target_boxes.view(-1, 4)  # Shape: (B * num_anchors * H * W, 4)

    iou = calculate_iou(pred_bboxes, target_boxes)

    # Compute IoU loss
    iou_loss = 1 - iou.mean()  # You can customize the loss calculation based on your requirements

    return iou_loss


def calculate_mAP(pred_boxes, target_boxes, threshold=0.5):
    # Convert to numpy arrays
    pred_boxes = np.array(pred_boxes)
    target_boxes = np.array(target_boxes)

    # Compute average precision for each class
    average_precisions = []
    for class_idx in range(num_classes):
        y_true = np.zeros(len(target_boxes))
        y_scores = np.zeros(len(pred_boxes))
        for i, pred_box in enumerate(pred_boxes):
            iou = calculate_iou(pred_box, target_boxes[i])
            if iou >= threshold:
                y_scores[i] = 1
        for i, target_box in enumerate(target_boxes):
            iou = calculate_iou(pred_boxes[i], target_box)
            if iou >= threshold:
                y_true[i] = 1
        average_precisions.append(average_precision_score(y_true, y_scores))
    mAP = np.mean(average_precisions)
    return mAP



def calculate_precision_recall_f1(pred_boxes, target_boxes, threshold=0.5):
    # Convert to numpy arrays
    pred_boxes = np.array(pred_boxes)
    target_boxes = np.array(target_boxes)

    y_true = np.zeros(len(target_boxes))
    y_pred = np.zeros(len(pred_boxes))

    for i, pred_box in enumerate(pred_boxes):
        for j, target_box in enumerate(target_boxes):
            iou = calculate_iou(pred_box, target_box)
            if iou >= threshold:
                y_pred[i] = 1
                y_true[j] = 1

    precision, recall, f1, _ = precision_recall_fscore_support(y_true, y_pred, average='binary')

    return precision, recall, f1
