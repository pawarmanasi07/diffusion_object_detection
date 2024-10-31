import cv2

def draw_bounding_boxes(image, predicted_bbox, target_bbox):
    # Convert image to RGB (OpenCV uses BGR by default)
    image_rgb = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)

    # Draw predicted bounding box
    cv2.rectangle(image_rgb, (int(predicted_bbox[0]), int(predicted_bbox[1])), (int(predicted_bbox[2]), int(predicted_bbox[3])), (0, 255, 0), 2)
    cv2.putText(image_rgb, "Predicted bbox", (int(predicted_bbox[0]), int(predicted_bbox[1]) - 10), cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0, 255, 0), 2)

    # Draw target bounding box
    cv2.rectangle(image_rgb, (int(target_bbox[0]), int(target_bbox[1])), (int(target_bbox[2]), int(target_bbox[3])), (255, 0, 0), 2)
    cv2.putText(image_rgb, "Ground Truth", (int(target_bbox[0]), int(target_bbox[1]) - 10), cv2.FONT_HERSHEY_SIMPLEX, 0.5, (255, 0, 0), 2)

    # Convert RGB image back to BGR
    image_bgr = cv2.cvtColor(image_rgb, cv2.COLOR_RGB2BGR)

    return image_bgr

# Example usage
image_path = "sampled_data/images/train2017/000000030163.jpg"
image = cv2.imread(image_path)

# Example predicted and target bounding boxes (x1, y1, x2, y2)
predicted_bbox = (161.875, 200.375, 300.25, 332.5)
target_bbox = (114.5199966430664, 227.72999572753906, 285.8599853515625, 307.1000061035156)

# Draw bounding boxes on the image
image_with_boxes = draw_bounding_boxes(image.copy(), predicted_bbox, target_bbox)

# Save the image with bounding boxes
cv2.imwrite("image3.jpg", image_with_boxes)
