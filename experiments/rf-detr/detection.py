from io import BytesIO
import os

from PIL import Image
from inference import get_model
import requests
import roboflow
import supervision as sv

# define the image URL
url = "https://media.roboflow.com/dog.jpeg"
image = Image.open(BytesIO(requests.get(url).content))

# get the model
roboflow.login()
model = get_model("rfdetr-base")

# predictions
predictions = model.infer(image, confidence=0.5)[0]

# detections
detections = sv.Detections.from_inference(predictions)
labels = [pred.class_name for pred in predictions.predictions]

# annotate the image
annotated_image = image.copy()
annotated_image = sv.BoxAnnotator(color=sv.ColorPalette.LEGACY).annotate(annotated_image, detections)
annotated_image = sv.LabelAnnotator(color=sv.ColorPalette.ROBOFLOW).annotate(annotated_image, detections, labels)

# save the annotated image
output_path = "annotated_output.jpg"
annotated_image.save(output_path)
print(f"Annotated image saved to: {output_path}")


