import torch
from torchvision import models, transforms
from PIL import Image
import matplotlib.pyplot as plt
import json
import requests  # Add the missing import

# Load the pre-trained model with the recommended weights
model = models.resnet50(weights=models.ResNet50_Weights.DEFAULT)
model.eval()

# Define the image transformations
preprocess = transforms.Compose([
    transforms.Resize(256),
    transforms.CenterCrop(224),
    transforms.ToTensor(),
    transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225]),
])

# Load the image
img_path = 'charlesdeluvio-Mv9hjnEUHR4-unsplash.jpg'  # Replace with your image path
input_image = Image.open(img_path)

# Preprocess the image
input_tensor = preprocess(input_image)
input_batch = input_tensor.unsqueeze(0)  # Create a mini-batch as expected by the model

# Move the input and model to GPU for speed if available
if torch.cuda.is_available():
    input_batch = input_batch.to('cuda')
    model.to('cuda')

with torch.no_grad():
    output = model(input_batch)

# Load ImageNet class labels
LABELS_URL = "https://raw.githubusercontent.com/anishathalye/imagenet-simple-labels/master/imagenet-simple-labels.json"
labels = json.loads(requests.get(LABELS_URL).text)

# The output has unnormalized scores. To get probabilities, you can run a softmax on it.
probabilities = torch.nn.functional.softmax(output[0], dim=0)

# Show the top 5 predicted labels
top5_prob, top5_catid = torch.topk(probabilities, 5)
for i in range(top5_prob.size(0)):
    print(labels[top5_catid[i]], top5_prob[i].item())

# Visualize the image
plt.imshow(input_image)
plt.title("Image Classification using ResNet50")
plt.show()

