import torch
import torchvision.transforms as transforms
from PIL import Image

"""# Load the EDSR model
model = torch.hub.load('zhanghang1989/EDSR-PyTorch', 'edsr_x3')

# Load the input image using PIL
input_image = Image.open('input_image.jpg')

# Preprocess the image for the model
preprocess = transforms.Compose([
    transforms.Resize(128),  # Resize for model input
    transforms.ToTensor()    # Convert to tensor
])
input_tensor = preprocess(input_image).unsqueeze(0)

# Upscale the image using EDSR
with torch.no_grad():
    output_tensor = model(input_tensor)

# Save the enhanced image
output_image = transforms.ToPILImage()(output_tensor.squeeze(0))
output_image.save('enhanced_image.jpg')
"""

def enhance_photos(photo_dict):
    enhanced_photos = {}
    # Load the EDSR model
    model = torch.hub.load('zhanghang1989/EDSR-PyTorch', 'edsr_x3')

    for photo_name, photo in photo_dict.items():
        input_image = photo

        # Preprocess the image for the model
        preprocess = transforms.Compose([
            transforms.Resize(128),  # Resize for model input
            transforms.ToTensor()    # Convert to tensor
        ])
        input_tensor = preprocess(input_image).unsqueeze(0)

        # Upscale the image using EDSR
        with torch.no_grad():
            output_tensor = model(input_tensor)

        # Save the enhanced image
        output_image = transforms.ToPILImage()(output_tensor.squeeze(0))
        
        enhance_photos[photo_name]=output_image
        
    return enhanced_photos

"""def enhance_photos(photo_dict):
    enhanced_photos = {}

    for photo_name, photo in photo_dict.items():
        # Apply brightness, contrast, and sharpness enhancements
        enhanced_photo = cv2.convertScaleAbs(photo, alpha=1, beta=20)  # Adjust brightness (alpha and beta)
        enhanced_photo = cv2.convertScaleAbs(enhanced_photo, alpha=0.5, beta=0)  # Adjust contrast (alpha and beta)
        
        # Apply sharpness enhancement
        #blurred = cv2.GaussianBlur(enhanced_photo, (0, 0), 3)
        #sharpened = cv2.addWeighted(enhanced_photo, 1.5, blurred, -0.5, 0)  # Adjust parameters as needed

        enhanced_photos[photo_name] = enhanced_photo

    return enhanced_photos"""