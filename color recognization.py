from PIL import Image

def get_center_color(image_path, region_size=50):
    # Open the image
    img = Image.open(image_path)

    # Get the dimensions of the image
    width, height = img.size

    # Define the coordinates of the center
    center_x = width // 2
    center_y = height // 2

    # Define the region of interest (ROI) around the center
    left = max(0, center_x - region_size // 2)
    top = max(0, center_y - region_size // 2)
    right = min(width, center_x + region_size // 2)
    bottom = min(height, center_y + region_size // 2)

    # Crop the image to the ROI
    roi = img.crop((left, top, right, bottom))

    # Convert the ROI to RGB mode
    roi_rgb = roi.convert('RGB')

    # Get pixel data from the cropped region
    pixels = list(roi_rgb.getdata())

    # Calculate the average color
    avg_color = [
        sum(pixel[i] for pixel in pixels) // len(pixels)
        for i in range(3)
    ]

    return avg_color

# Example usage
image_path = 'image.jpg'
avg_color = get_center_color(C:\Users\Hp\OneDrive\Documents\GitHub\ENGG2112\1.jpg)
print("Average RGB color:", avg_color)
