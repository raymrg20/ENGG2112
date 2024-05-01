import cv2

def pixelate(image, pixel_size):
    # Get the dimensions of the image
    height, width = image.shape[:2]

    # Resize the image to smaller dimensions
    small_image = cv2.resize(image, (width // pixel_size, height // pixel_size))

    # Enlarge the small image to original size
    pixelated_image = cv2.resize(small_image, (width, height), interpolation=cv2.INTER_NEAREST)

    return pixelated_image

# Load the image
image_path = "apple.jpg"
original_image = cv2.imread(image_path)

# Set the pixel size for pixelation (adjust as needed)
pixel_size = 25

# Pixelate the image
pixelated_image = pixelate(original_image, pixel_size)

# Display the original and pixelated images
cv2.imshow("Original Image", original_image)
cv2.imshow("Pixelated Image", pixelated_image)
cv2.waitKey(0)
cv2.destroyAllWindows()
