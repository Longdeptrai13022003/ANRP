import cv2

# Load the image
image = cv2.imread(r'C:\Users\trand\OneDrive\Pictures\Screenshots\ok2.png')

# Convert the image to its negative
negative_image = cv2.bitwise_not(image)
# Define the path to save the output
output_path = r'C:\Users\trand\OneDrive\Pictures\Screenshots\ok2_negative.png'

# Save the negative image
cv2.imwrite(output_path, negative_image)

# Display the original and negative images
cv2.imshow('Original Image', image)
cv2.imshow('Negative Image', negative_image)

# Wait for a key press and close the windows
cv2.waitKey(0)
cv2.destroyAllWindows()
