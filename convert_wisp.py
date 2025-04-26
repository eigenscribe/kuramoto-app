import base64
import os

# Make sure the directory exists
os.makedirs("static/images", exist_ok=True)

# Read the base64 file
with open("wisp.base64", "r") as f:
    base64_data = f.read()

# Decode the base64 data to binary
image_data = base64.b64decode(base64_data)

# Write the binary data to the JPG file
with open("static/images/wisp.jpg", "wb") as f:
    f.write(image_data)

print("Conversion complete. Image saved to static/images/wisp.jpg")