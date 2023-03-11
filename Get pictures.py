import requests
from tqdm import tqdm
from PIL import Image
from io import BytesIO

# create a progress bar for the range of URLs to download
for i in tqdm(range(1, 10000), desc="Downloading images"):
    url = f"https://www.autoshkollaonline.com/kosove/images/{i}.jpg"
    response = requests.get(url)

    # check if the response content is an actual image
    content_type = response.headers.get("content-type")
    if "image" not in content_type:
        print(f"Skipped {url} - content is not an image", end="\r", flush=True)
        print("\x1b[1A", end="")
        continue

    # read the image content from the response
    image_content = response.content

    # check if the image is valid using Pillow library
    try:
        Image.open(BytesIO(image_content)).verify()
    except:
        print("\n")
        print(f"Skipped {url} - content is not a valid image", end="\r", flush=True)
        print("\x1b[1A", end="")
        print("\x1b[1A", end="")
        continue

    # write the image content to a file
    with open(f"image_{i}.jpg", "wb") as f:
        f.write(image_content)