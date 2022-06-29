from PIL import Image

def preprocess(image):
    image = image.convert('1')
    image = image.resize((28,28), Image.ANTIALIAS)
    return image
