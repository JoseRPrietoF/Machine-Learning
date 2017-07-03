import os
from PIL import Image
import numpy as np

"""Returns:
X = np array with all image samples
Y = targets

* 196*196*24 original
url= http://cswww.essex.ac.uk/mv/allfaces/index.html"""
def proces_dataset_faces(path, blackAndWhite=True, width = 196, height=196):
    X = []
    Y = []
    print("Path %s" % path)
    for directory in os.listdir(path):
        print("Accesing on %s " % directory)
        for filepath in os.listdir(path+directory):

            f = path+directory+"/"+filepath
            if f.endswith(".jpg"):
                arr = image2pixelarray(f, width, height,blackWhite=blackAndWhite)
                arr = np.array(arr)
                X.append(arr.ravel())
                Y.append(directory)

    return np.array(X), np.array(Y)

def get_photos(path):
    x = []
    y = []
    for filename in os.listdir(path):

        if filename.endswith(".jpg"):
            img_path = path + filename

            arr = image2pixelarray(img_path,512,512)
            arr = np.array(arr)
            x.append(arr.ravel())
            # etiqueta:
            foo = img_path.split(".")
            etiqueta = foo[0].split("_")[-1]
            y.append(etiqueta)
    return np.array(x),np.array(y)

def image2pixelarray(filepath, w,h,blackWhite = False):
    """
    Parameters
    ----------
    filepath : str
        Path to an image file

    Returns
    -------
    list
        A list of lists which make it simple to access the greyscale value by
        im[y][x]
    """
    #print(filepath)
    if blackWhite:
        im = Image.open(filepath).convert('L')
    else:
        im = Image.open(filepath)
    im = im.resize((w, h), Image.ANTIALIAS)
    (width, height) = im.size
    greyscale_map = list(im.getdata())
    greyscale_map = np.array(greyscale_map)
    greyscale_map = greyscale_map.reshape((height, width))
    return greyscale_map
