import os
from PIL import Image
import numpy as np
import shutil, urllib, zipfile


"""Returns:
X = np array with all image samples
Y = targets

* 196*196*24 original
url= http://cswww.essex.ac.uk/mv/allfaces/index.html"""
def proces_dataset_faces(path, blackAndWhite=True, width = 196, height=196):
    X = []
    Y = []
    print("Path %s" % path)

    if not os.path.isdir("data"):
        os.mkdir("data")
    maybe_download(path)
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

urls = ["http://cswww.essex.ac.uk/mv/allfaces/faces94.zip",
       "http://cswww.essex.ac.uk/mv/allfaces/faces95.zip",
       "http://cswww.essex.ac.uk/mv/allfaces/faces96.zip"]
def maybe_download(path = "data/preprocess_faces"):
    if not os.path.isdir(path):
        if not os.path.isdir("tmp"):
            os.mkdir("tmp")
        for url in urls:
            file_name = url.split('/')[-1]
            fullfilename = os.path.join('tmp', file_name)
            print("Downloading: %s" % (file_name))
            u = urllib.urlretrieve(url,fullfilename)
            print("Saving %s in %s" % (file_name, fullfilename))
            zip_handler = zipfile.ZipFile(fullfilename, "r")
            dest = "tmp/"+file_name.split(".")[0]
            zip_handler.extractall(dest)
            #file_ame/file_name/female-male-other/ - name/jpgs
            copy_files(dest,path)


        shutil.rmtree('tmp')

#recursively
def copy_files(path,dest = "data/preprocess_faces/", extension="jpg"):
    has_jgp = False
    for directory in os.listdir(path):
        if directory.endswith(".jpg"):
            has_jgp = True
    if not has_jgp:
        for directory in os.listdir(path):
            full_dir = path +"/"+ directory
            copy_files(full_dir,dest,extension)
    else: #copy dir on dest
        #copy
        print("Copying %s into %s" % (path,dest))
        copy_overwritting(path, dest)

def copy_overwritting(path, dest):
    #path ex = "tmp/faces94/faces94/female/asamma
    dir_name = path.split("/")[-1]
    full_dest = os.path.join(dest, dir_name)
    print("verifying %s exists" % dest)
    if not os.path.isdir(dest):
        os.mkdir(dest)
    print("verifying %s exists" % full_dest)
    if not os.path.isdir(full_dest):
        os.mkdir(full_dest)
    for root, dirs, files in os.walk(path, topdown=False):
        for name in files:
            print("%s - %s " % (os.path.join(root, name),full_dest))
            shutil.copy(os.path.join(root, name), full_dest)

#maybe_download("foo")

#X,Y = proces_dataset_faces("data/foo")
#print(Y)