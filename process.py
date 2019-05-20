"""run after setup_asl.py and before main.py"""

import pickle
from hand3d.utils.general import plot_hand
import matplotlib.pyplot as plt
import scipy
import imageio
import os

def rgb2gray(rgb):
    r, g, b = rgb[:,:,0], rgb[:,:,1], rgb[:,:,2]
    gray = 0.2989 * r + 0.5870 * g + 0.1140 * b
    return gray

images = pickle.load(open("./pickle/images.pickle","rb"))
labels = pickle.load(open("./pickle/labels.pickle","rb"))
classes = pickle.load(open("./pickle/classes.pickle","rb"))

for i in range(len(images)):
    plt.imshow(images[i])
    plt.axis("off")
    plt.savefig("tmp.png", bbox_inches="tight")
    image = scipy.misc.imread("tmp.png")
    # image = scipy.misc.imresize(image, (28, 28))
    # image = rgb2gray(image)

    if not os.path.exists("data3/"+str(classes[labels[i]])):
        os.mkdir("data3/"+str(classes[labels[i]]))

    imageio.imwrite("data3/"+str(classes[labels[i]])+"/"+str(i)+".png",image)

    plt.clf()
