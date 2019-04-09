from settings import ASL_TRAIN_PATH
import tensorflow as tf
import numpy as np
import os
import pickle
import scipy.misc
from mpl_toolkits.mplot3d import Axes3D

from hand3d.nets.ColorHandPose3DNetwork import ColorHandPose3DNetwork
from hand3d.utils.general import detect_keypoints, trafo_coords, plot_hand, plot_hand_3d

# network input
image_tf = tf.placeholder(tf.float32, shape=(1, 240, 320, 3))
hand_side_tf = tf.constant([[1.0, 0.0]])  # left hand (true for all samples provided)
evaluation = tf.placeholder_with_default(True, shape=())

# build network
net = ColorHandPose3DNetwork()
hand_scoremap_tf, image_crop_tf, scale_tf, center_tf,\
keypoints_scoremap_tf, keypoint_coord3d_tf = net.inference(image_tf, hand_side_tf, evaluation)

# start tensorflow and initialize network
gpu_options = tf.GPUOptions(per_process_gpu_memory_fraction=0.8)
sess = tf.Session(config=tf.ConfigProto(gpu_options=gpu_options))
net.init(sess)

def process_data():
    """
    - read the training directory of asl images
    - preprocess the images with hand3d
    - save the images, labels, and classes as binary files
    """
    images = []
    labels = []
    classes = []

    # create the binary data folder if needed
    if not os.path.exists('./pickle'):
        os.mkdir('./pickle')

    count = 0 # each letter corresponds to an int
    for label in list(os.walk(ASL_TRAIN_PATH)): # walk directory
        full_path, image_list = label[0], label[2]
        letter = full_path[len(ASL_TRAIN_PATH)+1:] # get letter class
        if len(letter) > 0:
            # get list of file paths to each image
            image_path_list = [ASL_TRAIN_PATH+"/"+letter+"/"+file for file in image_list]
            if len(image_path_list) > 0:
                classes.append(letter)
                print(letter, count)
                # iterate each image
                for i in range(len(image_path_list)):
                    # read image and get hand from image
                    image = get_hand(image_path_list[i])
                    images.append(image)
                    labels.append(count)

                count += 1

    # write data to binary files
    with open("./pickle/images.pickle","wb") as f:
        pickle.dump(images,f)
    with open("./pickle/labels.pickle","wb") as f:
        pickle.dump(labels,f)
    with open("./pickle/classes.pickle","wb") as f:
        pickle.dump(classes,f)

def get_hand(image_path):
    """
    - read in the image from the provided path
    - process the image with hand3d and Tensorflow
    - return the finger position vectors corresponding to the hand
    """
    image_raw = scipy.misc.imread(image_path)
    image_raw = scipy.misc.imresize(image_raw, (240, 320))
    image_v = np.expand_dims((image_raw.astype('float') / 255.0) - 0.5, 0)

    hand_scoremap_v, image_crop_v, scale_v, center_v,\
    keypoints_scoremap_v, keypoint_coord3d_v = sess.run([hand_scoremap_tf, image_crop_tf, scale_tf, center_tf,
                                                         keypoints_scoremap_tf, keypoint_coord3d_tf],
                                                        feed_dict={image_tf: image_v})

    keypoint_coord3d_v = np.squeeze(keypoint_coord3d_v)
    return keypoint_coord3d_v

process_data();
