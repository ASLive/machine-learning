import cv2
import numpy as np
import os
import re
import time
import pickle
import scipy.misc
import tensorflow as tf
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

start_dir = r"C:\Users\louis\OneDrive\Documents\ASLive\video_output"
start_time = time.time()

def mkdir(directory):
	if not os.path.exists(directory):
		os.makedirs(directory.replace(" ",""))

def get_hands(phrase_file_dir):
	hands = [] # hand data for this mp4 file
	for frame_file in os.listdir(phrase_file_dir): # for each frame in mp4 file
		if frame_file.endswith(".png"):
			frame_file_path = phrase_file_dir+"\\"+frame_file
			try:
			 hands.append(get_hand(frame_file_path))
			except:
				pass
	return hands

def frames_to_hands():
	for phrase_name in os.listdir(start_dir): # for each phrase
		phrase_dir = ""
		phrase_dir = start_dir+"\\"+phrase_name
		pickle_dir = phrase_dir+"\\pickle"
		mkdir(pickle_dir)
		for phrase_file in os.listdir(phrase_dir): # for each mp4 file of the phrase
			phrase_file_dir = phrase_dir+"\\"+phrase_file
			hand_data_file_name = pickle_dir+"\\"+phrase_file+".pickle"
			print(phrase_name + " " + phrase_file)
			# get hand data
			hand_data = get_hands(phrase_file_dir)
			# write hand data to file
			with open(hand_data_file_name,"wb") as f:
				pickle.dump(hand_data, f)

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

frames_to_hands()

