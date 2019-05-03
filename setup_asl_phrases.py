import cv2
import numpy as np
import os
import re
import time
import pickle
import scipy.misc
import tensorflow as tf
from settings import ASL_FRAME_PATH, ASL_TRAIN_VIDEO_PATH
from hand3d.nets.ColorHandPose3DNetwork import ColorHandPose3DNetwork
from hand3d.utils.general import detect_keypoints, trafo_coords, plot_hand, plot_hand_3d

FPS = 29.97 # Frame rate for Youtube ASL Music Videos

start_time = time.time()

def mkdir(directory):
	if not os.path.exists(directory):
		os.makedirs(directory.replace(" ",""))

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

def mkdir(directory):
	if not os.path.exists(directory):
		os.makedirs(directory.replace(" ",""))

def get_hands(phrase_file_dir):
	"""
	- helper func for frames_to_hands()
	- get the hand3d coords far a particular frame folder
	"""
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
	"""process each frame into the hand3d coords"""
	for phrase_name in os.listdir(ASL_FRAME_PATH): # for each phrase
		phrase_dir = ""
		phrase_dir = ASL_FRAME_PATH+"\\"+phrase_name
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

def frame_capture(file):
	"""
	input: video file
	output: list of frames
	"""
	frames = [] # return value

	# open video file
	cap = cv2.VideoCapture(file)
	cap.set(cv2.CAP_PROP_FPS, FPS)

	while(True):
		ret, frame = cap.read() # capture frame
		frames.append(frame) # add to return value

		if not ret:
			print("Reading time: %s seconds" % (time.time()-start_time))
			break

	# close video file
	cap.release()
	cv2.destroyAllWindows()

	return frames

def read_frames():
	"""read frames of files of phrases"""
	data = {} # {phrase1_name: [[file1_frames], [file2_frames],..],...}
	for phrase_name in os.listdir(ASL_TRAIN_VIDEO_PATH): # for each phrase
		data[phrase_name] = []
		phrase_dir = ""
		phrase_dir = ASL_TRAIN_VIDEO_PATH+"\\"+phrase_name
		print("Reading: "+phrase_name)
		for phrase_file in os.listdir(phrase_dir): # for each file of the phrase
			if phrase_file.endswith(".mp4"):
				frames = frame_capture(phrase_dir + "\\"+ phrase_file)
				data[phrase_name].append(frames) # get the phrase of the file

	return data

def write_frames(data):
	"""write frames to proper file structure"""
	mkdir(ASL_FRAME_PATH)
	for phrase_name in data.keys(): # for each phrase
		phrase_dir = ASL_FRAME_PATH+"\\"+phrase_name
		mkdir(phrase_dir)
		print("Writing: " + phrase_name)
		print("Writing time: %s seconds" % (time.time()-start_time))
		for phrase_file_num in range(len(data[phrase_name])): # for each file of the phrase
			phrase_file_dir = phrase_dir+"\\"+str(phrase_file_num)
			mkdir(phrase_file_dir)
			for frame_num in range(len(data[phrase_name][phrase_file_num])): # for each frame
				frame = data[phrase_name][phrase_file_num][frame_num]
				frame_file_dir = phrase_file_dir+"\\"+str(frame_num)+".png"
				cv2.imwrite(frame_file_dir.replace(" ", ""), frame)

def collect_pickle_data():
	"""
	get the hand3d data for all phrase file1_frames
	and save it to the project pickle dir
	"""
	phrase_names = []
	phrase_labels = []
	phrases = []
	phrase_counter = -1
	for phrase_name in os.listdir(ASL_FRAME_PATH): # for each phrase
		phrase_dir = ""
		phrase_dir = ASL_FRAME_PATH+"\\"+phrase_name
		pickle_dir = phrase_dir+"\\pickle"
		phrase_names.append(phrase_name)
		phrase_labels.append(phrase_counter)
		phrases.append([])
		phrase_counter += 1
		for phrase_file in os.listdir(phrase_dir): # for each mp4 file of the phrase
			phrase_file_dir = phrase_dir+"\\"+phrase_file
			hand_data_file_name = pickle_dir+"\\"+phrase_file+".pickle"
			# load array of hands for this phrase file
			hand_data = pickle.load(open(hand_data_file_name, "rb"))
			phrases[phrase_counter].append(hand_data)

    # write data to binary files
	with open(r"C:\Users\louis\OneDrive\Documents\machine-learning\pickle\phrases.pickle","wb") as f:
		pickle.dump(phrases,f)
	with open(r"C:\Users\louis\OneDrive\Documents\machine-learning\pickle\phrase_labels.pickle","wb") as f:
		pickle.dump(phrase_labels,f)
	with open(r"C:\Users\louis\OneDrive\Documents\machine-learning\pickle\phrase_names.pickle","wb") as f:
		pickle.dump(phrase_names,f)

#write_frames(read_frames())
#frames_to_hands()
collect_pickle_data()
