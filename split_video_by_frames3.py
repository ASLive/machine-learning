import cv2
import numpy as np
import os
import re
import time

FPS = 29.97 # Frame rate for Youtube ASL Music Videos
end_dir = r"C:\Users\louis\OneDrive\Documents\ASLive\video_output"
start_dir = r"C:\Users\louis\OneDrive\Documents\ASLive\videos"
start_time = time.time()

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

def mkdir(directory):
	if not os.path.exists(directory):
		os.makedirs(directory.replace(" ",""))

def read_frames():
	"""read frames of files of phrases"""
	data = {} # {phrase1_name: [[file1_frames], [file2_frames],..],...}
	for phrase_name in os.listdir(start_dir): # for each phrase
		data[phrase_name] = []
		phrase_dir = ""
		phrase_dir = start_dir+"\\"+phrase_name
		print("Reading: "+phrase_name)
		for phrase_file in os.listdir(phrase_dir): # for each file of the phrase
			if phrase_file.endswith(".mp4"):
				frames = frame_capture(phrase_dir + "\\"+ phrase_file)
				data[phrase_name].append(frames) # get the phrase of the file

	return data
def write_frames(data):
	"""write frames to proper file structure"""
	mkdir(end_dir)
	for phrase_name in data.keys(): # for each phrase
		phrase_dir = end_dir+"\\"+phrase_name
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


write_frames(read_frames())
