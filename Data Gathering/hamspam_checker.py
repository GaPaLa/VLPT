
# create GUI window to show an image

import matplotlib
import matplotlib.pyplot as plt
import matplotlib.image as mpimg
import os
from os import listdir
from os.path import isfile, join
from PIL import Image
import numpy as np

os.chdir('/media/idmi/Music/Dissertation/VLPT/Data Gathering/')

matplotlib.use("Qt5agg")

print('Each image to classified into:\n1: Minecraft vanilla survival/hardcore with NO mods and NO artifacts\n2: Minecraft vanilla survival with artifacts\n3:Minecraft Creative/Modded/Server/Multiplayer')



# load labels file
#labels_file = open('spamham_samples/frames_labels.txt', 'r')
labels_file = open('spamham_samples/manual_frames_labels.txt', 'r')
pre_labelled = labels_file.readlines()

frame_index=-1
current_label_index=0

#load 
image = np.random.normal(0,1,[720,1280,3])
fig = plt.figure()
ax = plt.gca()
window = ax.imshow(image)

# start checking after a certain ID in case you already saw half teh samples, closed the progra,m watnt o check the rest and dont want to start again
while pre_labelled[frame_index].split(',')[0] != 'SFTWnCl9WmM.1035':
    frame_index+=1


while frame_index < len(pre_labelled):


    # get next ham frame
    frame_index+=1
    prev=frame_index
    while pre_labelled[frame_index].split(',')[-1] != '1\n':
        frame_index+=1

    # get&display frame
    image_filename = ''.join(pre_labelled[frame_index].split(',')[0:-1])
    image = mpimg.imread('spamham_samples/frames/'+image_filename+'.jpeg') # load next image
    window.set_data(image)

    # get reponse to current image
    print(image_filename)
    label = input('')
    if label == '1':
        frame_index -= 3 # 30 if back gets stuck

    # save current image index and label to file
    #label=label
    #current_label_index=current_label_index
    #link = pre_labelled[current_label_index].split('.')[0] # remove extension
    #line = link + ',' + str(label) + '\n'
    #with open('spamham_samples/frames_labels.txt', 'a') as labels_file:
    #    labels_file.write(line)






