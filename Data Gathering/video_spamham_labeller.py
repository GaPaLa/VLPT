
# create GUI window to show an image

import matplotlib
import matplotlib.pyplot as plt
import matplotlib.image as mpimg
import os
import numpy as np
from os import listdir
from os.path import isfile, join
from PIL import Image

#os.chdir('/media/idmi/Music/Dissertation/VLPT/Data Gathering/')

#matplotlib.use("Qt5agg")

print('Each image to classified into:\n1: Minecraft vanilla survival/hardcore with NO mods and NO artifacts\n2: Minecraft vanilla survival with artifacts\n3:Minecraft Creative/Modded/Server/Multiplayer')



# load spamham images
print('loading filenames...')
all_filenames = listdir('spamham_samples/frames')
print('loaded!')



# load labels file
labels_file = open('spamham_samples/frames_labels.txt', 'r')
already_labelled = labels_file.read()
already_labelled = already_labelled.split('\n')
for i in range(len(already_labelled)):
    link = already_labelled[i].split(',')[0]
    already_labelled.append(link)
labels_file.close()







# crete list of frames to check
frames_to_label = []
for filename in all_filenames:
    filename = '.'.join(filename.split('.')[0:-1])
    if not (filename in already_labelled):
        frames_to_label.append(filename)
print(len(frames_to_label))

fig = plt.figure()
ax = plt.gca()
image = np.random.normal(0,1,[720,1280,3])
window = ax.imshow(image)
for filename in frames_to_label:

    # get next unlabelled frame
    try:
        image = mpimg.imread('spamham_samples/frames/'+filename+'.jpeg') # load next image
    except:
        continue
    # display frame at for labelling
    window.set_data(image)

    # get label for current image
    valid=False
    while not valid:
        print(filename)
        label = input('')
        try:
            if int(label) in [1,2,3]:
                valid=True
            else:
                print('not [1,2,3] try again')
        except:
                print('NaN, try again')

    # save current image index and label to file
    line = filename + ',' + str(label) + '\n'
    with open('spamham_samples/frames_labels.txt', 'a') as labels_file:
        labels_file.write(line)