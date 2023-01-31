
# create GUI window to show an image

import matplotlib
import matplotlib.pyplot as plt
import matplotlib.image as mpimg
import os
from os import listdir
from os.path import isfile, join
from PIL import Image

os.chdir('/media/idmi/Music/Dissertation/VLPT/Data Gathering/')

matplotlib.use("Qt5agg")

print('Each image to classified into:\n1: Minecraft vanilla survival/hardcore with NO mods and NO artifacts\n2: Minecraft vanilla survival with artifacts\n3:Minecraft Creative/Modded/Server/Multiplayer')



# load spamham images
print('loading filenames...')
all_filenames = listdir('spamham_samples/frames')
print('loaded!')

# load labels file
if isfile('spamham_samples/frames_labels.txt'):
    labels_file = open('spamham_samples/frames_labels.txt', 'r')
    pre_labelled = labels_file.read()
    pre_labelled = pre_labelled.split('\n')
    for i in range(len(pre_labelled)):
        link = pre_labelled[i].split(',')[0]
        pre_labelled.append(link)
    labels_file.close()
else:
    labels_file = open('spamham_samples/frames_labels.txt', 'rw')
    pre_labelled=''
    labels_file.close()






#load 

current_label_index=0
frame_index = 0
while all_filenames[frame_index].split('.')[0] in pre_labelled:
    frame_index+=1
current_label_index=frame_index
image = mpimg.imread('spamham_samples/frames/'+all_filenames[current_label_index])

fig = plt.figure()
ax = plt.gca()
window = ax.imshow(image)

while frame_index < len(all_filenames):

    # display frame at current_abe;_index for labelling
    window.set_data(image)

    # immediately get next unlabelled frame
    frame_index+=1
    while all_filenames[frame_index].split('.')[0] in pre_labelled:
        frame_index+=1
    image = mpimg.imread('spamham_samples/frames/'+all_filenames[frame_index]) # load next image

    # get label for current image
    valid=False
    while not valid:
        print(all_filenames[current_label_index].split('.')[0])
        label = input('')
        try:
            if int(label) in [1,2,3]:
                print('ok!')
                valid=True
            else:
                print('not [1,2,3] try again')
        except:
                print('NaN, try again')
    # save current image index and label to file
    label=label
    current_label_index=current_label_index
    link = all_filenames[current_label_index].split('.')[0] # remove extension
    line = link + ',' + str(label) + '\n'
    with open('spamham_samples/frames_labels.txt', 'a') as labels_file:
        labels_file.write(line)
    
    # update labelling target to next unlabelled frame
    current_label_index = frame_index







