
# create GUI window to show an image

import matplotlib
import matplotlib.pyplot as plt
import matplotlib.image as mpimg
from os import listdir
from os.path import isfile, join

matplotlib.use("Qt5agg")

print('Each image to classified into:\n1: Minecraft vanilla survival/hardcore with NO mods and NO artifacts\n2: Minecraft vanilla survival with artifacts\n3:Minecraft Creative/Modded/Server/Multiplayer')

def get_response(img):
    plt.imshow(img)
    
    valid=False
    while not valid:
        label = input('')
        try:
            if int(label) in [1,2,3]:
                print('ok!')
                valid=True
            else:
                print('not [1,2,3] try again')
        except:
                print('NaN, try again')

    return label








# load spamham images
all_filenames = listdir('spamham_samples/frames')

# load labels file
if isfile('spamham_samples/frames_labels.txt'):
   labels_file = open('spamham_samples/frames_labels.txt', 'a')
else:
    labels_file = open('spamham_samples/frames_labels.txt', 'w')
labels_file.close()








#load 

for image_name in all_filenames:

    image = mpimg.imread('spamham_samples/frames/'+image_name)

    label = get_response(image)

    link = ''.join(image_name.split('.')[0:-1]) # remove extension
    line = link + ',' + str(label) + '\n'
    with open('spamham_samples/frames_labels.txt', 'a') as labels_file:
        labels_file.write(line)








