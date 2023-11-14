from torch.multiprocessing import Process, Queue, Event, set_start_method
import os
import numpy as np
import shutil
import cv2
import math











AGENT_RESOLUTION=[128,128]
NUM_WORKERS=7

dataset_dir = 'DATASET/train/'
output_video_dir = 'DATASET/train/numpy256/'










frame_cache=[]
current_video_id=''
def save_frame(video_id, frame, frame_index, state):
    (frame_cache, current_video_id) = state
    if current_video_id != video_id:
        current_video_id = video_id
        frame_cache = []
    frame_cache.append(frame)
    if len(frame_cache) == 256:
        np.save(f'{output_video_dir}{video_id},{frame_index}.npy', np.array(frame_cache))
        frame_cache = []
    return (frame_cache, current_video_id)



# https://stackoverflow.com/questions/13538748/crop-black-edges-with-opencv
def crop_resize(image):
    # crop black borders

    y_nonzero, x_nonzero, _ = np.nonzero(image>16)
    try:
        image2 = image[np.min(y_nonzero):np.max(y_nonzero), np.min(x_nonzero):np.max(x_nonzero)]
        image2 = cv2.resize(image2, AGENT_RESOLUTION, interpolation=cv2.INTER_LINEAR)
        return image2
    except:
        return cv2.resize(image, AGENT_RESOLUTION, interpolation=cv2.INTER_LINEAR)

    return image



def get_next_frame(video, frame_index_soft):
    
    ret=True
    while video.get(cv2.CAP_PROP_POS_FRAMES)<round(frame_index_soft) and ret:
        ret, frame = video.read()
    #video.set(cv2.CAP_PROP_POS_FRAMES, round(frame_index_soft))
    
    ret, frame = video.read()

    frame_index_soft += video.get(cv2.CAP_PROP_FPS)/20
    
    return ret, frame, frame_index_soft



def video_numpyer(id):
    while True:
      
        #get next video from queue
        video_path = global_target_videos_queue.get()
        video_id = video_path.split('/')[-1].split('.')[0]

        # check video has not already been processed
        numpied = ' '.join(os.listdir(output_video_dir))
        if video_id in numpied:
            print(video_id, 'already processed, skipping....')
            continue




        # open CV video capture
        video = cv2.VideoCapture(video_path)
        num_frames = int(video.get(cv2.CAP_PROP_FRAME_COUNT))
        print(video_path, video_id, num_frames)
        fps = video.get(cv2.CAP_PROP_FPS)

        absolute_frame_index,softframe = 0,0
        ret=True

        ## WARNING: THIS LEAVES THE LAST 0-255 FRAMES UNSAVED
        state=([],'')
        while True:

            ret,frame,softframe = get_next_frame(video, softframe)
            if ret == False:
                print(video_path, 'DIED @', absolute_frame_index, '  of', num_frames*(20/fps))
                if absolute_frame_index< (num_frames*(20/fps) - 256):
                    print('------------!!!!!!!!!!!!!!!!!!!!!! EARLY EXIT^', video_path)
                break

            cv2.cvtColor(frame, code=cv2.COLOR_BGR2RGB, dst=frame)
            frame = np.asarray(np.clip(frame, 0, 255), dtype=np.uint8)
            frame = crop_resize(frame)

            state = save_frame(video_id, frame, absolute_frame_index, state)
            absolute_frame_index+=1

            # if that newest frame filled the current chunk, save chunk and make new chunk


















############### GET LIST OF ALL TRAINING EPISODES
print('DataLoader opening dir',dataset_dir, os.getcwd())
video_IDs = os.listdir(dataset_dir+'videos/')
for i in range(len(video_IDs)):
    video_IDs[i] = video_IDs[i].split('.')[0]
try:
  numpyvideo_IDs = os.listdir(dataset_dir+'numpy256/')
except:
  os.mkdir(dataset_dir+'numpy256/')
  numpyvideo_IDs = os.listdir(dataset_dir+'numpy256/')
for i in range(len(numpyvideo_IDs)):
    numpyvideo_IDs[i] = numpyvideo_IDs[i].split('.')[0].split(',')[0]



############### GET LIST OF ALL TRAINING EPISODES
action_IDs = os.listdir(dataset_dir+'actions/')
for i in range(len(action_IDs)):
    action_IDs[i] = action_IDs[i].split('.')[0]
video_IDs = os.listdir(dataset_dir+'videos/')
for i in range(len(video_IDs)):
    video_IDs[i] = video_IDs[i].split('.')[0]
# ----- CHECK WPM OF TRANSCRIPT FEBORE ACCEPTING
word_files = os.listdir(dataset_dir+'transcripts/')
word_IDs=[]
ham=0
for i in range(len(word_files)):
    transcript = dataset_dir+'transcripts/'+word_files[i] #num processes
    with open(transcript,'r') as file:
        raw=file.read()
        lines = raw.split('\n')
        while '' in lines:
            lines.remove('')
        lastline=lines[-1]
        numwords = len(lines)
        maxtime = int(lastline.split(",")[-1])
        duration_secs = float(maxtime)/(1000.)
        num_foreign_tokens = raw.count('\n24,')
        if num_foreign_tokens/numwords > 0.04:   # if more than 4% of tokens are unknown to the tokenizer, assume it is not alanguage known to the tokenizer
            #print('NON-ENGLISH LANGUAGE! SKIPPING!', num_foreign_tokens/numwords)
            continue
        if numwords/duration_secs < 3.6:       # at 4 tokens/second, and 20fps, we average 0.199 word_tokens/frame -> 20% of frames have words. This means we average 80% silence. Using D=4, we get 1 - (4.0/1000)*1*4*50 ) = ~20% silence. Any fewer than 4 tok/sec results in >20% silence tokens in context.
            continue
        else:
            ham+=1
            word_IDs.append(word_files[i].split('.')[0])
print("Filtered videos with. low WPM. remaining :", str(int(100*(float(ham)/(i+1))))+'%')


unique_ids = list( set(word_IDs).intersection(set(video_IDs)).intersection(set(action_IDs)) ) # for training LM, dont need actions.


print('completed numpy', numpyvideo_IDs)
unique_ids = list( set(word_IDs).intersection(set(video_IDs)) )




unique_ids = list(  set(unique_ids) - set(numpyvideo_IDs) )   # do not include pre-numpied frames_ # this skips processing the remainder of videos which have only had 1 chunk processed. this can be remendied by checking video length wiht OpenCV and compaing the final number at the end of the batch. If the chunk number is smalle,r it is not the last one, and we nee top get more chunks starting from that index.

import random
#unique_ids = sorted(unique_ids)
random.shuffle(unique_ids)


print(len(unique_ids), 'remaining to be processed...')
# --- create queue to pop videos to be converted
global global_target_videos_queue
global_target_videos_queue = Queue()
global done_queue
done_queue = [Queue() for _ in range(NUM_WORKERS)]
for unique_id in unique_ids:
  video_path = os.path.abspath(os.path.join(dataset_dir+'videos', unique_id + ".mp4"))
  global_target_videos_queue.put(video_path)




# -- create workers
processes=[0]*NUM_WORKERS
for i in range(NUM_WORKERS):
    processes[i] = Process(
            target=video_numpyer,
            args=[i],
            daemon=False)
    processes[i].start()

while True:
  pass