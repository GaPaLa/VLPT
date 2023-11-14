# NOTE: this is _not_ the original code of IDM!
# As such, while it is close and seems to function well,
# its performance might be bit off from what is reported
# in the paper.

import shutil
from argparse import ArgumentParser
import pickle
import cv2
import numpy as np
import json
import torch as th
import os
import random

from concurrent.futures import ThreadPoolExecutor
from queue import Queue
import threading

from agent import resize_image, AGENT_RESOLUTION, ENV_KWARGS
from inverse_dynamics_model import IDMAgent

from matplotlib import pyplot as plt

DEVICE=th.device('cuda')


KEYBOARD_BUTTON_MAPPING = {
    "key.keyboard.escape" :"ESC",
    "key.keyboard.s" :"back",
    "key.keyboard.q" :"drop",
    "key.keyboard.w" :"forward",
    "key.keyboard.1" :"hotbar.1",
    "key.keyboard.2" :"hotbar.2",
    "key.keyboard.3" :"hotbar.3",
    "key.keyboard.4" :"hotbar.4",
    "key.keyboard.5" :"hotbar.5",
    "key.keyboard.6" :"hotbar.6",
    "key.keyboard.7" :"hotbar.7",
    "key.keyboard.8" :"hotbar.8",
    "key.keyboard.9" :"hotbar.9",
    "key.keyboard.e" :"inventory",
    "key.keyboard.space" :"jump",
    "key.keyboard.a" :"left",
    "key.keyboard.d" :"right",
    "key.keyboard.left.shift" :"sneak",
    "key.keyboard.left.control" :"sprint",
    "key.keyboard.f" :"swapHands",
}

# Template action
NOOP_ACTION = {
    "ESC": 0,
    "back": 0,
    "drop": 0,
    "forward": 0,
    "hotbar.1": 0,
    "hotbar.2": 0,
    "hotbar.3": 0,
    "hotbar.4": 0,
    "hotbar.5": 0,
    "hotbar.6": 0,
    "hotbar.7": 0,
    "hotbar.8": 0,
    "hotbar.9": 0,
    "inventory": 0,
    "jump": 0,
    "left": 0,
    "right": 0,
    "sneak": 0,
    "sprint": 0,
    "swapHands": 0,
    "camera": np.array([0, 0]),
    "attack": 0,
    "use": 0,
    "pickItem": 0,
}


# Matches a number in the MineRL Java code regarding sensitivity
# This is for mapping from recorded sensitivity to the one used in the model
CAMERA_SCALER = 360.0 / 2400.0



"""From VPT paper: we apply the IDM over a
video using a sliding window with stride 64 frames and only use the pseudo-label prediction for
frames 32 to 96 (the center 64 frames). By doing this, the IDM prediction at the boundary of the
video clip is never used except for the first and last frames of a full video."""
### !thankfully IDM model is not  transformerXL - all internal hidden_states passed to itself are None, so we can just convolve it without worrying about weird reucrrence order stuff being messed up













batches_queue = Queue()








os.chdir('/content/drive/MyDrive/_DISSERTATION/')







# https://stackoverflow.com/questions/13538748/crop-black-edges-with-opencv
def crop_resize(image):
    try:
        # crop black borders
        y_nonzero, x_nonzero, _ = np.nonzero(image>16)
        image_out = image[np.min(y_nonzero):np.max(y_nonzero), np.min(x_nonzero):np.max(x_nonzero)]
        assert image_out.shape[0]>=1
        assert image_out.shape[1]>=1
        # scale image
        image_out = resize_image(image_out, AGENT_RESOLUTION)
        return image_out

    except:
        #scale cropped image to agent size
        image_out = resize_image(image, AGENT_RESOLUTION)
        #cv2.imwrite(  'TRAINING/cropresize/'+str('cropped_resized'+str(image.mean()))+'.jpg',  image)
        return image_out




def get_next_frame(video, frame_index_soft):
    
    ret=True
    while video.get(cv2.CAP_PROP_POS_FRAMES)<round(frame_index_soft) and ret:
        ret, frame = video.read()
    #video.set(cv2.CAP_PROP_POS_FRAMES, round(frame_index_soft))
    
    ret, frame = video.read()

    frame_index_soft += video.get(cv2.CAP_PROP_FPS)/20
    
    return ret, frame, frame_index_soft







# give main a file and it will output all actios for all frames
# improved efficieny using sliding window and only using middle predictions
# when passing a video to  main, batch_size should be as large as possible
def Producer():
    global batches_queue
    print('in producer')
    print(os.getcwd())
    
    transcripts = 'DATASET/train/transcripts/'
    video_folder = 'DATASET/train/videos/'
    output_path = 'DATASET/train/actions/'
    batch_size=2
    required_resolution = ENV_KWARGS["resolution"]

    with th.no_grad():
        
        # LOAD VIDEOS
        input_videos = os.listdir(video_folder)
        precalculated_actions = os.listdir(output_path)
        random.shuffle(precalculated_actions)
    
    
        # CALCULATE ACTIONS FOR NEXT VIDEO
        target_ids=[]
        for video in input_videos:
            print(video)
            id = video.split('.')[0]
    
            # recheck which have been precalcuated in case another isntance is running
            precalculated_actions = ','.join(os.listdir(output_path))
            if id in precalculated_actions: # dont predict actions which weve already predicted and stored
                continue
            
            
                        
            

            # ------------------------------ CHECK VIDEO WPM: DONT LABEL VIDEOS WITH WPM<180 ; these have num_silence > (silence_per_)
            try:
                transcript = 'DATASET/train/transcripts/'+id
                with open(transcript,'r') as file:
                    raw = file.read()
                    lines = raw.split('\n')
                    while '' in lines:
                        lines.remove('')
                    lastline=lines[-1]
                    numwords = len(lines)
                    maxtime = lastline.split(",")[-1]
                    duration_secs = float(maxtime)/(1000.)
                    
                    num_foreign_tokens = raw.count('\n24,')
                    if num_foreign_tokens/numwords > 0.04:   # if more than 4% of tokens are unknown to the tokenizer, assume it is not alanguage known to the tokenizer
                        print('NON-ENGLISH LANGUAGE! SKIPPING!', num_foreign_tokens/numwords)
                        continue
                    
                    if numwords/duration_secs<3.8:       # at 4 tokens/second, and 20fps, we average 0.199 word_tokens/frame -> 20% of frames have words. This means we average 80% silence. Using D=4, we get 1 - (4.0/1000)*1*4*50 ) = ~20% silence. Any fewer than 4 tok/sec results in >20% silence tokens in context.
                        print("WPM IS TOO LOWWW!!! tok/sec=",numwords/duration_secs)
                        continue
            except:
                print(id, 'transcript not found!!!')
                continue
            
            

            
            
            # ------ OPEN VIDEO
            print('opening...')
            cap = cv2.VideoCapture(video_folder+video)
            fps = cap.get(cv2.CAP_PROP_FPS)
            num_frames = round(cap.get(cv2.CAP_PROP_FRAME_COUNT)*(20./fps))
            frame_tracker = 0

            # the frames  variable cnotains the batch of frames to estimate actions from.
            # it is always ([batch size]x64 + 64) frames.
            # every iteration, the next [batch size]x64
            frames = np.zeros([batch_size*64+64,128,128,3],dtype=np.float32)#@debug-dtype
            frames_batch = np.zeros([batch_size,128,128,128,3], dtype=np.float32) #@debug-dtype # fill frame batch with 42 so later we can identify which frames/actions to throw away because reached end of episode/scuffed but im out of time and smol brain AND based
            if fps == 0:
              print('bad video path')
              continue
            



            # continuously get bunch of new frames and return actions until video is over
            print('P getting:')
            current_frame = 0
            video_ended = False
            while not video_ended:
                print('.')
                
                # if at first frame, get starting 64 frames in addition to batchx64 frames (to get to 128 frames)
                # if not at first frame, increment b so that we dont overwrite the past context needed
                # every batch needs to fetch the current 64 frames being classified as well as the future context 32 frames and previous context 32 frames. However, we can have every batch rely on the previous batch's [64:128] to use as past 32 and first 32 to estimate. However, the first has none to depend on. This fills that in at teh stat of actions estimationg across this video
                previous_last_frame = current_frame
                if current_frame>0:
                    frames[0:64] = frames[-64:]
                else:
                    for t in range(64):
                        ret, frame, frame_tracker = get_next_frame(cap, frame_tracker) #(current_frame<=128 or current_frame!=num_frames), np.zeros([360,640,3], dtype=np.uint8)# #@debug         #print(frame.shape)         
                        if (not ret) or (frame is None):
                            video_ended = True
                            print("END VIDEO AT INDEX:",current_frame, num_frames)
                            break
                        # scale
                        frame = crop_resize(frame) #@debug
    
                        #frame = np.full([128,128,3],int(frame_tracker), dtype=np.int64)#@debug
                        # BGR -> RGB
                        frames[t] = frame[..., ::-1] #@debug #is always 128 frames. first 64 frames get removed and next 64 are added on every ieration
                        
                        current_frame += 1
    
                ### get next [batch size]x 64 frames (128 if first, however many possible if reach end (<=128)). each batch is size 128
                for b in range(batch_size):
                    # add the amount of needed frames.
                    for t in range(64):
                        ret, frame, frame_tracker = get_next_frame(cap, frame_tracker) # (current_frame<=128 or current_frame!=num_frames), np.zeros([360,640,3], dtype=np.uint8)# #@debug           #
                        if (not ret) or (type(frame)==None): #                     or (current_frame>(512+128)):
                            video_ended = True
                            print('not ret at fram_index=',current_frame, frame_tracker)
                            break
                        # scale
                        frame = crop_resize(frame)         #@debug
    
                        #frame = np.full([128,128,3],int(frame_tracker), dtype=np.int64) #@debug
                        # BGR -> RGB
                        frames[64+b*64+t] = frame[..., ::-1] #@debug #is always 128 frames. first 64 frames get removed and next 64 are added on every ieration
                        
                        current_frame +=1
                
                # organise next [batch size]x 64 frames. (give 128 frames to each batch)
                frame_1 = 0
                frame_2 = 128
                for b in range(batch_size):
                    frames_batch[b] = frames[frame_1:frame_2]
                    frame_1 += 64
                    frame_2 += 64

                #print(frames_batch.shape)
                

                print("!!!!!!!!!PRODUCED BATCH!!!!!!!!!")
                batches_queue.put((id, frames_batch.copy(), previous_last_frame, current_frame, video_ended, num_frames))

                a=0
                while batches_queue.qsize()>20:
                    if a==0:
                        print('P SLEEPING')
                        a=1
                    pass
        
            # --- VIDEO ENDED: CLOSE VIDEO STREAM
            print('done')
            cap.release()
        
        
        
        
        
        
        
        
        
        
        
        
        
        
        
        
        
        
        
def Consumer():
    global batches_queue
    try:
    
      print('in consumer')
      output_path = 'DATASET/train/actions/'
      model='VLPT/4x_idm.model'
      weights='VLPT/4x_idm.weights'
      batch_size=2

      # LOAD MODEL
      print('P:loading model..')
      with open(model, "rb") as modelfile:
        agent_parameters = pickle.load(modelfile)
      print('P:0')
      net_kwargs = agent_parameters["model"]["args"]["net"]["args"]
      pi_head_kwargs = agent_parameters["model"]["args"]["pi_head_opts"]
      pi_head_kwargs["temperature"] = float(pi_head_kwargs["temperature"])
      print('P:1')
      agent = IDMAgent(idm_net_kwargs=net_kwargs, pi_head_kwargs=pi_head_kwargs, device=DEVICE)
      agent.load_weights(weights)
      #agent.policy = th.compile(agent.policy, mode='reduce-overhead')
      agent.policy.eval()
      required_resolution = ENV_KWARGS["resolution"]
      print('P:loaded')

      current_video_id = 0
      while True:
      
      
          # -------- GET NEXT BATCH FROM VIDEO: MIGHT BE FROM A NEW VIDEO
          print('C: waiting...')
          id, frames_batch, previous_last_frame, current_frame, video_ended, num_frames = batches_queue.get()
          print('C: got.')
          if id != current_video_id:
              current_video_id = id
              print(" ------------------------------------- NEW VIDEO!!",current_video_id," resetting actions...")
              all_predicted_actions = {'buttons':np.zeros([num_frames,20]), 'camera':np.zeros([num_frames,2])}
          
          
          
          
          # ------------ PREDICT ACTIONS
          with th.no_grad():
              print("P: === Predicting actions ===. current:", current_frame, 'progress=',int((float(current_frame)/float(num_frames))*100))
              action_pred = agent.predict_actions(frames_batch, raw_frames=True)
              #action_pred = {'buttons': th.from_numpy(action_pred['buttons']),
              #              'camera': th.from_numpy(action_pred['camera'])} 
              #action_pred = {'buttons': th.from_numpy(action_pred['buttons']),
              #              'camera': th.from_numpy(action_pred['camera'])} 
              
              #frame_values = frames_batch.mean(axis=(2,3,4))#@debug
              #print(frame_values)
              #frame_values = frame_values.reshape([batch_size,128,1])#@debug
              #action_pred = {'buttons': np.repeat(frame_values,20,axis=2), #@debug
              #              'camera': np.repeat(frame_values,2,axis=2)} #@debug



          # ------------ UNBATCH ACTION PREDICTIONS AND SAVE TO LIST OF ACTIONS
          # --- CASE: FIRST BATCH
          if previous_last_frame==0: #(if this is the first batch, we need to save first actions taht would normally be cropped since the are not in the middle)
              all_predicted_actions['buttons'][0:32] = action_pred['buttons'][0,0:32] # if first frame, predict actions 0 through 32 despite not being centred. remove last 32 frames since this can be estimated windowed
              all_predicted_actions['camera'][0:32] = action_pred['camera'][0,0:32] # if first frame, predict actions 0 through 32 despite not being centred. remove last 32 frames since this can be estimated windowed            
              all_predicted_actions['buttons'][32:32+64*batch_size] = action_pred['buttons'][:,32:96].reshape([batch_size*64,20])
              all_predicted_actions['camera'][32:32+64*batch_size] = action_pred['camera'][:,32:96].reshape([batch_size*64,2])
              print('add:',0,32+64*batch_size,32+64*batch_size-0)
          
          # --- CASE: VIDEO ENDED
          elif video_ended: #(if this is the last batch, we need to save the last actions taht would normally be cropped since the are not in the middle)
              # remember, if we reached this state, current_frame has gone over the actual current frame - it is now past the number of frames
              end=current_frame
              # move actions from last full SEQs to saved actions 
              if current_frame-1 > previous_last_frame:
                  # END OF VIDEO: GET ALL FULL ACTION BATCHES FROM FULL FRAMES
                  # find which batch (index) the last frame is in
                  for b in range(batch_size):
                      if (64*(b)<((current_frame-1)-previous_last_frame)) and (((current_frame-1)-previous_last_frame)<=64*(b+1)):
                          end_batch_index = b
                          break
                  num_full_batches = end_batch_index # if the last full frame appears on batch 3, it must be a dirty batch, so the previous batch is the last full batch.
                  if ((current_frame-1)-previous_last_frame)%64==0:  # UNLESS the last full frame appears at the last element of the batch, in which case the last full batch is the last batch
                    end_batch_index += 1
                    num_full_batches += 1
                  start = previous_last_frame-32 # standard start
                  end = (start + num_full_batches*64) # end on the batch which is still full and has no full 42 frames
                  print(current_frame, previous_last_frame)
                  print(num_full_batches, end_batch_index)
                  print('a',start, end)
                  all_predicted_actions['buttons'][start:end] = action_pred['buttons'][0:end_batch_index,32:96].reshape([num_full_batches*64,20])   # at end frame, get all middle values froma l lbatches except last, whch will have been cut short
                  all_predicted_actions['camera'][start:end] = action_pred['camera'][0:end_batch_index,32:96].reshape([num_full_batches*64,2])
                  print('add:',start,end,end-start)
                  print('b')
                  # if the last SEQ with full frames in it has empty frames, add just the full frames to the 
                  # find first empty frame
                  
                  # GET LAST POTENTIALLY DIRTY BATCH # remember to not discard last 32 frames
                  start=end
                  full_frames_this_batch = (current_frame) - end
                  end_f = 32 + full_frames_this_batch  # 32 + (((current_frame-1)-previous_last_frame )%64) #start at 64 because up to 64 is fr
                  end = start+full_frames_this_batch
                  print(start, end, end_f)
                  
                  # if first frame, predict actions 0 through 32 despite not being centred. remove last 32 frames since this can be estimated windowed                    
                  worked=False
                  try:
                      all_predicted_actions['buttons'][start:end] = action_pred['buttons'][end_batch_index,32:end_f-32]
                      all_predicted_actions['camera'][start:end] = action_pred['camera'][end_batch_index,32:end_f-32]
                      print('c')
                  except:
                      for y in range (-2, 2):
                        try:
                          print('d', y)
                          num = end-start
                          all_predicted_actions['buttons'][start:end] = action_pred['buttons'][end_batch_index,32:32+num+y]
                          all_predicted_actions['camera'][start:end] = action_pred['camera'][end_batch_index,32:32+num+y]
                          worked=True
                          break
                        except:
                            pass
                  assert worked, 'worked'
                  print('add:',start,end,end-start)
              #crop actions to fit number of predicted actions, so we can later check that the num f estimated actions = num video frames                    
              all_predicted_actions['buttons'] = all_predicted_actions['buttons'][:end] #crop to actual used size
              all_predicted_actions['camera'] = all_predicted_actions['camera'][:end] #crop to actual used size
          
          # --- CASE: NORMAL BATCH
          else:
              start = previous_last_frame-32
              end = current_frame-32
              all_predicted_actions['buttons'][start:end] = action_pred['buttons'][:,32:96].reshape([batch_size*64,20])
              all_predicted_actions['camera'][start:end] = action_pred['camera'][:,32:96].reshape([batch_size*64,2])
              print('add:',start,end,end-start)

          
          
          
          
          # ------- IF VIDEO ENDED, SAVE PREDICTED ACTIONS
          if video_ended:
              print(" ------------------------------------- VIDEO END!! saving actions...", current_video_id)
              
              if False:
                buttons_graph = all_predicted_actions['buttons'].mean(axis=1) #@debug
                camera_graph = all_predicted_actions['camera'].mean(axis=1) #@debug
                plt.plot(buttons_graph) #@debug
                plt.plot(camera_graph) #@debug
                plt.savefig(current_video_id+'.png') #@debug
                plt.clf() #@debug
                plt.close()
                print('F:',all_predicted_actions['buttons'].shape, num_frames, end, start, previous_last_frame, current_frame)
              assert all_predicted_actions['buttons'].shape[0] == num_frames, "FRAMES != ACTIONS "+str(num_frames)+' '+str(all_predicted_actions['buttons'].shape[0])
              #assert th.all(buttons_graph == th.arange(num_frames)), "WRONG ORDER"
          
              if True: #@debug
                print("SAVE ACTIONS TO JSON")
                print(output_path+id+'.IDMaction')
                with open(output_path+id+'.IDMaction','wb') as action_dumpfile:
                  pickle.dump(all_predicted_actions, action_dumpfile)
                #with open('/'+id+'.IDMaction','wb') as action_dumpfile:
                #  pickle.dump(all_predicted_actions, action_dumpfile)
    except Exception as e:
      print("ERRRRRORRRRRR C",e)
      
      
          
        




from threading import Thread

P = Thread(name = "Producer", target = Producer)
C = Thread(name = "Consumer2", target = Consumer)
P.start()
C.start()
