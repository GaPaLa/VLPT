# Code for loading OpenAI MineRL VPT datasets
# NOTE: This is NOT original code used for the VPT experiments!
#       (But contains all [or at least most] steps done in the original data loading)
import numpy as np
import os.path
import time
import glob
import os
import random
#from multiprocessing import Process, Queue, Event
from torch.multiprocessing import Process, Queue, Event, set_start_method
import torch as th
import pickle
import shutil
#try:
#     set_start_method('spawn', force=True)
#except RuntimeError:
#    pass

import numpy as np
import cv2

from agent import resize_image, AGENT_RESOLUTION

random.seed(1)
np.random.seed(1)




QUEUE_TIMEOUT = 10000







# If GUI is open, mouse dx/dy need also be adjusted with these scalers.s
# If data version is not present, assume it is 1.
MINEREC_VERSION_SPECIFIC_SCALERS = {
    "5.7": 0.5,
    "5.8": 0.5,
    "6.7": 2.0,
    "6.8": 2.0,
    "6.9": 2.0,
}

# https://stackoverflow.com/questions/13538748/crop-black-edges-with-opencv


    

    

def data_loader_worker(tasks_queue, output_queue, quit_workers_event, F_SEQ_LEN, max_subseqs_per_traj=float('inf'), start_time_sec='rand', LM_TIMEOUT_RATE=4, LM_SILENCE_TOKEN=2, words_only=False, worker_id=None, num_workers=None):
    """
    Worker for the data loader.
    """
    if words_only:
        print("WARNING!! words only is set to true! all frames will be set to empty")
    finished_video = 'None'
    
    
    # ------ NUmpy video chunk loader
    
    def load_frame(video_id, frame_index, data):
        (frame_cache, current_video_id, chunk_index, np_folder) = data
        
        if current_video_id != video_id or not (chunk_index <= frame_index < chunk_index + 256):
            current_video_id = video_id
            chunk_index = frame_index // 256 * 256
            filename = np_folder+'/' + f'{video_id},{chunk_index + 255}.npy'
            if not os.path.isfile(filename):
                print('could not find', filename)
                return False, None, frame_index+1, (frame_cache, current_video_id, chunk_index)
            frame_cache = np.load(filename)
        return True, frame_cache[frame_index - chunk_index], frame_index+1, (frame_cache, current_video_id, chunk_index, np_folder)
    

    # ------- fetch batches until quit training
    first_video=True
    while True:
        task = tasks_queue.get()
        if task is None:
            break
        trajectory_id, video_id, np_folder, transcription_path, actions_file_path = task
        VPT_precomputed_path = None

        print('video:',video_id)

        # LOAD ALL WORDS
        all_words = []
        all_word_ms = []
        with open(transcription_path) as words_data:
            for line in words_data.readlines():
                line = line.split(',')
                all_words.append( int(line[0]) )
                all_word_ms.append( int(line[1]) )
        all_word_ms = np.asarray(all_word_ms, dtype=np.int64)
        all_words = np.asarray(all_words, dtype=np.int64)
        obs_words = [] # by end, mst have same length as submitted frames. to keep word timings and buffer between SUBSEQs, we use this before the subsequencing splitting starts
        tokens_queue=[]


        # initalise numpy video loading
        state = ([], '', 0, np_folder)
        video_path = '/'.join(np_folder.split('/')[:-1])+'/videos/'
        video = cv2.VideoCapture(video_path+video_id+'.mp4')
        video_max_frames = video.get(cv2.CAP_PROP_FRAME_COUNT)
        video_fps = video.get(cv2.CAP_PROP_FRAME_COUNT)
        #get numpy max frames
        maxnp=0
        for filename in os.listdir(np_folder):
            if video_id in filename:
                maxnp = max(maxnp,   int(filename.split('.')[0].split(',')[-1]))
        ms_per_frame = 1000/20 # numpy vids are all preprocessed to be 20Hz. 
        #print('ms_per_frame',maxnp, video_id,ms_per_frame)
        video.release()



        # LOAD ACTIONS FILE 
        # NOTE: this is modified. The original data_loader files laods JSONs produced by BC_recording_software. This loads pickle files produced by run_inverse_dynamics.py         
        with open(actions_file_path, 'rb') as file:
            IDM_actions = pickle.load(file)
            num_actions = IDM_actions['buttons'].shape[0]


        # LOAD VPT actions file 
        num_VPT_actions=num_actions
        if VPT_precomputed_path!=None:
            actions_VPT = th.load(VPT_precomputed_path)
            num_VPT_actions=actions_VPT['buttons'].shape[1]
            



        if start_time_sec=='rand' and first_video==True:
            start_time_sec = random.randint(120,   int((ms_per_frame/1000.)*maxnp)  )
            first_video=False
        else:
            start_time_sec=120

        # repeatedly get subseqs from video until video out of frames
        frame_index = int(  start_time_sec*20.   )
        video_tracker = frame_index
        num_subseq = max_subseqs_per_traj
        frames_seq = []
        frames_seq_ms=[]
        actions_buttons = []
        actions_camera = []
        VPT_actions_buttons = []
        VPT_actions_camera = []
        subseq_idx=0
        ret=True
        while ret and subseq_idx<max_subseqs_per_traj: # we are reading one F_SEQ_LENhaead for always know 
            assert len(frames_seq)==len(actions_buttons) and  len(frames_seq)==len(frames_seq_ms) and len(frames_seq)==len(actions_buttons) and len(frames_seq)==len(actions_camera) 
        
            #print(video_id, "GETTING NEW SUBSEQ", subseq,'BETWEEN' , frame_index, frame_index+F_SEQ_LEN, ' |FINAL:', num_frames)
            if quit_workers_event.is_set():
                print('breaking: quit_event')
                break

            # GET SUBSEQ OF ACTIONS
            if len(frames_seq)>0: # if not the first SUBSEQ, use last [W,F,A] from SUBSEQ as first [W,F,A] here (the last one was only used for future word token and was not used as input, so we need ot use it here)
                frames_seq = frames_seq[-LM_TIMEOUT_RATE:]
                frames_seq_ms= frames_seq_ms[-LM_TIMEOUT_RATE:]
                actions_buttons = actions_buttons[-LM_TIMEOUT_RATE:]
                actions_camera = actions_camera[-LM_TIMEOUT_RATE:]
            else: # if this is first SUBSEQ, we get next F_SEQ_LEN actions,words,frames +1 to account for future word needed.
                frames_seq = []
                frames_seq_ms=[]
                actions_buttons = []
                actions_camera = []
                
            limit=0
            nulls=0
            while len(frames_seq) < F_SEQ_LEN+LM_TIMEOUT_RATE:  # FOR EVERY ACTION, GET [previous frame sequence: 512], [previous word sequence: 128], [previous action sequence: 128]
                #print(video_id, "frames:", len(frames_seq), limit)r
                if frame_index>=num_actions:
                    print(frame_index, num_actions, len(frames_seq), subseq_idx, video_id)
                    break
                #if limit>2*F_SEQ_LEN:
                #    print("!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!runaway collecting", frame_index, subseq, num_frames, len(frames_seq))
                #    break


                #  --- GET NEXT ACTION LABEL
                action = { 'buttons':IDM_actions['buttons'][frame_index], 'camera':IDM_actions['camera'][frame_index] }
                is_null_action = (action['buttons']==0.).all() and (action['camera']==5.).all()
                actions_buttons.append(np.expand_dims(action['buttons'],0))
                actions_camera.append(np.expand_dims(action['camera'],0))
                #actions_buttons.append(np.full(np.expand_dims(action['buttons'],0).shape,float(frame_index))) #@DEBUG
                #actions_camera.append(np.full(np.expand_dims(action['camera'],0).shape,float(frame_index))) #@DEBUG

                #frame_index += 1 #fps/20.#if variabel framerate but this is dificult for IDM! so i dont have any non-20-fps videos for here!
                limit+=1
                
                
                # --- GET NEXT VPT ACTION
                if VPT_precomputed_path!=None:
                    #print('ALEUFHOIASUEH&F(OPAEHFIOABF)&(',actions_VPT['buttons'].shape)
                    VPT_action = { 'buttons':actions_VPT['buttons'][0,frame_index], 'camera':actions_VPT['camera'][0,frame_index] }
                    VPT_actions_buttons.append(th.unsqueeze(VPT_action['buttons'],0))
                    VPT_actions_camera.append(th.unsqueeze(VPT_action['camera'],0))
                    vpt_action_index+=1 # remember, this index is precomputed. it has pre-computed the dataloader. it is always in order given that the data loader has not changed, so its made the same frame drops and everything. What is next in the precomputed tensor is next here too.

                
                # --- GET NEXT WORD
                #check for langauge tokens that occur during the 50ms span of this frame
                #print('checking tween:',     ((frame_index+LM_TIMEOUT_RATE)*ms_per_frame - 50)     ,     (frame_index+LM_TIMEOUT_RATE)*ms_per_frame      )
                word_ms_start = (frame_index+LM_TIMEOUT_RATE) * ms_per_frame - ms_per_frame
                word_ms_end = (frame_index+LM_TIMEOUT_RATE) * ms_per_frame
                word_index = np.where(  (all_word_ms>=word_ms_start )
                                       &(all_word_ms<word_ms_end  ))
                #print('fidx=',frame_index, 'fms:',frame_index*ms_per_frame, 's:e',word_ms_start,word_ms_end , word_index)
                # append to buffer of tokens which are to be assigned to a frame
                for i in range(word_index[0].shape[0]):
                    token_id = all_words[word_index[0][i]]
                    #print(video_id,'seen token', token_id,'@ms:',all_word_ms[word_index[0][i]])
                    tokens_queue.append(token_id)
                    #tokens_queue.append(float(frame_index))########@@@DEBUG FULL
                    
                # if language tokens available in buffer AND we are at a LM active time step, take token from buffer and use as token for current timestep (associate with current frame).
                if len(tokens_queue)>0 and (len(frames_seq)%LM_TIMEOUT_RATE==0): # if there were skipped over tokens during LM timeout (D), they shuold be added to the queue and can be popped one at a time now during the silence
                    obs_words.append(tokens_queue.pop(0))
                else:
                    obs_words.append(LM_SILENCE_TOKEN)
                    

                # --- GET NEXT FRAME
                if not words_only:
                    ret, frame, video_tracker, state = load_frame(video_id, video_tracker, state)
                if ret:
                    frame_ms = (ms_per_frame*frame_index) #frame occurence timestamp is in ms. assumes frames coming in at 20Hz. Therefore each frame is worth 50ms

                    # Skip null actions as done in the VPT paper
                    # NOTE: in VPT paper, this was checked _after_ transforming into agent's action-space.
                    #       We do this here as well to reduce amount of data sent over.                    
                    # Check if there is a word at this frame or not.
                    if is_null_action and obs_words[-1]==LM_SILENCE_TOKEN:   #@@@@@@ REMOVING ACTIONS MESSES WITH WORD TIMING IF A WORD OCCURS DURING THAT NULL ACTION - SO WE DONT REMOVE ALL NULL ACTIONS, instead, we mask the target action so VPT is not trained to output null. ONLY REMOVE A NULL ACTION if no wor occurs during it. @@@@ WARNING. THSI GREATLY SPEEDS UP WPM, ADJUST D ACCORDINGLY AFTER MEASURING IT.
                        #print('null removed')
                
                        nulls+=1
                        limit+=1
                        obs_words.pop()
                        actions_camera.pop()
                        actions_buttons.pop()
                        #dont have to pop latest frame since we havent added it in the first place

                        #print(" !!!!!!!!!!!!!!!!! found nul #",nulls, " in subseq, at", frame_index)
                        frame_index +=1
                        continue # dont ad frame,word,action to this SUBSEQ if action and word are null, just move on to next

                    #cv2.cvtColor(frame, code=cv2.COLOR_BGR2RGB, dst=frame)
                    #frame = np.asarray(np.clip(frame, 0, 255), dtype=np.uint8)
                    #frame = crop_resize(frame)
                    frame = np.expand_dims(frame,0)
                    #frame=np.full(frame.shape,float(frame_index))########@@@DEBUG FULL
                    frames_seq.append(frame)
                    frames_seq_ms.append(frame_ms)
                else:
                    print(f"dead frame, exit subseq", frame_index, num_actions)
                    break
                
                
                frame_index += 1
            
            if len(frames_seq)<F_SEQ_LEN:
                print('early exit subeq', frame_index, maxnp, num_actions)
                break
                    
                
                
            # each seq is constituent of F_SEQ_LEN + the next LM_TIMEOUT_RATE worht of tokens (i.e. the next word token). always only feed the NN the past and give next word as label.
            # save sequence sample. we pass most recent [F_SEQ_LEN] frames and actions, and pass all words as {[tokens],[ms]}. These are reformatted later
            frames_seq_np = np.expand_dims(np.concatenate(frames_seq[-(F_SEQ_LEN+LM_TIMEOUT_RATE):-LM_TIMEOUT_RATE], axis=0),0) # format to numpy, add batch dim
            
            frames_seq_ms_np = np.expand_dims(np.asarray(frames_seq_ms[-(F_SEQ_LEN+LM_TIMEOUT_RATE):-LM_TIMEOUT_RATE]),0) # format to numpy, add batch di

            actions_buttons_np = np.expand_dims(np.concatenate(actions_buttons[-(F_SEQ_LEN+LM_TIMEOUT_RATE):-LM_TIMEOUT_RATE],axis=0), 0)
            actions_camera_np = np.expand_dims(np.concatenate(actions_camera[-(F_SEQ_LEN+LM_TIMEOUT_RATE):-LM_TIMEOUT_RATE],axis=0), 0)

            if VPT_precomputed_path!=None:
                VPT_actions_buttons_np = th.unsqueeze(th.cat(VPT_actions_buttons[-(F_SEQ_LEN+LM_TIMEOUT_RATE):-LM_TIMEOUT_RATE],dim=0), 0)
                VPT_actions_camera_np = th.unsqueeze(th.cat(VPT_actions_camera[-(F_SEQ_LEN+LM_TIMEOUT_RATE):-LM_TIMEOUT_RATE],dim=0), 0)
            else:
                VPT_actions_camera_np=None
                VPT_actions_buttons_np=None

            words_seq_np = np.asarray([obs_words[-(F_SEQ_LEN+LM_TIMEOUT_RATE):-LM_TIMEOUT_RATE]], dtype=np.int64) 
            # use most recent F_SEQ_LEN word tokens as word input, since these are in line with most recent frames being given. last token is always the next (future) token
            
            words_labels_np = np.asarray([obs_words[-F_SEQ_LEN:]], dtype=np.int64) 

            #print(video_id, subseq, obs_words)
            #print(video_id, "seq", subseq, words_seq_np)
            #print(video_id, subseq, words_labels_np)
            try:
              assert actions_buttons_np[0].shape[0] == F_SEQ_LEN, '1 '+str(actions_buttons_np.shape)
              assert frames_seq_np[0].shape[0] == F_SEQ_LEN, '2 '+str(frames_seq_np.shape)
              assert words_seq_np[0].shape[0] == F_SEQ_LEN, '3 '+str(words_seq_np.shape)
              assert words_labels_np[0].shape[0] == F_SEQ_LEN, '4 '+str(words_labels_np.shape)
              assert(len(obs_words)%LM_TIMEOUT_RATE==0), '5 '+str(len(obs_words))+", "+str(LM_TIMEOUT_RATE)
              #1/0
              #print('QUEUEING:',video_path.split('/')[-1], frame_index, output_queue )
              output_queue.put(( video_id, subseq_idx, frames_seq_np, frames_seq_ms, words_seq_np, words_labels_np, actions_buttons_np, actions_camera_np, finished_video), timeout=QUEUE_TIMEOUT)
              finished_video = 'None' #after we have indicated which episodes have ended, remove them from the list. Otherwise, the next time the episode shows up again in the batch and we are expecting it to be trained on, it will still be marked as needing to have its mems delted and so its memes will be cleared betweene very batch :(
                    

              #
              #while tasks_queue.qsize()%num_workers!=0:# or tasks_queue.qsize()>=num_workers*3:  #dont collect another sequence  until all workers have made a batch. wait until its been taken to makea  enw one
              #    pass
              #time.sleep(1) # allow other threads to detect this cahnge before altering it
              while output_queue.qsize()>0:
                  time.sleep(0.01)

            except Exception as e:
              print("data loader batch failed. Exiting early.. :",e)
              break
            #if WPM too high and words get weirded out, remove this episode early
            if len(tokens_queue) > 50:    # at most, info is delayted by ~2 seconds. This is usually mostly taken up by the start 
                print('!!!token queue overflow!! cropping..', len(tokens_queue), frame_index, video_id)
                tokens_queue = tokens_queue[-50:]
                #break
            subseq_idx+=1

        print(" ----------------------------ENDED VIDEO",video_id)
        finished_video = video_id # after an episode ends, we need to tell the trainer it has ended so we can clear teh episodes mems to free up VRAM. however, deletions are read at the beginning of the batch and we want to make sure they are cleared After the abtch, so we need to instruct it to delete the menms for the video After the last batch for that vid, so we stoare it here and tell the next bath to delete it. Then we delete that message so it isnt re-sent multiple time per the above ^
        if quit_workers_event.is_set():
            break

    # Tell that we ended
    output_queue.put(None)




























class DataLoader:
    """
    Generator class for loading batches from a dataset

    This only returns a single step at a time per worker; no sub-sequences.
    Idea is that you keep track of the model's hidden state and feed that in,
    along with one sample at a time.

    + Simpler loader code
    + Supports lower end hardware
    - Not very efficient (could be faster)
    - Loads up individual files as trajectory files (i.e. if a trajectory is split into multiple files,
      this code will load it up as a separate item).
    """
    #@ edit to output 256 consecutive langauge tokens + 256*D consecutive frames as a single episode
    def __init__(self, dataset_dir, n_workers=8, batch_size=8, n_epochs=1, max_queue_size=16, F_SEQ_LEN=512, max_subseqs_per_traj=float('inf'), start_time=5, LM_TIMEOUT_RATE=4, LM_SILENCE_TOKEN=2, words_only=False, precomputed=False): #start time in seconds
        assert n_workers >= batch_size, "Number of workers must be equal or greater than batch size"
        self.dataset_dir = dataset_dir
        self.n_workers = n_workers
        self.n_epochs = n_epochs
        self.batch_size = batch_size
        self.max_queue_size = max_queue_size
        self.precomputed = precomputed # model training loop is designed to use precomputed vanilla VPT action outputs for KL-divergence loss.

        ############### GET LIST OF ALL TRAINING EPISODES
        action_IDs = os.listdir(dataset_dir+'actions/')
        for i in range(len(action_IDs)):
            action_IDs[i] = action_IDs[i].split('.')[0]
        video_IDs = os.listdir(dataset_dir+'numpy256/')
        for i in range(len(video_IDs)):
            video_IDs[i] = video_IDs[i].split('.')[0].split(',')[0]
        video_IDs = list(set(video_IDs))
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
                if numwords/duration_secs < 3.9:       # at 4 tokens/second, and 20fps, we average 0.199 word_tokens/frame -> 20% of frames have words. This means we average 80% silence. Using D=4, we get 1 - (4.0/1000)*1*4*50 ) = ~20% silence. Any fewer than 4 tok/sec results in >20% silence tokens in context.
                    continue
                else:
                    ham+=1
                    word_IDs.append(word_files[i].split('.')[0])
        print("Filtered videos with. low WPM. remaining :", str(int(100*(float(ham)/(i+1))))+'%')


        unique_ids = list( set(word_IDs).intersection(set(video_IDs)).intersection(set(action_IDs)) )
        self.unique_ids = sorted(unique_ids)
        #self.unique_ids.remove('6_WwMk1rBBk') ############################################################################################# @DEBUG
        print('videos:',len(video_IDs), 'transcripts:',len(word_IDs), 'actions:',len(action_IDs), 'GOT',len(self.unique_ids),'VIDEOS!')

        # Create tuples of (video_path, words_path, action_path) for each unique_id
        demonstration_tuples = []
        for unique_id in self.unique_ids:
            video_id = unique_id
            action_path = os.path.abspath(os.path.join(dataset_dir+'actions', unique_id + ".IDMaction"))
            words_path = os.path.abspath(os.path.join(dataset_dir+'transcripts', unique_id)) #google auto turns .txt into .gdoc so we cant use that
            np_folder = os.path.abspath(os.path.join(dataset_dir+'numpy256/'))
            demonstration_tuples.append((video_id, np_folder, words_path, action_path))

        assert n_workers <= len(demonstration_tuples), f"n_workers should be lower or equal than number of demonstrations {len(demonstration_tuples)}"





        self.finished_videos=[]




        # create file to store new dataloader log
        with open('TRAINING/dataloader.log', 'w') as file:
          pass
        # Repeat dataset for n_epochs times, shuffling the order for
        # each epoch
        #print("###########################################################################################################",random.Random(0).random())
        rng = random.Random(1)
        self.demonstration_tuples = []
        for i in range(n_epochs):
            demonstration_tuples_copy = demonstration_tuples.copy()
            rng.shuffle(demonstration_tuples_copy) # use seed so if training crashes, we can find the epoch and subseq it occurred at and set the dataloaders to continue from that point
            self.demonstration_tuples += demonstration_tuples_copy

        self.task_queue = Queue()
        self.n_steps_processed = 0
        for trajectory_id, task in enumerate(self.demonstration_tuples):
            self.task_queue.put((trajectory_id, *task))
        for _ in range(n_workers):
            self.task_queue.put(None)

        if not os.path.isdir('/vidcache'):
            os.mkdir('/vidcache')

        self.output_queues = [Queue(maxsize=max_queue_size) for _ in range(n_workers)]
        self.quit_workers_event = Event()
        self.processes = [
            Process(
                target=data_loader_worker,
                args=(
                    self.task_queue,
                    output_queue,
                    self.quit_workers_event,
                    F_SEQ_LEN,
                    max_subseqs_per_traj,
                    start_time,
                    LM_TIMEOUT_RATE,
                    LM_SILENCE_TOKEN,
                    words_only,
                    0,
                    n_workers
                ),
                daemon=True
            )
            for output_queue in self.output_queues
        ]
        for process in self.processes:
            process.start()




    def __iter__(self):
        return self

    def __next__(self):
        batch_frames = []
        batch_frames_ms = []
        batch_words = []
        batch_words_labels = []
        batch_actions_buttons = []
        batch_actions_camera = []
        batch_video_id = []
        batch_subseq = []
        batch_vpt_buttons = []
        batch_vpt_camera = []
        batch_finished_videos = []

        for i in range(self.batch_size):
            workitem = self.output_queues[self.n_steps_processed % self.n_workers].get(timeout=QUEUE_TIMEOUT)
            if workitem is None:
                # Stop iteration when first worker runs out of work to do.print
                # Yes, this has a chance of cutting out a lot of the work,
                # but this ensures batches will remain diverse, instead
                # of having bad ones in the end where potentially
                # one worker outputs all samples to the same batch.
                raise StopIteration()
            video_id, subseq_id, frames_seq, frames_seq_ms, words_seq, words_labels, actions_buttons, actions_camera, finished_video = workitem

            batch_frames.append(frames_seq)
            batch_frames_ms.append(frames_seq_ms)
            batch_words.append(words_seq)
            batch_words_labels.append(words_labels)
            batch_actions_buttons.append(actions_buttons)
            batch_actions_camera.append(actions_camera)
            batch_video_id.append(video_id)
            batch_subseq.append(subseq_id)

            if not finished_video in batch_finished_videos:
                batch_finished_videos.append(finished_video)
            self.n_steps_processed += 1
        # NOTE: data_loader takes in an ID (file locations) and returns {buttons:[t,20],camera:[t,2]}
        # batch_x is now an array of these, when in actuality we want a single {buttons:[b,t,20],camera:{b,t,2}}. similar with frames and words
        batch_frames = {'img':np.concatenate(batch_frames, axis=0),
                        'ms':np.concatenate(batch_words_labels, axis=0)}
        
        batch_words = {'input_ids':np.concatenate(batch_words, axis=0),
                        'labels':np.concatenate(batch_words_labels, axis=0)}
        
        batch_actions = {'buttons':np.concatenate(batch_actions_buttons, axis=0),
                        'camera':np.concatenate(batch_actions_camera, axis=0)}


        return batch_video_id, batch_subseq, batch_frames, batch_words, batch_actions, batch_finished_videos

    def __del__(self):
        for process in self.processes:
            process.terminate()
            process.join()
