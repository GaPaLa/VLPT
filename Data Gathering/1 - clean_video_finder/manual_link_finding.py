"""
if False:
    !pip install keyboard
    !pip install yt_dlp
    #!pip uninstall torch
    # os.system('c: && cd ProgramData/Anaconda3 && python -m pip install --force-reinstall torch torchvision torchaudio --extra-index-url https://download.pytorch.org/whl/cu117')
    #!pip uninstall torchvision
    #!pip uninstall torchaudio
    #!pip install --force-reinstall torch torchvision torchaudio --extra-index-url https://download.pytorch.org/whl/cu117
    !pip install numpy==1.21.5
    !pip install ftfy regex tqdm
    !pip install https://github.com/aboSamoor/pycld2/zipball/e3ac86ed4d4902e912691c1531d0c5645382a726
    !pip install git+https://github.com/openai/CLIP.git
    !pip install transformers
    !pip install sacremoses
    !pip install textblob
    !pip install pickle
    !pip install matplotlib
"""
import requests
import yt_dlp
import cv2
import os
#from sklearn import svm
import numpy as np
import pycld2 as cld2
import re
import json
import keyboard
import time

















############################################################################################### THE POINT OF THE FOLLOWING CODE IS TO GET A BUNCH OF MINECRAFT-RELATED VIDEO LINKS. THAT IS IT. get the from YT search and manual selection. account for whats already beens een before adding something.
all_links_seen = []
spam_links=[]
ham_links=[]
downloaded_videos=[]










if False:
    # MANUAL LINKS IMPORT
    # get the list of manually added links to check, get video data, add to video list
    with open('spamham_samples/manual_clean_ids.txt', 'r') as file:
        manual_links = file.read().split('\n')

    # add manually selected links to list of links to check for download
    # if this manual link to add was already added by all seen links, done add it again
    for link in manual_links:
        if not link in all_links_seen:
            all_links_seen.append(link)


    """# LOAD VIDEO DETAILS FROM FILE
    import pickle
    #pickle.dump(all_videos_to_search, open('manual_metad','wb'))
    all_videos_to_search = pickle.load(open('manual_metad','rb'))"""





















    
# DEFINE SEARCH & REGEX TERMS
# Search terms
search_queries = [
    "minecraft tutorial survival vanilla new world lets play",
    "minecraft tutorial hardcore new world lets play 1.16",
    "minecraft how to survival vanilla new world lets play 1.16",
    "minecraft tutorial first start survival lets play 1.16",
    "minecraft lets play tutorial survival new world 1.16",
    "minecraft lets play tutorial hardcore new world",
    "minecraft guide walkthrough survival new world 1.16",
    "minecraft how to first night playthrough",
    "minecraft tutorial lets play no webcam 1.16",
    "minecraft survival hardcore tutorial how to guide learn basics beginner lets play walkthrough gameplay new world first night first day 1.16",
    "minecraft survival vanilla guide let's play",
    "minecraft vanilla survival let's play",
    "minecraft vanilla hardcore let's play",
    "minecraft early game lets play vanilla"
]

# Title/tags
exclusion_terms = [
    'mods',
    'modded',
    'modding',
    '-mod',
    'ps3',
    'ps4',
    'ps5',
    'xbox',
    'bedrock',
    'pocket',
    'bed rock',
    'no commentary',
    'creative',
    'multiplayer',
    'server',
    'playstation',
    'timelapse',
    'animation',
    'minecraft pe',
    'pocket edition',
    'bedrock edition',
    'skyblock',
    'realistic',
    'shader',
    'how to install',
    'how to download',
    'realmcraft',
    'realms',
    'animation',
    'animatic',
    'music video',
    'movie',
    'friend',
    'parody',
    'island',
    'skyblock',
    'squad',
    'hack',
    'hacked',
    'hacking',
    'battle',
    'minecraft but',
    'minecraft, but',
    'sky',
    'music',
    'resource pack',
    'rlcraft',
    'dragon',
    'pokemon',
    'treecapi',
    'challenge',
    'download',
    'install',
    'deutsch',
    'german',

    'pixel art', 'pixle art'
    

    'feat.',
    ' we ',  # dont want multiplayer
    ' w/',
    ' gang ',
    ' multi-play',
    ' multiplay'

    'smp',
    'rtx',
    'feature',
    'lucky block',
    'bedrock',
    'pocket',
    ' pe ',
    'unturned',
    'roblox',
    'dimension',
    ' top ',
    'minigame',
    'endercraft',
    

    'broken', #dont want hacks or glitches
    'broke',
    'glitch',
    'texturepack',
    'texture pack',
    'resourcepack',
    'resource pack',
    'pixelmon',
    'pixlemon',
    'dual',
    'prison',
    
    'modpack',
    ' end', # dont want late game
    'nether',
    'dimension',
    'dragon',
    'wither',
    'custom',
    'roleplay',
    'troll',
    'clon',
    'ancient',
    'stranger',
    'zoo',
    'hunger game',
    'secret',
    'tips',
    'trick',
    'hangout',
    'hang out',
    'hanging',
    'fight',
    'bending',
    'bug',
    'issue',
    ' vs ',
    'react',
    'void',
    'hide and seek',
    'hero',
    'herobrine',
    'kill',
    'bukkit',
    'scien',
    'toy',
    'real life',
    'irl',
    'buddy',
    'buddie',
    ' buds ',
    ' bud ',
    'drone',
    'exile',
    'mini-game',
    'mini game',
    'competit',
    ' scp ',
    'auto',
    'inferno',
    'faction',
    'moon',
    'planet',
    ' pvp ',
    'cringe',
    'hypixel',
    'wii',
    'nintendo',
    'psycho',
    'blooper',
    'fnaf',
    'police',
    'military',
    'bedwar',
    'bed war',
    'jail',
    'story',
    'five nights at',
    'adventure map',
    'amongus',
    'among us',

    # languages
    ' le ',
    'survie'



    'no commentary', # filters to ensure useful langauge input
    'silent',
    'asmr',
    'no talking',
    'silent',
    'silence'
]


title_inclusion_terms_minedojo = [
    ['minecraft'],
#    ['surviv', 'hardcore'],
#    ['guide', 'tutorial', 'how to', 'surviv', 'learn', 'basic','noob','dummies'],
    ["let's play", "lets play",'walkthrough','playthrough', 'walk through', 'play through', 'gameplay', "play", 'episode', 'survival', 'ep.', 'chapter', 'chap.', 'start', 'beginning', 'new world', 'new game', 'fresh world', 'fresh game', 'clean world', 'clean game', 'new play', 'fresh play', 'from scratch']
#    ['first','begin','start','fresh','new'],
#    ['world','game','play','day','night']
#    ['new world', 'brand new', ' seed ', 'first', ' 1', ' 01', ' 001', 'early', 'one', 'fresh', 'start', 'begin']
] 

#@@@
title_inclusion_terms_minedojo=[['new world', 'new save', 'episode 1 ', 'episode 001', 'episode 01', 'ep. 1 ', 'ep 1 ', 'ep. 01', 'ep 01', 'ep 001', 'ep.001', 'episode #1', 'episode #001', 'episode #01', 'ep. #1', 'ep #001', 'ep. #001']]

#@inproceedings{fan2022minedojo,
#  title = {MineDojo: Building Open-Ended Embodied Agents with Internet-Scale Knowledge},
#  author = {Linxi Fan and Guanzhi Wang and Yunfan Jiang and Ajay Mandlekar and Yuncong Yang and Haoyi Zhu and Andrew Tang and De-An Huang and Yuke Zhu and Anima Anandkumar},
#  booktitle = {Thirty-sixth Conference on Neural Information Processing Systems Datasets and Benchmarks Track},
#  year = {2022},
#  url = {https://openreview.net/forum?id=rc8o_j8I8PX}
#}





# JSON->python extraction from https://stackoverflow.com/questions/13938183/python-json-string-to-list-of-dictionaries-getting-error-when-iterating
if False:
    print('importing minedojo video metadata')

    with open("D:\VLPT\Data Gathering\DATASET\youtube_full.json",'r') as data:
        data = data.read()
        minedojo_videos = json.loads(data)
    # add links in here to links to check
    for video_inf in minedojo_videos:
        link = video_inf['id']
        title=video_inf['title']
        duration = video_inf['duration']

        #print('\n\n\n\n\n',title)
            
        try:
            is_spam=False
            # check title is english
            _,_,details=cld2.detect(title)
            if details[0][0]!= 'ENGLISH':         # get only English videos
                is_spam=True
                #print('lang spam title', details[0])

            # check video duration
            if duration < 600:
                is_spam=True
                #print('durationspam=')

            # CHECK TITLE AGAINST REGEX
            for term in exclusion_terms:
                
                if term.lower() in title.lower():
                    is_spam = True
                    #print(term.lower(), 'titlespamex=',term)
            for term in title_inclusion_terms_minedojo:
                includes_subterm=False
                for subterm in term:
                    if subterm.lower() in title.lower():
                        includes_subterm = True
                if not includes_subterm:
                    is_spam=True
                    #print(term, 'titlespaminc=')
            # Check for multiplayer in title [... with H..] e.g. "playing minecraft with Harry"
            if 'with' in title.lower():
                try:
                    char_after_with = title[title.lower().find('with') + 5]
                    if char_after_with.lower() != char_after_with:
                        is_spam = True
                        #print(title, 'titlespam: "with"')
                except:
                    pass
            #check for episode number in title and make sure it is small
            nums = re.findall(r'\d+', title) #https://stackoverflow.com/questions/17007257/python-3-finding-the-last-number-in-a-string
            if len(nums)>0:
                if int(nums[len(nums)-1]) > 100:
                    is_spam=True
                    #print('episode spam NUMBER,', title)    


            if not is_spam:
                if not link in all_links_seen:
                    all_links_seen.append(link)
                    
        except:
            pass
if True:
    # UPDATE TEXT FILE OF ALL LINKS SEEN
    with open('all_seen_links.txt', 'r') as file:
        all_seen_links_file = file.read().split('\n')
        for link in all_seen_links_file:
            if not link in all_links_seen:
                all_links_seen.append(link)

    while '' in all_links_seen:
        all_links_seen.remove('')

############################################################################################################################################ FINISH GETTINGS LINKS WORTH CHECKING. WRITE LINK TO FILE






  # RE-CHECK SPAM HAM IN FILES
with open('ham_video.txt','r') as ham_videos:
    ham_videos = ham_videos.read().split('\n')
with open('ham_transcript.txt','r') as ham_transcript:
    ham_transcript = ham_transcript.read().split('\n')
with open('ham_metadata.txt','r') as ham_metadata:
    ham_metadata = ham_metadata.read().split('\n')

ham_links = list(set(ham_videos))



# gt list of already dientifited spam
with open('spam_video.txt', 'r') as file:
    spam_video = file.read().split('\n')
with open('spam_transcript.txt', 'r') as file:
    spam_transcript = file.read().split('\n')
with open('spam_metadata.txt','r') as spam_metadata:
    spam_metadata = spam_metadata.read().split('\n')

spam_links = list(set(spam_transcript).union(set(spam_video)).union(set(spam_metadata)))

############################################################################################################################################  FROM SEEN LINKS THAT HAVE NOT BEEN CLASSIFIED, ONE AT A TIME, GET YT_SEARCH METADATA FROM IT ONLINE, CHCK IS FOR SPAM/HAM, CLASSIFY AND SEND TO APPROPRIATE FILE.

# from seen links and spam/ham so far, get list of links that need to be classified
import random
links_to_check = list(   set(all_links_seen) - set(spam_links).union(set(ham_links))   ) 
random.shuffle(links_to_check)


len(links_to_check)





















print(cv2.__version__)

# CHECK HAM/SPAM ON EMTADATA, THEN VIDEO, THEN AUDIO

### FOR CHECKING VIDEO SPAM HAM.

def elapsedTime(start):
    now = time.time()
    print('['+str(now-start)+']')
    return now

# https://pypi.org/project/yt-dlp/#usage-and-options
def longer_than_10_mins(info, *, incomplete):
    """Download only videos longer than 10 minutes"""
    duration = info.get('duration')
    if duration and duration < 600:
        return 'The video is too short'
ydl_opts={'match_filter':longer_than_10_mins, 'download':False, 'fps':60}









from random import shuffle

# load target LM's tokenizer for checking WPM as tokens/ms
YDL_OPTIONS = {'noplaylist':True, 'download':False, 'ignoreerrors':True, 'externaldownloader':'aria2c', 'quiet':True, 'downloader':'aria2c' } #'list_subs':True,




def convert_seconds_to_time(seconds):
    minutes, seconds = divmod(seconds, 60)
    return f"{minutes}:{seconds:02d}"


############# PARALLELISE REPRESENTATIVE FRAMES FETCHING
from multiprocessing.pool import ThreadPool
pool = ThreadPool()
from PIL import Image
import numpy as np
import io
def get_frame_fraction(data):
    target_frames, url, process_idx, duration = data
  
    sec =  round( ( (((duration/target_frames) * process_idx ))+5)*((duration-15)/(duration+5))  )   # shift all segments up five seconds, squish so end at 15 secs before ending
    time = convert_seconds_to_time(sec)

    #cap.set(1, round(frame_index))           
    #ret, frame = cap.read()
    #if not ret:
    #    print('NOT RET!')
    #    return None
    #cap.release()
    #frame = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
    response = requests.get(
    "https://api.clipshot.dev/frame",
    params={
        "youtube_id": url,
        "timestamp": time,
        "key": "B792BF13-EA554CF8-A2BD9288-6E8B17C3", # Replace with your own
        },
    )
    if response.status_code == 200:
        img = Image.open(io.BytesIO(response.content))
        img_np = np.array(img)[...,::-1]

    else:
        return None

    
    return img_np

















import psutil

from concurrent.futures import ThreadPoolExecutor
from queue import Queue
import threading

buffer = Queue()
stop_event = threading.Event()

def producer(links):
    current_link = 0
    while not stop_event.is_set():
        while psutil.virtual_memory()[1] < 1024975360:
            time.sleep(10)
            pass
    	
    	
        link = links[current_link]
        is_spam=False

        ## from link, fetch metadata
        with yt_dlp.YoutubeDL(YDL_OPTIONS) as ydl:
            video = ydl.extract_info(link, download=False)

        if video == None:
            print("VIDEO NONE NULL")
        else:






            ############################################################################## CHECK LINK METADATA FOR SPAM/HAM (title, tags, transcript...)
            if not is_spam:

                ############################# check video duration
                if video['duration'] < 480: # 8 minutes minimum
                    is_spam=True
                    #print('durationspam=')

                ################################ CHECK TITLE AGAINST REGEX
                title = video['title']
                for term in exclusion_terms:
                    if term.lower() in title.lower():
                        is_spam = True
                        #print(term.lower(), 'titlespamex=',title)
                for term in title_inclusion_terms_minedojo:
                    includes_subterm=False
                    for subterm in term:
                        if subterm.lower() in title.lower():
                            includes_subterm = True
                    if not includes_subterm:
                        is_spam=True
                        #print(term, 'titlespaminc=')
                # Check for multiplayer in title [... with H..] e.g. "playing minecraft with Harry"
                if 'with' in title.lower():
                    char_after_with = title[title.lower().find('with') + 5]
                    if char_after_with.lower() != char_after_with:
                        is_spam = True
                        #print(title, 'titlespam: "with"')
                #check for episode number in title and make sure it is small
                #nums = re.findall(r'\d+', title) #https://stackoverflow.com/questions/17007257/python-3-finding-the-last-number-in-a-string
                #if len(nums)>0:
                #    if int(nums[len(nums)-1]) > 30:
                #        is_spam=True
                #        #print('episode spam NUMBER,',title)    
                
                
                # CHECK DESCRIPTION AGAINST REGEX
                # cut description or referring to a buch of different youtube videos
                description = video['description'].lower()

                if 'youtube.' in description:
                    description = description.split('youtube.')[0]
                if '//youtu.be' in description:
                    description = description.split('//youtu.be')[0]

                for term in exclusion_terms:
                    #if term in [' we ']: # these words are check for in title but not description becuase description is too likely to have them while being good
                    #    continue
                    if term in description       and (not (term=='creative' and ('survival' in description or 'hardcore' in description)))            and ( not (term=='server' and ('discord server' in description) ))         and not(term==' we ')     and not(term=='music'):
                        is_spam=True
                        #print(term, 'descriptionspam')

                # CHECK TAGS AGAINST SPAM TAGS (remove ps3/xbox/console/bedrock/ps4/ps5/xbox series...
                tags = video['tags']
                for term in exclusion_terms:
                    if term in tags:
                        is_spam=True
                        #print(term, 'tagspam=')
                        
                # CHECK LANGUAGE (title, description)
                _,_,details=cld2.detect(description)
                if details[0][0]!= 'ENGLISH':         # get only English videos
                    is_spam=True
                    #print('description spam language',details[0])
                _,_,details=cld2.detect(title)
                if details[0][0]!= 'ENGLISH':         # get only English videos
                    is_spam=True
                    #print('lang spam title', details[0])
            if is_spam:
                pass
                #with open('spam_metadata.txt', 'a') as file:
                #    file.write(link+'\n')

            else:
                pass
                #with open('ham_metadata.txt', 'a') as file:
                #    file.write(link+'\n')











            ###################################### CHECK VDIDEO HAM SPAM
            if (link in ham_videos) or (link in spam_links): # dont re-check already classified videos.
                pass
            else:
                if (not is_spam) and not (link in ham_videos):
                    formats = video['formats']
                    #shuffle(formats)
                    duration = video['duration']
                    #print(formats)
                    
                    if False:
                        for f in formats:
                            if f.get('format_note',None)=='720p':# or f.get('format_note',None)=='480p' or f.get('format_note',None)=='360p':
                                url = f.get('url',None)
                                break
                    url = link

                    #print("!!!FOUND CLEAN METADATA! GATHERING FRAMES FOR", title)
                    #wait for free memory before dwonloading frames

                    diameter=4
                    frame_reqs = []
                    for i in range(diameter**2 +1):
                        frame_reqs.append((diameter**2, url, i, duration))
                    results = pool.map(get_frame_fraction, frame_reqs)

                    # if some frames failed, they are all corrupted
                    video_failed=False
                    for image in results:
                        if image is None:
                            print("VIDEO FAILED")
                            video_failed=True

                    if not video_failed:
                        joint = np.zeros([0,results[1].shape[1]*diameter,3], dtype=np.uint8)
                        for i in range(diameter):
                            row = np.concatenate(results[i*diameter:i*diameter+diameter], axis=1)
                            joint = np.concatenate([joint, row], axis=0)
                        print('!!!PRODUCED!!!')
                        buffer.put((link,joint, title, description))
                        #print(buffer.qsize())

        current_link +=1


import gc

def consumer():
    print('in consumer')
    while not stop_event.is_set():

        link, joint, title, description = buffer.get()
        print('!!!CONSUMED!!!')

        if True:
            joint = cv2.resize(joint, (1280, 720))
            #window.set_data(joint)
            cv2.imshow('i show u miecraft pls respond',joint)
            cv2.waitKey(1000)
            del joint
            gc.collect()    

            # get user response label for current image    https://www.geeksforgeeks.org/how-to-detect-if-a-specific-key-pressed-using-python/
            print('\n\n\n\n\n',title, description)
            print('taking user input [1,2,3]')
            valid=False
            while not valid:
                label = input('')
                try:
                    label=int(label)
                    if label in [1,2,3]:
                        valid=True
                    else:
                        print('not [1,2,3] try again')
                except:
                        print('NaN, try again')
            if label ==1:
                is_spam=False
            else:
                is_spam=True
            cv2.imshow('i show u miecraft pls respond',np.zeros([1280,720,3], dtype=np.uint8))

            # save response to file
            if is_spam:
                with open('spam_video.txt','a') as file:
                    line = link+','+str(label)+'\n'
                    file.write(line)
            else:
                with open('ham_video.txt','a') as file:
                    line = link+','+'\n'
                    file.write(line)

            print("!!!SAVED RESPONSE!!!", label)


print(os.getcwd())
os.chdir('/home/idmi/')





from threading import Thread
if False:
    print('start exec')
    with ThreadPoolExecutor() as executor:
        # start producer threads
        n_threads=20
        links_per_thread = int(len(links_to_check)/n_threads)
        for i in range(n_threads):
            print(i)
            start = i*links_per_thread
            end = min( len(links_to_check),  start+links_per_thread)
            this_threads_links = links_to_check[start:end]
            executor.submit(producer, this_threads_links.copy())
        
        # start consumer thread
        executor.submit(consumer)
else:
    
    from threading import Thread

    print('start exec')
    # start producer threads
    n_threads = 40
    links_per_thread = int(len(links_to_check) / n_threads)
    for i in range(n_threads):
        print(i)
        start = i * links_per_thread
        end = min(len(links_to_check), start + links_per_thread)
        this_threads_links = links_to_check[start:end]
        t = Thread(target=producer, args=(this_threads_links.copy(),))
        t.start()

    # start consumer thread
    t = Thread(target=consumer)
    t.start()
