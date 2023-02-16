import sys, os



dir = os.getcwd()
input_folder = "J:\\VLPT\\Data Gathering\\DATASET\\all\\videos\\"
output_folder = 'J:\\VLPT\\Data Gathering\\DATASET\\all\\clean_videos\\'

input_videos = os.listdir(input_folder)




# GET LIST OF HAM VIDEOS
with open('DATASET/all/ham_video.txt','r') as ham_videos:
    ham_videos = ham_videos.read().split('\n')
with open('DATASET/all/ham_transcript.txt','r') as ham_transcript:
    ham_transcript = ham_transcript.read().split('\n')
with open('DATASET/all/ham_metadata.txt','r') as ham_metadata:
    ham_metadata = ham_metadata.read().split('\n')

ham_links = list(set(ham_videos).intersection(set(ham_metadata)))




# COMPRESS HAM VIDEOS & MOVE TO CLEAN FOLDER
for videoname in input_videos:

    if not (videoname.split('.')[0] in ham_links):
        continue

    inputfile = input_folder+videoname
    outputfile = output_folder+videoname #'.'.join(videoname.split('.')[0:-1])+'.mp4'

    #command = 'ffmpeg -i "'+inputfile+'" -filter:v fps=20 -vcodec libx265 -preset medium -crf 30 "'+outputfile+'"'
    #command = 'ffmpeg -i "'+inputfile+'" -filter:v fps=20 -vcodec libaom-av1 -preset medium -crf 23 "'+outputfile+'"'
    #command = 'ffmpeg -y -vsync 0 -hwaccel cuda -hwaccel_output_format cuda -i "'+inputfile+'" -filter:v fps=20 -vcodec hevc_nvenc -b:v 5M "'+outputfile+'"'
    command = 'move "'+inputfile+'" "'+outputfile+'"' #'ffmpeg -n -hwaccel cuda -hwaccel_output_format cuda -i "'+inputfile+'" -filter:v fps=20 -vcodec hevc_nvenc -cq:v 30 -preset slow -b:v 5M "'+outputfile+'"'

    os.system(command)