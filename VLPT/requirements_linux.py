############################################### LINUX
import sys, os

#os.system("sudo apt install git")
#os.system("sudo add-apt-repository ppa:openjdk-r/ppa && sudo apt-get update && sudo apt-get install openjdk-8-jdk")
#os.system('pip install git+https://github.com/minerllabs/minerl@v1.0.0')
os.system('pip install -r "requirements.txt"')

#os.system('conda install cudatoolkit=11.7')
os.system('python3 -m pip install transformers')
#os.system("python3 -m pip install 'git+https://github.com/katsura-jp/pytorch-cosine-annealing-with-warmup' ")

#os.system('python3 -m pip install git+https://github.com/jianfch/stable-ts.git')
#os.system('python3 -m pip install -U openai-whisper')
#os.system('sudo apt update && sudo apt install ffmpeg')
os.system('python3 -m pip install sacremoses')

os.system('python3 -m pip install "numpy<1.24.0" ')
#from transformers import TransfoXLLMHeadModel
#LM = TransfoXLLMHeadModel.from_pretrained("transfo-xl-wt103") #, @later, load from stored weights

os.chdir('/media/idmi/DISSERTATN/VLPT/VLPT')
transfo_xl_dir = sys.exec_prefix + "/lib/python3.10/site-packages/transformers/models/transfo_xl/"
#transfo_xl_dir = "/media/idmi/Z/Ubuntu_folder/miniconda3/envs/minerl/lib/python3.8/site-packages/transformers/models/transfo_xl/"
#modelling_name = 'modeling_transfo_xl.py'
#os.rename(transfo_xl_dir+modelling_name , transfo_xl_dir+modelling_name+'.bak')
os.system("sudo apt install git")
os.system("sudo add-apt-repository ppa:openjdk-r/ppa && sudo apt-get update && sudo apt-get install openjdk-8-jdk")
os.system('pip install git+https://github.com/minerllabs/minerl@v1.0.0')
os.system('pip install -r "requirements.txt"')

os.system('conda install cudatoolkit=11.7')
os.system('pip install transformers')
os.system("pip install 'git+https://github.com/katsura-jp/pytorch-cosine-annealing-with-warmup' ")

#os.system('pip install git+https://github.com/jianfch/stable-ts.git')
#os.system('pip install -U openai-whisper')
#os.system('sudo apt update && sudo apt install ffmpeg')

transfo_xl_dir = sys.exec_prefix + "/lib/python3.8/site-packages/transformers/models/transfo_xl/"
modelling_name = 'modeling_transfo_xl.py'
os.rename(transfo_xl_dir+modelling_name , transfo_xl_dir+modelling_name+'.bak')
os.system( 'cp '+modelling_name +' '+ transfo_xl_dir+modelling_name)
utilities_name = 'modeling_transfo_xl_utilities.py'
os.rename(transfo_xl_dir+utilities_name , transfo_xl_dir+utilities_name+'.bak')
os.system( 'cp '+utilities_name +' '+ transfo_xl_dir+utilities_name)
