############################################### LINUX
import sys, os


# CORE
#os.system('pip3 install numpy --pre torch --force-reinstall --index-url https://download.pytorch.org/whl/nightly/cu117')
os.system('pip3 install opencv-python')

# LANGAUGE MODEL
os.system('pip3 install transformers')
os.system('sudo apt install ffmpeg')
os.system('pip3 install sacremoses')
from transformers import TransfoXLLMHeadModel
LM = TransfoXLLMHeadModel.from_pretrained("transfo-xl-wt103") #, @later, load from stored weights

#transfo_xl_dir = '/usr/local/lib/python3.9/dist-packages/transformers/models/transfo_xl/'   # <- for colab
##transfo_xl_dir = sys.exec_prefix + "/lib/python3.8/site-packages/transformers/models/transfo_xl/"
#modelling_name = 'modeling_transfo_xl.py'
#os.rename(transfo_xl_dir+modelling_name , transfo_xl_dir+modelling_name+'.bak')
#os.system( 'cp '+modelling_name +' '+ transfo_xl_dir+modelling_name)
#utilities_name = 'modeling_transfo_xl_utilities.py'
#os.rename(transfo_xl_dir+utilities_name , transfo_xl_dir+utilities_name+'.bak')
#os.system( 'cp '+utilities_name +' '+ transfo_xl_dir+utilities_name)

# TRANSCRIPTION
os.system('pip3 install openai-whisper')
os.system('pip3 install stable-ts==1.4.0')

# MINERL
os.system("sudo add-apt-repository -y ppa:openjdk-r/ppa && sudo apt-get update && sudo apt-get install -y openjdk-8-jdk")
#os.system('pip3 install git+https://github.com/minerllabs/minerl@v1.0.0') this never works. I got it to work once on 1 laptop. ill just copy the files over from there
#os.system('pip3 install gym==0.19')
#os.system('pip3 install gym3')
#os.system('pip3 install attr')
#os.system('python3 -m pip install "numpy<1.24.0" ')

# TOOLS
os.system('pip3 install https://github.com/aboSamoor/pycld2/zipball/e3ac86ed4d4902e912691c1531d0c5645382a726')
os.system("pip3 install 'git+https://github.com/katsura-jp/pytorch-cosine-annealing-with-warmup' ")
