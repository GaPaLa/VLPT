from transformers import TransfoXLTokenizer
import os, sys



# load target LM's tokenizer
tokenizer = TransfoXLTokenizer.from_pretrained("transfo-xl-wt103")




# load ASR model 
# https://github.com/jianfch/stable-ts
from stable_whisper import load_model
from stable_whisper import stabilize_timestamps
model = load_model('base')        
stab_segments = results['segments']
first_segment_word_timestamps = stab_segments[0]['whole_word_timestamps']
print(first_segment_word_timestamps)



# take in input file name and output trnscription file name
def audio_to_tokens_ms(audio_file, output_name):

    ### Use ASR library: file -> words + ms
    ASR_words=[]
    ASR_secs=[]
    results = model.transcribe(audio_file)
    stabilize_timestamps(results, top_focus=True)
    

    
    # convert words to tokens for agent and transfer ms timestamps accordingly - HARD. NOTE: ASSUMES THAT decode(ASR_tokenizer(text)) = decode(GPT2_tokenizer(text)), WHICH MAY NOT BE TRUE
    string = ''.join(ASR_words)
    agent_words = tokenizer(string)['token_ids']
    word_ms = [-50]*len(agent_words)
    tok1_index,ch1_index = 0,0
    tok2_index,ch2_index = 1,0 # start comparing tokens after BOS token. conveniently, this also sets BOS timestamp at ms=-50. 
    current_token_timestamps = []
    while tok2_index<len(agent_words): # keep going until all agent words have timestamps
        char1 = ASR_words[tok1_index][ch1_index]
        char2 = agent_words[tok2_index][ch2_index]
        assert (char1 == char2), "tokenizer differences: original string cannot be converted"

        current_token_timestamps.append(ASR_secs[tok1_index])
        
        if ch1_index == len(tok1_index):
            tok1_index += 1
            ch1_index=0

        if ch2_index == len(tok2_index):
            # finished that token. We have all ms that token occurred across. just assume that it occurred at first mention
            # final timestamp is just first observed timestamp, so taht starts are aligned. This is important becasue it retain causality. The point is also to get across linguitic information, not copy timing, and when a word is stated is when the information is  decided, so that is what should be used to most acurately most linguistic information across time
            final_timestamp = min(current_token_timestamps)
            if word_ms[tok2_index-1] == final_timestamp:
                final_timestamp += 1 # it doesnt actually matter if one token in ASR ends up as two tokens in agent_words, as long as order is preserved, since the agent formats them into one token per frame and uses D, so it takes close otegther words back roughly into average WPM anyway.
            
            word_ms[tok2_index] = final_timestamp
            tok2_index += 1
            ch2_index=0



    # convert timestamps from seconds to ms
    word_ms = int(word_ms*1000) 


    with open(output_name, 'w') as output_file:
        for word, ms in zip(agent_words, word_ms):
            line = word+', '+ms+'\n'
            output_file.write()

    return agent_words, word_ms





audio_folder = '..audio/folder'
audio_files = os.list_files(audio_file)
for file in audio_files:
    output_name = file.split('.opus')[0]+'.txt'
    audio_to_tokens_ms(file, file)





audio_to_tokens_ms('/media/idmi/Music/Dissertation/VLPT/Data Gathering/Audio/MC.opus', 'output.txt')
