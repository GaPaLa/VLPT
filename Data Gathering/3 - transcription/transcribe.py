import ssl
ssl._create_default_https_context = ssl._create_unverified_context
from transformers import TransfoXLTokenizer
import torch
torch.cuda.is_available()
import os
# load target LM's tokenizer
tokenizer = TransfoXLTokenizer.from_pretrained("transfo-xl-wt103")
# load ASR model 
# https://github.com/jianfch/stable-ts
from stable_whisper import load_model, stabilize_timestamps

import numpy as np



input_videos_location = 'DATASET/train/videos/'
output_location = 'DATASET/train/transcripts/'





### Use ASR library: file -> words + ms
model = load_model('base')





replace = { ' redstone':' red stone',
            ' gasts': ' ghouls',
            ' gast': ' ghoul',
            ' ghasts': ' ghouls',
            ' ghast': ' ghoul',
            ' disenchant':' dis enchant',
            ' disenchantment':' dis enchantment',
            ' piglin': ' goblin',
            ' YouTube':' Twitch', #youtube is <unk> but twitch is not :P
            ' YouTuber':' streamer',
            ' glowstone':' glow stone',
            ' Oop':' oops',
            ' oops':' oops',
            ' unbreaking':' un breaking',
            ' PVP':' player fighting',
            ' PVE':' player fighting',
            ' composter':' compost',
}



filenames = os.listdir(input_videos_location)
import random
random.shuffle(filenames)

for filename in filenames:


    # check if this video has already been transcribed
    link = '.'.join(filename.split('.')[0:-1])
    transcribeds = os.listdir(output_location)
    transcribeds_ = []
    for transcript in transcribeds:
        translink = '.'.join(transcript.split('.')[0:-1])
        transcribeds_.append(translink)
    if link in transcribeds_:
        continue


    # from stable whisper, extract words and timestamps
    print('transcribing', filename)
    results = model.transcribe(input_videos_location+filename)#, language='en'
    stab_segments = results['segments']
    first_segment_word_timestamps = stab_segments[0]['whole_word_timestamps']
    stab_segments = stabilize_timestamps(results, top_focus=True)

    ASR_words=[]
    ASR_secs=[]
    for i in range(len(stab_segments)):
        for j in range(len(stab_segments[i]['word_timestamps'])):
            wordms = stab_segments[i]['word_timestamps'][j]
            token = wordms['word']
            ms = wordms['timestamp']
            #print(token, ms)
            ASR_words.append(token)
            ASR_secs.append(ms)






    # convert words to agent's LM tokens and transfer ms timestamps accordingly - HARD. NOTE: ASSUMES THAT decode(ASR_tokenizer(text)) = decode(GPT2_tokenizer(text)), WHICH MAY NOT BE TRUE
    # NOTE: The Transfo_XL LM used has not BOS token. This code keeps this in mind.
    string = ''.join(ASR_words)
    agent_words_index = tokenizer(string)['input_ids'] #remove EOS token
    # decode token indices to strings
    agent_words=[]
    for t in range(len(agent_words_index)):
        agent_words.append(tokenizer.decode([agent_words_index[t]]))

    word_ms = [0]*len(agent_words)
    tok1_index,ch1_index = 0,0
    tok2_index,ch2_index = 0,0
    current_token_timestamps = [] # is modified though. whe n a new agent token is found is is rest to empty. wesearch through every timestamp that token could occur at and save it to list list. we then decide when the token occured.
    step=0
    while tok2_index<len(agent_words) and tok1_index<len(ASR_words): # keep going until all agent words have timestamps

        # according to timestamp at current ASR token that the curretnASR character index is at, give the timestamp associated with that word to the current token that agents character index is at.
        current_token_timestamps.append(ASR_secs[tok1_index])

        char1 = ASR_words[tok1_index][ch1_index] #['hi ', ' there', ' how']
        char2 = agent_words[tok2_index][ch2_index] #['hi ', ' there', ' how']    print(step)

        print(ASR_words[tok1_index], char1, char2, tok1_index, tok2_index)


        # ensure that we are at the same point in speech in both texts. If not, could be ebcause ok tokenizer differences: accounted for here: 1) ASR uses spaces in outputted words, agent tokenizer does not. 2) agent tokenizer outputs <UNK> for unknown words. 
        if char1 != char2:
            print('mismatch on: "'+char1+'", "'+char2+'"')
            if char1==' ': #deak with ASR tokeniser giving spaces berfore/after words and agent tokenizer not doing so
                ch1_index+=1 # ASR words have spaces, agent does not. so we ignore space difference and scooth char1 ahead if it hits a space. now that we have done this change, restart the iteration so we can check if they are now different again or need to be shifted again because 'Hello ', ' there'
                if ch1_index == len(ASR_words[tok1_index]):
                    print('1')
                    tok1_index += 1
                    ch1_index=0
                continue
            if char2 == '<': # deal with <UNK> - add it and timestamp and skip past rest of <UNK> characters
                final_timestamp = min(current_token_timestamps)
                #if word_ms[tok2_index-1] == final_timestamp:
                #    final_timestamp += 1 # it doesnt actually matter if one token in ASR ends up as two tokens in agent_words, as long as order is preserved, since the agent formats them into one token per frame and uses D, so it takes close otegther words back roughly into average WPM anyway. What this does is if one ASR token produces two Agent tokens, they will have the same timestamp, so we need to say the second one happened 1ms later than it did to preserve order.
                word_ms[tok2_index] = final_timestamp # save agent token ms to this list. This and agent_words
                tok2_index += 1
                ch2_index = 0
                current_token_timestamps=[]
                continue
            elif char2 == '@':   # agent tokenizer encodes '2.0' to ['2','@.@','0'] :(
                final_timestamp = min(current_token_timestamps)
                #if word_ms[tok2_index-1] == final_timestamp:
                #    final_timestamp += 1 # it doesnt actually matter if one token in ASR ends up as two tokens in agent_words, as long as order is preserved, since the agent formats them into one token per frame and uses D, so it takes close otegther words back roughly into average WPM anyway. What this does is if one ASR token produces two Agent tokens, they will have the same timestamp, so we need to say the second one happened 1ms later than it did to preserve order.
                word_ms[tok2_index] = final_timestamp # save agent token ms to this list. This and agent_words
                tok2_index += 1
                ch2_index = 0
                current_token_timestamps=[]
                continue
            else:
                # deal with <UNK>
                ch1_index+=1 # ASR words have spaces, agent does not. so we ignore space difference and scooth char1 ahead if it hits a space. now that we have done this change, restart the iteration so we can check if they are now different again or need to be shifted again because 'Hello ', ' there'
                if ch1_index == len(ASR_words[tok1_index]):
                    print('1')
                    tok1_index += 1
                    ch1_index=0
                continue
                #assert (char1 == char2), "tokenizer differences: original string cannot be converted"

        # move CHAR1 up one
        if ch1_index == len(ASR_words[tok1_index])-1:
            tok1_index += 1
            ch1_index=0
        else:
            ch1_index+=1

        # move CHAR2 up one
        if ch2_index == len(agent_words[tok2_index])-1:
            # reached end of current token. We have all ms that characters in token occurred across. just assume that this whole token occurred at first character's ms
            # final timestamp is just first observed subtimestamp, so that frame and word starts are aligned. This is important becasue it retain causality. The point is also to get across linguitic information, not copy timing, and when a word is stated is when the information is  decided, so that is what should be used to most acurately most linguistic information across time
            final_timestamp = min(current_token_timestamps)
            #if word_ms[tok2_index-1] == final_timestamp:
            #    final_timestamp += 1 # it doesnt actually matter if one token in ASR ends up as two tokens in agent_words, as long as order is preserved, since the agent formats them into one token per frame and uses D, so it takes close otegther words back roughly into average WPM anyway. What this does is if one ASR token produces two Agent tokens, they will have the same timestamp, so we need to say the second one happened 1ms later than it did to preserve order.
            
            word_ms[tok2_index] = final_timestamp # save agent token ms to this list. This and agent_words
            tok2_index += 1
            ch2_index=0

            current_token_timestamps = [] # reset timestamps associated with current token for next token
        else:
            ch2_index+=1


        step += 1




    # get final [words, ms] an save to file
    import numpy as np
    final_agent_words=agent_words_index
    final_agent_words_ms = (np.asarray(ASR_secs)*1000).astype(np.uint32)

    output_filename = output_location+link

    with open(output_filename,'w') as file:
        pass

    with open(output_filename, 'a') as file:
        for i in range(len(final_agent_words)):
            line = str(final_agent_words[i]) + ',' + str(final_agent_words_ms[i]) + '\n'
            file.write(line)

