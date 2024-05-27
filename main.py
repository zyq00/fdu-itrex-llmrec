from intel_extension_for_transformers.transformers import AutoModelForCausalLM, BitsAndBytesConfig
from transformers import AutoTokenizer
import pandas as pd
import pyarrow as pa
import torch
import json
from tqdm import tqdm 
import lancedb
from embed import *
from gradio import Chatbot
import gradio as gr
from llmHelper import *
from chatagent import *
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
from loguru import logger
logger.add("./log/main.log", backtrace=True, diagnose=True)
like_his=[]
dislike_his=[]
max_suggestions=100
user_history = []
ragManager=RAGManager()
movie_set =  pd.read_csv(env_config.movie_data_dir+'movie_set.csv', encoding='utf-8',on_bad_lines='skip')
movie_set=movie_set.astype(str)
movie_name_dict =movie_set.set_index('movie_id')['name'].to_dict()
movie_id_dict =movie_set.set_index('name')['movie_id'].to_dict()
def movie_dialogue(message, history):
    global like_his,dislike_his
    messages= []
    humans=[]
    for human, assistant in history:
        messages.append({"role": "user", "content": human})
        messages.append({"role": "assistant", "content": assistant})
    messages.append({"role": "user", "content": message})
    #从历史对话解析出用户喜好
    like,dislike = get_like_dislike(message)
    like =  [movie_id_dict[name] for name in like if movie_id_dict.get(name) is not None ]
    dislike =  [movie_id_dict[name] for name in dislike if movie_id_dict.get(name) is not None ]
    like_his +=like
    dislike_his +=dislike

    
    if len(like)>0:
        #rag retrive top50
        candidates = ragManager.get_candidates(messages)
        sasrec_result = ragManager.sasrec_predict(like_his,candidates)
        pmt_mng = Prompts()
        msg = pmt_mng.recommandation_prompt(messages ,sasrec_result)
    else :
        msg=messages
    #print(candidates)
    logger.info("rec msg:"+json.dumps(msg))

    return getresponse(msg)




def main():


    gr.ChatInterface(movie_dialogue, chatbot=Chatbot(min_width=200, height=800), theme=gr.themes.Soft()).launch(
        share=True)

        
    
    
    


if __name__ == '__main__':
    main()