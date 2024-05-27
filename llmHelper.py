from intel_extension_for_transformers.transformers import AutoModelForCausalLM, BitsAndBytesConfig
from transformers import AutoTokenizer
import pandas as pd
import pyarrow as pa
import torch
from tqdm import tqdm 
import lancedb
from embed import *
from SASRec.model import SASRec
import env_config
import numpy as np
import json
root_dir='./llm.Cache/'
data_dir ='./data/douban/moviedata/'
embed_model_name=root_dir+'maidalun1020/bce-embedding-base_v1'
model_name_or_path = root_dir+'Qwen/Qwen-7B-Chat_4bit'
#model_name_or_path=root_dir+'maidalun1020/bce-embedding-base_v1'
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
from loguru import logger
logger.add("./log/llmHelper.log", backtrace=True, diagnose=True)
max_suggestions=50
class RAGManager:
    def __init__(self):
        self.embed_tokenizer = AutoTokenizer.from_pretrained(embed_model_name,trust_remote_code=True)
        self.embed_tokenizer.model_max_length=256
        self.embed_model = AutoModelForCausalLM.from_pretrained(embed_model_name,trust_remote_code=True,load_in_4bit=True)

        print('bce-embedding-base_v1 loaded')
        args = env_config.args
        self.sasrec_model = SASRec(184333, 34781677, args).to('cpu') # no ReLU activation in original SASRec implementation?
        for name, param in self.sasrec_model.named_parameters():
            try:
                torch.nn.init.xavier_normal_(param.data)
            except:
                pass # just ignore those failed init layers
        
        self.sasrec_model.load_state_dict(torch.load(args.state_dict_path, map_location=torch.device('cpu')))
        '''  
        #量化版qwen
        self.qwen_tokenizer = AutoTokenizer.from_pretrained(model_name_or_path,trust_remote_code=True)
        woq_config = BitsAndBytesConfig(
                                        load_in_4bit=True,bnb_4bit_use_double_quant=True,bnb_4bit_quant_type="nf4",
                                        bnb_4bit_compute_dtype=torch.float16)
                                        
        self.qwen_model = AutoModelForCausalLM.from_pretrained(  
                                                            model_name_or_path,
                                                            quantization_config=woq_config,trust_remote_code=True
                                                        )     
        print(model_name_or_path+' loaded')  
        '''                        
        uri ="./data/douban/lmdb_bce"
        lmdb = lancedb.connect(uri)
        self.tbl = lmdb.open_table("movies")
        self.movie_set =pd.read_csv(data_dir+'movie_set.csv', encoding='utf-8',on_bad_lines='skip') 
        self.movie_set['movie_id'] = self.movie_set['movie_id'].astype(str)
                
        self.movie_set=self.movie_set.astype(str)
        self.movie_name_dict =self.movie_set.set_index('movie_id')['name'].to_dict()
    
    def getNameById(self,id):
        row = self.movie_set['movie_id'] = id
        if len(row)>=1:
            return row.loc[1]['movie_id']
        else :
            return None
    
    def get_candidates(self,messages):
        input = json.dumps(messages)
        logger.info("get_candidates")
        embeding = embed_func(input,self.embed_tokenizer,self.embed_model)[0]
        #从向量库搜索出top50个相关电影
        retreviex_records = self.tbl.search(embeding).limit(max_suggestions).to_pandas()
        print(retreviex_records)
        #根据搜索结果 获取候选电影的movie_id
        result = self.movie_set.merge(retreviex_records, on='movie_id', how='inner')['movie_id']
        logger.info("candidates"+result)
        candidates = [int(num) for num in result]
        
        return candidates
        
        
    def get_answer(self,query : str):
        embeding = embed_func(query,self.embed_tokenizer,self.embed_model)[0]
        #从向量库搜索出top50个相关电影
        retreviex_records = self.tbl.search(embeding).limit(max_suggestions).to_pandas()
        print(retreviex_records)
        #根据搜索结果 获取候选电影的原始信息
        result = self.movie_set.merge(retreviex_records, on='movie_id', how='inner')
        result=result['name'].str.cat(sep=',')
        print(result)
        input = Prompts.sys_prompt.format_map(
        {
            "user_input": query,
            "movie_list": result
        })
        print('input:',input)
        generate_kwargs = dict(do_sample=False, temperature=0.9, num_beams=4)
        input_ids = self.qwen_tokenizer(input, return_tensors="pt").input_ids.to(device)
        gen_ids = self.qwen_model.generate(input_ids, max_new_tokens=32, **generate_kwargs)
        gen_text = self.qwen_tokenizer.batch_decode(gen_ids, skip_special_tokens=True)
        print('output1:',gen_text)
        
        
        
        return gen_text
    
    def queryTest(self,query:str):
        generate_kwargs = dict(do_sample=False, temperature=0.9, num_beams=4)
        input_ids = self.qwen_tokenizer(query, return_tensors="pt").input_ids.to(device)
        gen_ids = self.qwen_model.generate(input_ids, max_new_tokens=32, **generate_kwargs)
        gen_text = self.qwen_tokenizer.batch_decode(gen_ids, skip_special_tokens=True)
        return gen_text

    def sasrec_predict(self,user_his,candidates):
        if len(candidates)<=5:
            return candidates
        user_his =[int(item) for item in user_his]
        #用sasrec算法获取top5
        logger.info("user_his:"+json.dumps(user_his)+"candidates:"+json.dumps(candidates))
        predictions = -self.sasrec_model.predict(*[np.array(l) for l in [[1], [user_his], candidates]])
        
        predictions = predictions[0] # - for 1st argsort DESC
        predictions = predictions.tolist()
        # 使用 zip() 函数将 candidates 和 predictions 组合成元组的列表
        combined = list(zip(candidates, predictions))

        # 使用 sorted() 函数和 lambda 函数根据评分对元组列表进行降序排序
        sorted_combined = sorted(combined, key=lambda x: x[1], reverse=True)
        sorted_combined=[item[0] for item in sorted_combined if self.movie_name_dict.get(str(item[0])) is not None]
        # 获取评分最高的五个元素
        top_five = sorted_combined[:5]
        #print(top_five)
        logger.info("top_five:"+json.dumps(top_five))
        
        return top_five

def test():
    rag= RAGManager()
    rag.sasrec_predict([1316580, 1291546],[30188270, 30222870, 30194783, 5041543, 26935777, 19933547, 26873275, 25812412, 10796587, 23067452, 10545972, 2213591, 1985752, 1308034, 1441193, 1764825, 1301054, 1296451, 1302476, 1440338, 2128970, 1302936, 3005670, 3100969, 1307038, 5053542, 1302870, 1293602, 1300966, 1302058, 1303559, 5065863, 1305657, 1307463, 1306947, 1303407, 1578669, 1298904, 1449917, 1294104, 1309031, 1419963, 6078900, 30352566, 26856351, 26954641, 4189907, 2988294, 2980609, 3011033]
                       )
    print("finish")
           
def main():
    test()

if __name__ == '__main__':
    main()