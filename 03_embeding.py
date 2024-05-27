from intel_extension_for_transformers.transformers import AutoModelForCausalLM, BitsAndBytesConfig
from transformers import AutoTokenizer
import pandas as pd
import pyarrow as pa
import torch
from tqdm import tqdm 
import lancedb
from embed import *


root_dir='/study/01_common/resource/llm.Cache/'
model_name='maidalun1020/bce-embedding-base_v1'
data_dir ='./data/douban/moviedata/'
model_name_or_path=root_dir+model_name
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
embed_ready = pd.read_csv(data_dir+'combined.csv')
embed_ready['movie_id'] = embed_ready['movie_id'].astype(str)
combined_flag = pd.read_csv(data_dir+'combined_flag.csv')
combined_flag['movie_id'] = combined_flag['movie_id'].astype(str)

# 转成向量
sentences = embed_ready['combined'].to_numpy()

#加载llm
tokenizer = AutoTokenizer.from_pretrained(model_name_or_path,trust_remote_code=True)
tokenizer.model_max_length=256
#input_ids = tokenizer(prompt, return_tensors="pt").input_ids
'''
#量化版
woq_config = BitsAndBytesConfig(
                                load_in_4bit=True,bnb_4bit_use_double_quant=True,bnb_4bit_quant_type="nf4",
                                bnb_4bit_compute_dtype=torch.float16)
                                
woq_model = AutoModelForCausalLM.from_pretrained(  
                                                    model_name_or_path,
                                                    quantization_config=woq_config,trust_remote_code=True
                                                )                                
'''
#正常版
woq_model = AutoModelForCausalLM.from_pretrained(  
                                                    model_name_or_path,trust_remote_code=True
                                                ) 
uri ="./data/douban/lmdb_bce"
lmdb = lancedb.connect(uri)
schema = pa.schema([pa.field("movie_id", pa.string()),pa.field("vector", pa.list_(pa.float32(), list_size=250002))])#qwen-7b:151936 ;bce:250002
tbl = lmdb.create_table("movies", schema=schema,exist_ok=True)

embeddings = []
print("herere11")
progress_bar = tqdm(total = len(embed_ready),desc = "Processing")
print("herere22")
errCount = 0
# 遍历 DataFrame
toInsert=pd.DataFrame(columns=['movie_id','vector'])
for index,row1 in embed_ready.iterrows() :
    row2 = combined_flag.loc[index]
    if row2['status'] == 1:
        # 跳过状态为 1 的行
        continue
    try:
        id=row1['movie_id']
        sentence = row1['combined']
        embedding = embed_func(sentence,tokenizer,woq_model)[0]
        #embeddings.append(embedding)
        toInsert.loc[len(toInsert)] = [id,embedding]
        combined_flag.at[index,'status'] = 1
        #combined_flag.iloc[index]=row2
        if len(toInsert)>=50:
            tbl.add(toInsert)
            combined_flag.to_csv(data_dir+'combined_flag.csv')
            progress_bar.update(50)
            toInsert=pd.DataFrame(columns=['movie_id','vector'])
        
        
        
        
        
        
    except:
        errCount+=1
        print('err')

print('success errCount:',errCount)
#测试项目
query = "I would love to see more drama and romance movies"
max_suggestions = 10
vector = embed_func(query,tokenizer,woq_model)

retreviex_records = tbl.search(vector).limit(max_suggestions).to_pandas()

retreviex_records.to_csv('result.csv', index=False)
