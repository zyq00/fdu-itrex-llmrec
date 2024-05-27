import os
#用于在镜像站下载各种模型 
# 设置环境变量
os.environ['HF_ENDPOINT'] = 'https://hf-mirror.com'
cache_dir = '/study/01_common/resource/llm.Cache'
'''
model_name='sentence-transformers/all-MiniLM-L6-v2'
#model_name='Qwen/Qwen-7B'
#model_name='Qwen/Qwen-32B'
#model_name='Intel/neural-chat-7b-v3-1'
#model_name='maidalun1020/bce-embedding-base_v1'

# 下载模型
os.system('huggingface-cli download --resume-download '+model_name+' --local-dir '+cache_dir+'/'+model_name)
'''
def downloads(model_names):
    #下载多个

    #model_names=['Qwen/Qwen-32B','Qwen/Qwen1.5-32B','Qwen/Qwen1.5-32B-Chat','Qwen/Qwen1.5-14B','Qwen/Qwen1.5-14B-Chat']
    for md_name in model_names:
        os.system('huggingface-cli download --resume-download '+md_name+' --local-dir '+cache_dir+'/'+md_name)
    

if __name__ == '__main__':
    model_names=['Qwen/Qwen1.5-7B-Chat','Qwen/Qwen1.5-14B','Qwen/Qwen1.5-14B-Chat','Qwen/Qwen-32B','Qwen/Qwen1.5-32B','Qwen/Qwen1.5-32B-Chat']
    downloads(model_names)