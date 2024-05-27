import argparse
llm_dir='./llm.Cache/'
movie_data_dir ='./data/douban/moviedata/'

#原始模型
qwen_chat='Qwen/Qwen-7B-Chat'
#finetuning后用来整合输出的模型
model_chat='Qwen/Qwen-7B-Chat-FT'
#finetuning后用来rerank的模型
model_rerank='Qwen/Qwen-7B-Chat-RK'

def str2bool(s):
    if s not in {'false', 'true'}:
        raise ValueError('Not a valid boolean string')
    return s == 'true'
parser = argparse.ArgumentParser()
parser.add_argument('--dataset',type=str)
parser.add_argument('--train_dir', type=str)
parser.add_argument('--batch_size', default=128, type=int)
parser.add_argument('--lr', default=0.001, type=float)
parser.add_argument('--maxlen', default=50, type=int)
parser.add_argument('--hidden_units', default=50, type=int)
parser.add_argument('--num_blocks', default=2, type=int)
parser.add_argument('--num_epochs', default=201, type=int)
parser.add_argument('--num_heads', default=1, type=int)
parser.add_argument('--dropout_rate', default=0.5, type=float)
parser.add_argument('--l2_emb', default=0.0, type=float)
parser.add_argument('--device', default='cpu', type=str)
parser.add_argument('--inference_only', default=False, type=str2bool)
parser.add_argument('--state_dict_path', default='SASRec/douban_default/SASRec.epoch=30.lr=0.001.layer=2.head=1.hidden=50.maxlen=50.pth', type=str)

args = parser.parse_args()
