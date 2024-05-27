import intel_extension_for_pytorch as ipex
from intel_extension_for_transformers.transformers.modeling import AutoModelForCausalLM
from transformers import AutoTokenizer
import torch
from intel_extension_for_transformers.transformers import AutoModelForCausalLM, BitsAndBytesConfig

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
rootDir='/study/01_common/resource/llm.Cache/'



import bitsandbytes as bnb 
def find_all_linear_names(model):
    """
    找出所有全连接层，为所有全连接添加adapter
    """
    cls = bnb.nn.Linear4bit
    lora_module_names = set()
    for name, module in model.named_modules():
        if isinstance(module, cls):
            names = name.split('.')
            lora_module_names.add(names[0] if len(names) == 1 else names[-1])
 
    if 'lm_head' in lora_module_names:  # needed for 16-bit
        lora_module_names.remove('lm_head')
    return list(lora_module_names)

def save4bit(modelName):
    # 在这里对各种后面可能用到的模型进行低比特量化并保存量化后的模型
    #modelName="Qwen/Qwen1.5-7B-Chat"
    #modelName = "Intel/neural-chat-7b-v3-1"
    model_name_or_path = rootDir+modelName
    saved_dir=rootDir+modelName+'-4bit'
    generate_kwargs = dict(do_sample=False, temperature=0.9, num_beams=4)
    prompt = "Once upon a time, a little girl"
    tokenizer = AutoTokenizer.from_pretrained(model_name_or_path,trust_remote_code=True)
    input_ids = tokenizer(prompt, return_tensors="pt").input_ids.to(device)

    woq_config = BitsAndBytesConfig(load_in_4bit=True, bnb_4bit_quant_type="nf4")
    woq_model = AutoModelForCausalLM.from_pretrained(  
                                                        model_name_or_path,
                                                        quantization_config=woq_config,trust_remote_code=True
                                                    )

    # save quant model
    woq_model.save_pretrained(saved_dir)
    print(find_all_linear_names(woq_model))

    gen_ids = woq_model.generate(input_ids, max_new_tokens=32, **generate_kwargs)
    gen_text = tokenizer.batch_decode(gen_ids, skip_special_tokens=True)
    print(gen_text)


def load4bit(modelName,prompt):
    model_name_or_path = rootDir+modelName+'_4bit'
    tokenizer = AutoTokenizer.from_pretrained(model_name_or_path,trust_remote_code=True)
    woq_config = BitsAndBytesConfig(
                                        load_in_4bit=True,bnb_4bit_use_double_quant=True,bnb_4bit_quant_type="nf4",
                                        bnb_4bit_compute_dtype=torch.float16)
                                        
    model = AutoModelForCausalLM.from_pretrained(  
                                                            model_name_or_path,
                                                            quantization_config=woq_config,trust_remote_code=True
                                                        )
    generate_kwargs = dict(do_sample=False, temperature=0.9, num_beams=4)
    input_ids = tokenizer(prompt, return_tensors="pt").input_ids.to(device)
    gen_ids = model.generate(input_ids, max_new_tokens=32, **generate_kwargs)
    gen_text = tokenizer.batch_decode(gen_ids, skip_special_tokens=True)
    print(gen_text)   
    
if __name__ == '__main__':
    #save4bit("Qwen/Qwen-7B-Chat")
    save4bit("maidalun1020/bce-embedding-base_v1")
    #load4bit("Qwen/Qwen-7B-Chat","你是一名专业的电影人,你将根据用户喜欢和不喜欢的电影列表推断出用户的观影喜好,并在给出的50个候选电影中选出5部推荐电影,并给出理由。\n用户:我喜欢盗梦空间、星际穿越、春夏秋冬又一春、小偷家族、霸王别姬，不喜欢上海堡垒、人类消减计划、大白鲨。请给我推荐一些电影。\n候选电影列表:阿尔法,特殊保镖,决赛,舍赫拉查德,钱袋,白塔,收割者,绝命秒杀,塔巴德,第三国度,枪杀,马蹄铁理论,梅纳什,我是杀手女仆,宽恕,夺宝七杀,灼热之夏,马赛克,得心拳手,เทริด,马小兵的夏天,最后的我来也,晚班,妈妈，晚餐吃什么？,棋盘游戏,猫咪情缘,阿信,金钱,阿什利,床伴,同意,白天的星星,赛德克·巴莱,我是大佬,重返十七岁,阿米尔,十七,誓言,孤注一掷,杀手生涯,泰国黑帮,业余赌徒,黑帮大佬中头奖,寻枪,不归单程路,小偷,特殊助手,二十美元,桃色情人,克莱尔,西部来的人,最后武士,重返奥兹国,拳王阿里,蛊惑独行侠,立法者摩西,爱丽丝姨妈,莫卧儿大帝,非洲的愤怒,围歼街头,白奴交易,爱人谋杀 - 电影,太阳泪 - 电影,天敌 - 电影,赌王 - 电影,黑色姐妹帮 - 电影,太阳的女儿 - 电影,我爱马文 - 电影,麦肯纳的黄金 - 电影,马丁·翟述伟 - 电影,呼吸 - 电影,五指 - 电影,乞丐博士,荒野寻踪 - 电影,全款交收 - 电影,摩斯探长：应许之地 - 电影,麦尔斯 - 电影,阿里与尼诺 - 电影,父亲的草原母亲的河 - 电影,战场 - 电影,良心的救赎 - 电影,告别茉莉 - 电影,遗愿清单 - 电影,伊兹的礼物 - 电影,窥见 - 电影,那空村落 - 电影,异闻录之灵瞳 - 电影,沙漠玫瑰 - 电影,烈焰篮球 - 电影,辛巴 - 电影,阿米什的恩典 - 电影,The Money Master - 电影,女拳霸 - 电影,马乌甲 - 电影,莫莉·马圭尔斯 - 电影,新流氓医生 - 电影,警报 - 电影,Kaliya Mardan - 电影,塔努嫁玛努 - 电影,阿尔坎 - 电影,迷幻之城 - 电影,迷幻之城2 - 电影")