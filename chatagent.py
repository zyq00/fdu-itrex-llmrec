import openai
openai.api_key = "EMPTY"
openai.base_url = 'http://127.0.0.1:8000/v1/'
from prompts import Prompts
import env_config
import pandas as pd
from loguru import logger
import json
logger.add("./log/chatbot.log", backtrace=True, diagnose=True)

def rerank(candidates,like,dislike,hints):
    movie_set =  pd.read_csv(env_config.movie_data_dir+'movie_set.csv', encoding='utf-8',on_bad_lines='skip')
    movie_set=movie_set.astype(str)
    movie_dict =movie_set.set_index('name')['movie_id'].to_dict()
    pmt_mng = Prompts()
    msg = pmt_mng.predic_prompt(candidates=candidates,user_like=like,user_dislike=dislike, hints_answer=hints)
    response = openai.chat.completions.create(
        model="./llm.Cache/Qwen/Qwen-7B-Chat-RK-3000",
        #model="/study/01_common/resource/llm.Cache/Intel/neural-chat-7b-v3-3",
        messages=msg
    )
    result = response.choices[0].message.content
    result_array = [item.strip().strip('"') for item in result.split(',')]
    converted_array = [movie_dict[item] for item in result_array if item in movie_dict.keys()]
    int_array = list(map(int, converted_array))
    
    return int_array

def getresponse(message):
    logger.info("input:"+json.dumps(message,indent=2,ensure_ascii=False))
    response = openai.chat.completions.create(
        model="./llm.Cache/Qwen/Qwen-7B-Chat-RK-3000",
        #model="/study/01_common/resource/llm.Cache/Intel/neural-chat-7b-v3-3",
        messages=message
    )
    logger.info("response:"+json.dumps(str(response.choices[0].message.content),indent=2,ensure_ascii=False))
    return response.choices[0].message.content
    
def recommandation(user_history,rec_items):
    #根据用户历史聊天记录，rerank模型输出结果，进行电影推荐
    pmt_mng = Prompts()
    pmt = pmt_mng.recommandation_prompt(user_history=user_history,rec_items=rec_items)
    return getresponse(pmt)


    
def get_like_dislike(messages):
    #msg = {"role": "system", "content": "从聊天记录中找出用户喜欢和不喜欢的电影,忽略不能确定是否喜欢的电影,用json格式输出{like:[],dislike:[]}"}
    messages = [{"role":"user","content":"从聊天记录中找出用户喜欢和不喜欢的电影,忽略不能确定是否喜欢的电影,不能出现聊天记录中没有的电影,用json格式输出{like:[],dislike:[]}"+"以下是用户的输入:"+messages}]
    response = getresponse(messages)
    like=[]
    dislike=[]
    try:
        parsed_data = json.loads(response)
        like = parsed_data['like']
        dislike = parsed_data['dislike']
    except :
        logger.error('json解析失败')
    return like,dislike
            
if __name__ == '__main__':
    response = openai.chat.completions.create(
        model="./llm.Cache/Qwen/Qwen-7B-Chat-RK-3000",
        #model="/study/01_common/resource/llm.Cache/Intel/neural-chat-7b-v3-3",
        messages=[{'role': 'system', 'content': '你是一名专业的电影人,你将根据用户喜欢和不喜欢的电影列表推断出用户的观影喜好,并在给出的51个候选电影中选出5部推荐电影,推荐的电影中不能含有用户喜欢和不喜欢的电影.结果以json列表形式给出,不需要额外的话.'}, {'role': 'user', 'content': '喜欢的电影列表:"青春之杀人者", "鬼太鼓座", "不良少年", "狂怒 - 电影", "花筐 - 电影", "少年时代 - 电影", "死亡解剖 - 电影", "暗杀坂本龙马 - 电影", "水手服与机关枪 - 电影", "超市之女 - 电影", "女税务官", "绿窗艳影", "飞车党", "再见吧！可爱的大地", "美国朋友 - 电影", "阳炎座", "夏之妹", "撒谎者", "小人国的舞会", "脑髓地狱", "樱之森之满开之下 - 电影", "布拉格的大学生", "幸福的馨香", "黑蜥蜴 - 电影", "8万伏特霹雳神龙", "泡吧侦探3 - 电影", "青空娘 - 电影", "家庭私小说 - 电影", "秋津温泉", "祝福", "妻之告白 - 电影", "大爵士乐队", "初恋·地狱篇", "受监护的女人 - 电影", "秋之来临", "双", "流浪记", "卡门归乡 - 电影", "小人物狂想曲 - 电影", "胎儿密猎时刻", "墨东绮谭", "第九日", "玛琳娜的杀戮四段式", "年轻的日子", "万字", "第三度嫌疑人", "新宿小偷日记 - 电影", "写乐", "侦探小说 - 电影", "无赖汉".不喜欢的电影列表:.候选列表:"世界是平的", "沉眠之虎", "假面骑士对修卡", "与鲨同游", "严重伤害 - 电影", "真爱伴鹅行 - 电影", "无序有序 - 电影", "走向幸福 - 电影", "尼斯大冒险 - 电影", "致命情挑 - 电影", "围困", "飞象计划", "血战", "艺术家", "唱歌神探", "美丽的女磨坊主", "美国之窗", "飓风", "神经侠侣", "沙滩上的月亮", "Gackt Live Tour 2003: Jougen no Tsuki - Saishu-Shou", "放学后 - 电影", "搏命擒贼", "英雄豪杰 - 电影", "纵情起舞", "多巴胺", "圣诞不眠夜", "碧血黄沙", "血的失败者", "死亡游戏", "灵动：鬼影实录3 - 电影", "美版星獸戰隊", "镖行天下前传之至尊国宝 - 电影", "镜像人·明日青春", "达摩祖师 - 电影", "杀人广告 - 电影", "决案 - 电影", "长寿商会", "爱神", "巴黎内部", "雷格", "Pulp - Anthology", "萨拉舞皇后", "常隆基", "破窗 - 电影", "这个杀手不太冷 - 电影", "我的男人", "法国女人 - 电影", "计中计", "呷醋大丈夫 - 电影", "千年等一天 - 电影".\nHint: 另一个推荐系统给出的结果为"碧血黄沙", "Gackt Live Tour 2003: Jougen no Tsuki - Saishu-Shou", "镖行天下前传之至尊国宝 - 电影", "无序有序 - 电影", "达摩祖师 - 电影".'}]
    )
    
    movie_set =  pd.read_csv(env_config.movie_data_dir+'movie_set.csv', encoding='utf-8',on_bad_lines='skip')
    movie_set=movie_set.astype(str)
    movie_dict =movie_set.set_index('name')['movie_id'].to_dict()
    result = response.choices[0].message.content
    #result_array = [item.strip('"') for item in result.split(', ')]
    result_array = [item.strip().strip('"') for item in result.split(',')]
    converted_array = [movie_dict[item] for item in result_array if item in movie_dict.keys()]
    int_array = list(map(int, converted_array))
    
    print(int_array)
    print(response.choices[0].message.content)