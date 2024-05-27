# -*- coding: utf-8 -*-
"""Used to record prompts, will be replaced by configuration"""
import pandas as pd
import env_config

class Prompts:
    
    def __init__(self):
        self.sys_instruction="你是一名专业的电影人,你将根据用户喜欢和不喜欢的电影列表推断出用户的观影喜好,并在给出的<k_num>个候选电影中选出5部推荐电影,推荐的电影中不能含有用户喜欢和不喜欢的电影.结果以json列表形式给出,不需要额外的话."
        movie_set =  pd.read_csv(env_config.movie_data_dir+'movie_set.csv', encoding='utf-8',on_bad_lines='skip')
        movie_set=movie_set.astype(str)
        self.movie_name_dict =movie_set.set_index('movie_id')['name'].to_dict()
        
    def predic_prompt(self,candidates,user_like,user_dislike,hints_answer=[]):
        user_like=user_like[user_like != 0]
        candidates_name = ', '.join([f'"{self.movie_name_dict[str(i)]}"' for i in candidates])
        user_like_name = ', '.join([f'"{self.movie_name_dict[str(i)]}"' for i in user_like])
        user_dislike_name = ', '.join([f'"{self.movie_name_dict[str(i)]}"' for i in user_dislike])
        hints_answer_name = ', '.join([f'"{self.movie_name_dict[str(i)]}"' for i in hints_answer])
        
        instruction=self.sys_instruction.replace("<k_num>",str(len(candidates)))
        input=f"喜欢的电影列表:{user_like_name}.不喜欢的电影列表:{user_dislike_name}.候选列表:{candidates_name}."
        if len(hints_answer)>0:
            input+=f"\nHint: 另一个推荐系统给出的结果为{hints_answer_name}."
        messages=[
          {"role": "system", "content": instruction},
          {"role": "user", "content": input},
          ]
        return messages
        
    def train_prompt(self,candidates,user_like,user_dislike,user_answer,hints_answer=[]):
        candidates_name = ', '.join([f'"{self.movie_name_dict[i]}"' for i in candidates])
        user_like_name = ', '.join([f'"{self.movie_name_dict[i]}"' for i in user_like])
        user_dislike_name = ', '.join([f'"{self.movie_name_dict[i]}"' for i in user_dislike])
        user_answer_name = ', '.join([f'"{self.movie_name_dict[i]}"' for i in user_answer])
        hints_answer_name = ', '.join([f'"{self.movie_name_dict[i]}"' for i in hints_answer])
        
        instruction=self.sys_instruction.replace("<k_num>",str(len(candidates)))
        input=f"喜欢:{user_like_name}.不喜欢:{user_dislike_name}.候选列表:{candidates_name}."
        if len(hints_answer)>0:
            input+=f"\nHint: 另一个推荐系统给出的结果为{hints_answer_name}."
        combined = {
            "instruction": instruction,
            "input": input,
            "output": user_answer_name
        }
        return combined
    
    def recommandation_prompt(self,messages,rec_items):
        rec_movies=', '.join([f'"{self.movie_name_dict[str(i)]}"' for i in rec_items])
        instruction = "你是一名专业的电影人,你将根据用户的发言判断用户的特点然后将推荐系统给出的五部电影推荐给用户,电影列表:"+rec_movies+'.'
        sys_pmt = {"role": "system", "content": instruction}
        last_pmt ={"role": "user", "content":  "向我推荐列表中的五部电影,电影列表:"+rec_movies+'.'}
        messages=[sys_pmt]+[last_pmt]
        return messages

        
