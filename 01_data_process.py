import pandas as pd
import lancedb
import env_config
import random
import numpy as np
import prompts
import jsonlines,json,math
random.seed(42)
np.random.seed(42)
def indexUser(data_dir):
    users_df = pd.read_csv(data_dir+'users.csv', encoding='utf-8',on_bad_lines='skip')
    users_df['id'] = range(1, len(users_df) + 1)
    users_df.to_csv(data_dir+'users.csv', index=False)

def processData(data_dir):
    # 简化起见 
    # 1.先统计每个电影的平均分低于2分直接认定为烂片直接去掉 电影数:14w->4w
    # 2.评价人数少于50的冷门电影去掉
    # 3.评价电影少于50部的用户的所有影评去掉 评论数:100w->20w 
    # 4.基于123合成movie_set

    # 5.ranking.csv中打分数量少于50部的用户去掉
    # 6.movie_set中不存在的对应movieid的rank去掉
    # 7.大于等于3分认为喜欢，小于3认为不喜欢
    # 8.基于567得到user_set
    # 最终得到两个数据集:
    # 1). movie_set 包含电影的名称、标签、评论等用于rag数据库 
    # 2). user_set 包含用户id、like、dislike用于微调模型rank能力 然后8:2分成train 和val
    # 加载原始数据
    movies_df = pd.read_csv(data_dir+'movies.csv', encoding='utf-8',on_bad_lines='skip')
    users_df = pd.read_csv(data_dir+'users.csv', encoding='utf-8',on_bad_lines='skip')
    ratings_df = pd.read_csv(data_dir+'ratings.csv', encoding='utf-8',on_bad_lines='skip')
    comments_df = pd.read_csv(data_dir+'comments.csv', encoding='utf-8',on_bad_lines='skip')

    print('movies_df:',len(movies_df))
    print('comments_df:',len(comments_df))


    # 确保 movieId 列数据类型一致
    movies_df['movie_id'] = movies_df['movie_id'].astype(str)
    ratings_df['movie_id'] = ratings_df['movie_id'].astype(str)
    comments_df['movie_id'] = comments_df['movie_id'].astype(str)

    #setp1
    # 将 ratings_df 按 movieId 进行分组,并计算平均值
    avg_ratings = ratings_df.groupby('movie_id')['rating'].mean().reset_index()

    # 将 avg_ratings 合并到 movie_df
    merged_df = pd.merge(movies_df, avg_ratings, on='movie_id', how='left')

    # 去掉低分项目
    high_rank_movies = movies_df[merged_df['rating'] >= 3]
    #filtered_df.to_csv('movies_pr.csv', encoding='utf-8')
    print('low_rank_movies:',len(high_rank_movies))

    #setp2
    # 计算每个电影的用户数量
    user_counts = comments_df.groupby('movie_id')['user_md5'].nunique().reset_index()
    user_counts.rename(columns={'user_md5': 'userCount'}, inplace=True)

    # 筛选出用户评论数量少于50的电影
    unpopular_movies = user_counts[user_counts['userCount'] < 50]['movie_id']
    print('unpopular_movies:',len(unpopular_movies))
    #高评分电影中筛掉冷门电影  
    high_rank_popular_movies = high_rank_movies[~high_rank_movies['movie_id'].isin(unpopular_movies)]
    print('high_rank_popular_movies:',len(high_rank_popular_movies))

    #setp3
    # 计算每个 userId 的 movieId 数量
    user_movie_counts = comments_df.groupby('user_md5')['movie_id'].count().reset_index()
    user_movie_counts.rename(columns={'movie_id': 'movieCount'}, inplace=True)

    # 筛选出 movieId 数量大于或等于 50 的 userId
    valid_users = user_movie_counts[user_movie_counts['movieCount'] >= 50]['user_md5']

    # 从 comments 数据中过滤这些 userId
    filtered_comments = comments_df[comments_df['user_md5'].isin(valid_users)]
    print('filtered_comments:',len(filtered_comments))

    # 聚合评论
    comments_grouped = filtered_comments.groupby('movie_id')['content'].agg(lambda x: '|'.join(x.unique())).reset_index()

    #setp4
    # 合并数据
    movie_sets = high_rank_popular_movies.merge(comments_grouped, on='movie_id', how='left')
    movie_set=movie_sets[['movie_id','name','actors','directors','genres','languages','year','storyline','tags','content']]
    movie_set = movie_set.astype(str)
    movie_set.to_csv(data_dir+'movie_set.csv')

    #step5
    # 统计每个 movieid 的数量,并过滤出数量少于 50 的
    movie_counts = ratings_df.groupby('user_md5')['movie_id'].count()
    valid_userIds = movie_counts[movie_counts >= 50].index

    # 从 ranking_df 中过滤出有效的 movieId
    ranking_set_df = ratings_df[ratings_df['user_md5'].isin(valid_userIds)]
    #setp6
    # 合并 ranking_set_df 和 movie_set_df,并过滤出存在于 movie_set_df 的 movieId
    merged_df = pd.merge(ranking_set_df, movie_set[['movie_id']], on='movie_id', how='inner')
    #step7
    # rating大于等于3为喜欢，反之为不喜欢
    # 创建一个空的 DataFrame 来存储最终结果
    result_df = pd.DataFrame(columns=['user_md5', 'like', 'dislike'])
    # 遍历每个 user_md5
    for user_md5, ratings in merged_df.groupby('user_md5'):
        
        # print(ratings)    
        # 将评分大于等于 3 的电影 ID 聚合到 like 列 从大到小排列
        like_movies = ratings[ratings['rating'] >= 3].sort_values(by='rating', ascending=False)['movie_id'].str.cat(sep='|')
        
        # 将评分小于 3 的电影 ID 聚合到 dislike 列 从小到大排列
        dislike_movies = ratings[ratings['rating'] < 3].sort_values(by='rating', ascending=True)['movie_id'].str.cat(sep='|')
        
        
        # 将结果添加到 result_df 中
        result_df = result_df._append({'user_md5': user_md5, 'like': like_movies, 'dislike': dislike_movies}, ignore_index=True)

    # 保存结果到新的 CSV 文件
    #result_df.to_csv('movie_preferences.csv', index=False)
    # 随机打乱数据
    df = result_df.sample(frac=1).reset_index(drop=True)
    #df['user_md5'] = range(1, len(df) + 1)
    
    # 计算分割点
    split_idx = int(0.8 * len(df))

    # 分割数据集
    train_df = df[:split_idx]
    test_df = df[split_idx:]

    # 保存训练集和测试集
    train_df.to_csv(data_dir+'train_set.csv', index=False)
    test_df.to_csv(data_dir+'test_set.csv', index=False)


def mergeMovieInfo(data_dir):
    movie_set = pd.read_csv(data_dir+'movie_set.csv')
    movie_set=movie_set.astype(str)
    # 组合需要的部分
    movie_set['combined'] = ("电影名: "+movie_set['name']+" "+"演员: "+movie_set['actors']+" "
                            +"导演: "+movie_set['directors']+" "+"类型: "+movie_set['genres']+" "
                            +"语言: "+movie_set['languages']+" "+"年份: "+movie_set['year']+" "
                            +"故事情节: "+movie_set['storyline']+" "+"标签: "+movie_set['tags']+" "
                            +"评论: "+movie_set['content'])

    # 去掉不需要的 只保留movieId和combined
    embed_ready=movie_set[['movie_id','combined']]

    embed_ready.to_csv(data_dir+'combined.csv', index=False)
    flag_set = movie_set[['movie_id']]
    #增加一个状态列0表示未处理embeding
    flag_set['status']=0
    flag_set.to_csv(data_dir+'combined_flag.csv', index=False)


def create_train_data(data_dir,sample_num):
    #生成用来finetuning rerank模型的文件
    movie_set = pd.read_csv(data_dir+'movie_set.csv', encoding='utf-8',on_bad_lines='skip')
    movie_set=movie_set.astype(str)
    movie_list = movie_set["movie_id"].values
    train_set = pd.read_csv(data_dir+'train_set.csv', encoding='utf-8',on_bad_lines='skip')
    train_set=train_set.astype(str)
    #movie_name_dict =movie_set.set_index('movie_id')['name'].to_dict()
    
    prompt_list = []
    user_list = train_set['user_md5'].values
    #打乱 随机采样
    random.shuffle(user_list)
    train_list = user_list[:sample_num]
    pmpt = prompts.Prompts()

    for user_md5 in train_list:
        #print(train_set.loc[train_set['user_md5'] == user_md5, 'like'].values[0])
        like_id = train_set.loc[train_set['user_md5'] == user_md5, 'like'].values[0].split('|')
        dislike_id = train_set.loc[train_set['user_md5'] == user_md5, 'dislike'].values[0].split('|')
        if len(dislike_id)==0 or dislike_id[0]=='nan':
            dislike_id=[]
        candidates=[]
        like = []
        dislike = []
        answer= []
        cnt = len(like_id)
        if cnt<10:
            continue
        if cnt >= 55:
            like = like_id[:50]
            answer = like_id[50:55]
        else :
            like = like_id[:cnt-5]
            answer = like_id[cnt-5:cnt]
        dislike = dislike_id[:10]                
        # 从 movie_set 中随机采样 45 个元素到 candidates 数组
        candidates = random.sample([x for x in movie_list if x not in answer], 45)
        candidates+=answer
        print(like)
        print(candidates)
        random.shuffle(candidates)    
            
        train_prompt = pmpt.train_prompt(candidates,like,dislike,answer)
        prompt_list.append(train_prompt)
    #print(prompt_list)
    with open(data_dir+f'alpaca_data_train_{sample_num}.json', 'w', encoding='utf-8') as file:
        json.dump(prompt_list, file,  ensure_ascii=False,indent=4)
        
        
    
    
    
    
if __name__ == '__main__':
    #processData(env_config.movie_data_dir)
    #mergeMovieInfo(env_config.movie_data_dir)
    #create_train_data(env_config.movie_data_dir,3000)
    #indexUser(env_config.movie_data_dir)
    print()