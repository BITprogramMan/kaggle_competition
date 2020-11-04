
import pandas as pd
import numpy as np
import lightgbm as lgb
from sklearn.metrics import roc_auc_score
from sklearn.model_selection import train_test_split
import gc



dir_path = '../input/riiid-test-answer-prediction/'
file_train = 'train.csv'
file_questions = 'questions.csv'
file_lectures = 'lectures.csv'


nrows = 100 * 10000


# 加载用户数据，其中剔除user_answer，因为我们只需要知道一个用户是否回答正确就可以了，不需要知道具体的答案
# 使用dtype表示读取进来的数据类型，可以防止类型错误以及减少内存使用
train = pd.read_csv(
                    dir_path + file_train, 
                    nrows=nrows, 
                    usecols=['row_id', 'timestamp', 'user_id', 'content_id', 
                             'content_type_id', 'task_container_id', 'answered_correctly',
                            'prior_question_elapsed_time','prior_question_had_explanation'],
                    dtype={
                            'row_id': 'int64',
                            'timestamp': 'int64',
                            'user_id': 'int32',
                            'content_id': 'int16',
                            'content_type_id': 'int8',
                            'task_container_id': 'int8',
                            'answered_correctly': 'int8',
                            'prior_question_elapsed_time': 'float32',
                            'prior_question_had_explanation': 'str'
                        }
                   )

lectures = pd.read_csv(
                       dir_path + file_lectures, 
                       usecols=['lecture_id','tag','part','type_of'], 
                       nrows=nrows,
                       dtype={
                           'lecture_id': 'int16',
                           'tag': 'int16',
                           'part': 'int8',
                           'type_of': 'str'
                       }
                    )


# In[97]:


# 加载questions数据
# 去掉question answer，这个特征没有用处
questions = pd.read_csv(
                        dir_path + file_questions, 
                        nrows=nrows,
                        usecols=['question_id','bundle_id','part','tags'], 
                        dtype={
                           'question_id': 'int16',
                           'bundle_id': 'int16',
                           'part': 'int8',
                           'tags': 'str'
                       }
                    )


# In[98]:


# 数据处理
# 首先对train中的prior_question_had_explanation进行处理，因为它是bool类型，我们使用0和1代替
train['prior_question_had_explanation'] = train['prior_question_had_explanation'].map({'True':1,'False':0}).fillna(-1).astype(np.int8)

# 同样，对lectures的type_of字段用整数代替
lectures['type_of'] = lectures['type_of'].map({'concept':0, 'intention':1, 'solving question':2, 'starter':3}).fillna(-1).astype(np.int8)

# 因为questions中的tags一条记录包含多个，我们对其进行分割，并得到其长度(将tags的长度作为一个特征)
# 感觉这里tags的长度作为特征不太合适，之后改
questions['tags'] = questions['tags'].map(lambda x:len(str(x).split(' ')))


# In[99]:


# 压缩内存
# 这里我觉得应该先count然后sort_values，最后选择记录数较多的user_id
# 即剔除记录数较少的user，剩下的才能更好的用于学习
max_num = 3000
train = train.groupby(['user_id']).tail(max_num)


# In[100]:


# 切分数据
# 分成lectures和questions
train_lectures = train[train['content_type_id']==1]
train_questions = train[train['content_type_id']==0]
del train
gc.collect()


# In[101]:


# 关联数据
# 将train_lectures中的用户数据与lectures中的课程数据关联起来
train_lectures_info = pd.merge(
        left=train_lectures,
        right=lectures,
        how='left',
        left_on='content_id',
        right_on='lecture_id'
        )

# 将train_questions中的用户数据与questions中的课程数据关联起来
train_questions_info = pd.merge(
        left=train_questions,
        right=questions,
        how='left',
        left_on='content_id',
        right_on='question_id'
        )

del train_lectures
del train_questions
gc.collect()


# In[102]:


# 自定义特征提取函数
# 文献课程类函数
def get_lecture_basic_features__user(train_lectures_info):
    gb_columns = ['user_id']
    gb_suffixes = 'lecture_'+'_'.join(gb_columns)
    
    agg_func = {
        'lecture_id': [np.size],
        'task_container_id': [lambda x: len(set(x))],
        'tag': [lambda x: len(set(x))],

        # part 展开
        'part': [lambda x: len(set(x))],

        # type_of 展开
        'type_of': [lambda x: len(set(x))],
    }
    columns = [
           gb_suffixes+'_size_lecture_id', 
           gb_suffixes+'_unique_task_container_id',
           gb_suffixes+'_unique_tag',
           gb_suffixes+'_unique_part',
           gb_suffixes+'_unique_type_of'
          ]  
    train_lectures_info__user_f = train_lectures_info.groupby(gb_columns).agg(agg_func).reset_index()
    
    train_lectures_info__user_f.columns = gb_columns + columns
    return train_lectures_info__user_f

def get_lecture_basic_features__user_tag(train_lectures_info):
    gb_columns = ['user_id','tag']
    gb_suffixes = 'lecture_'+'_'.join(gb_columns)
    agg_func = {
        'lecture_id': [np.size],
        'task_container_id': [lambda x: len(set(x))],
        'tag': [lambda x: len(set(x))],

        # part 展开
        'part': [lambda x: len(set(x))],
    }
    columns = [
               gb_suffixes+'_size_lecture_id', 
               gb_suffixes+'_unique_task_container_id',
               gb_suffixes+'_unique_tag',
               gb_suffixes+'_unique_part'
              ]    
    train_lectures_info__user_tag_f = train_lectures_info.groupby(gb_columns).agg(agg_func).reset_index()
    train_lectures_info__user_tag_f.columns = gb_columns + columns    
    return train_lectures_info__user_tag_f

# 问答类函数
def get_questions_basic_features__user(train_questions_info):
    gb_columns = ['user_id']
    gb_suffixes = 'question_'+'_'.join(gb_columns)
    agg_func = {
        'answered_correctly': [np.mean,np.sum,np.std],

        'question_id': [np.size],
        'task_container_id': [lambda x: len(set(x))],

        'prior_question_elapsed_time': [np.mean,np.max,np.min],

        'prior_question_had_explanation': [lambda x: len(set(x))],

        'bundle_id': [lambda x: len(set(x))],

        # part 展开
        'part': [lambda x: len(set(x))],
        'tags': [lambda x: len(set(x))],
    }
    columns = [
               gb_suffixes+'_answered_correctly_mean',
               gb_suffixes+'_answered_correctly_max',
               gb_suffixes+'_answered_correctly_min',

               gb_suffixes+'_size_question_id', 
               gb_suffixes+'_unique_task_container_id',
               gb_suffixes+'_prior_question_elapsed_time_mean',
               gb_suffixes+'_prior_question_elapsed_time_max',
               gb_suffixes+'_prior_question_elapsed_time_min',

               gb_suffixes+'_unique_prior_question_had_explanation',

               gb_suffixes+'_unique_bundle_id',
               gb_suffixes+'_unique_part',
               gb_suffixes+'_unique_tags',
              ]
    train_questions_info__user_f = train_questions_info.groupby(gb_columns).agg(agg_func).reset_index()
    train_questions_info__user_f.columns = gb_columns + columns    

    return train_questions_info__user_f

def get_questions_basic_features__user_part(train_questions_info):
    gb_columns = ['user_id','part']
    gb_suffixes = 'question_'+'_'.join(gb_columns)
    agg_func = {
        'answered_correctly': [np.mean,np.sum,np.std],

        'question_id': [np.size],
        'task_container_id': [lambda x: len(set(x))],

        'prior_question_elapsed_time': [np.mean,np.max,np.min],

        'prior_question_had_explanation': [lambda x: len(set(x))],

        'bundle_id': [lambda x: len(set(x))],

        # part 展开
        'part': [lambda x: len(set(x))],
        'tags': [lambda x: len(set(x))],
    }
    columns = [
               gb_suffixes+'_answered_correctly_mean',
               gb_suffixes+'_answered_correctly_max',
               gb_suffixes+'_answered_correctly_min',

               gb_suffixes+'_size_question_id', 
               gb_suffixes+'_unique_task_container_id',
               gb_suffixes+'_prior_question_elapsed_time_mean',
               gb_suffixes+'_prior_question_elapsed_time_max',
               gb_suffixes+'_prior_question_elapsed_time_min',

               gb_suffixes+'_unique_prior_question_had_explanation',

               gb_suffixes+'_unique_bundle_id',
               gb_suffixes+'_unique_part',
               gb_suffixes+'_unique_tags',
              ]    
    train_questions_info__user_part_f = train_questions_info.groupby(gb_columns).agg(agg_func).reset_index()
    train_questions_info__user_part_f.columns = gb_columns + columns    

    return train_questions_info__user_part_f

def get_questions_basic_features__content(train_questions_info):
    gb_columns = ['content_id']
    gb_suffixes = 'question_'+'_'.join(gb_columns)
    agg_func = {
        'answered_correctly': [np.mean,np.sum,np.std],

        'user_id': [np.size],

        'prior_question_elapsed_time': [np.mean,np.max,np.min],

        'prior_question_had_explanation': [lambda x: len(set(x))],
    }
    columns = [
               gb_suffixes+'_answered_correctly_mean',
               gb_suffixes+'_answered_correctly_max',
               gb_suffixes+'_answered_correctly_min',

               gb_suffixes+'_size_user_id', 
               gb_suffixes+'_prior_question_elapsed_time_mean',
               gb_suffixes+'_prior_question_elapsed_time_max',
               gb_suffixes+'_prior_question_elapsed_time_min',

               gb_suffixes+'_unique_prior_question_had_explanation',
              ]    
    
    train_questions_info__user_content_f = train_questions_info.groupby(gb_columns).agg(agg_func).reset_index()
    train_questions_info__user_content_f.columns = gb_columns + columns
    
    return train_questions_info__user_content_f


# In[103]:


# 特征提取
test_lectures_info__user_f = get_lecture_basic_features__user(train_lectures_info)
# test_lectures_info__user_tag_f = get_lecture_basic_features__user_tag(train_lectures_info)
test_questions_info__user_f = get_questions_basic_features__user(train_questions_info)
# test_questions_info__user_part_f = get_questions_basic_features__user_part(train_questions_info)
test_questions_info__user_content_f = get_questions_basic_features__content(train_questions_info)


# In[104]:


# 验证数据
valid_data = pd.DataFrame()


'''
这里在做的事是：
比如一个 用户有5条数据，我们将第5条选取出来，并根据前4条数据按照user_id或者content_id进行分组
分组后提取特征，将这些特征认为是第5条记录所对应的特征
对上面的过程进行重复，于是可以得到从第五条记录的特征到第一条记录的特征
当然，如果记录太少，得到的特征不能用来训练模型
'''
for i in range(3):
    
    # 获取训练标签数据(每个用户的最后一条记录)
    last_records = train_questions_info.drop_duplicates('user_id', keep='last')
    
    # 获取训练标签以前的数据
    # zip函数将对应的数据组成元组
    # dict函数将元组转化成字典
    # 得到的是user_id:row_id，这里的row_id是每个用户的最后一条数据
    map__last_records__user_row = dict(zip(last_records['user_id'],last_records['row_id']))
    
    # 如果train_questions_info和train_lectures_info中user_id为某一值
    # 则新建字段filter_row，其值为该user的最后一条数据的row_id
    train_questions_info['filter_row'] = train_questions_info['user_id'].map(map__last_records__user_row)
    train_lectures_info['filter_row'] = train_lectures_info['user_id'].map(map__last_records__user_row)
    
    # 如果train_questions_info和train_lectures_info中每一条记录，row_id<filter_row
    # 则留下这条记录，其余记录删除
    # 实际上就是删除了每个用户的最后一条记录
    train_questions_info = train_questions_info[train_questions_info['row_id']<train_questions_info['filter_row']]
    train_lectures_info = train_lectures_info[train_lectures_info['row_id']<train_lectures_info['filter_row']]
    
    # 对删除了最后一条记录的新的train_questions_info和train_lectures_info再一次提取特征
    train_lectures_info__user_f = get_lecture_basic_features__user(train_lectures_info)
    # train_lectures_info__user_tag_f = get_lecture_basic_features__user_tag(train_lectures_info)
    train_questions_info__user_f = get_questions_basic_features__user(train_questions_info)
    # train_questions_info__user_part_f = get_questions_basic_features__user_part(train_questions_info)
    train_questions_info__user_content_f = get_questions_basic_features__content(train_questions_info)
    
    # 将提取到的特征与用户的最后一条记录合并
    last_records = last_records.merge(train_lectures_info__user_f,on=['user_id'],how='left')
    last_records = last_records.merge(train_questions_info__user_f,on=['user_id'],how='left')
    last_records = last_records.merge(train_questions_info__user_content_f,on=['content_id'],how='left')
    
    # 将提取到的特征加入valid_data
    valid_data = valid_data.append(last_records)
    print(len(valid_data))


# In[105]:


# 训练数据
train_data = pd.DataFrame()


'''
对训练数据也进行相同的特征提取
'''
for i in range(10):
    
    # 获取训练标签数据
    last_records = train_questions_info.drop_duplicates('user_id', keep='last')
    
    # 获取训练标签以前的数据
    map__last_records__user_row = dict(zip(last_records['user_id'],last_records['row_id']))
    
    train_questions_info['filter_row'] = train_questions_info['user_id'].map(map__last_records__user_row)
    train_lectures_info['filter_row'] = train_lectures_info['user_id'].map(map__last_records__user_row)

    train_questions_info = train_questions_info[train_questions_info['row_id']<train_questions_info['filter_row']]
    train_lectures_info = train_lectures_info[train_lectures_info['row_id']<train_lectures_info['filter_row']]
    
    # 获取特征
    train_lectures_info__user_f = get_lecture_basic_features__user(train_lectures_info)
    # train_lectures_info__user_tag_f = get_lecture_basic_features__user_tag(train_lectures_info)
    train_questions_info__user_f = get_questions_basic_features__user(train_questions_info)
    # train_questions_info__user_part_f = get_questions_basic_features__user_part(train_questions_info)
    train_questions_info__user_content_f = get_questions_basic_features__content(train_questions_info)

    last_records = last_records.merge(train_lectures_info__user_f,on=['user_id'],how='left')
    last_records = last_records.merge(train_questions_info__user_f,on=['user_id'],how='left')
    last_records = last_records.merge(train_questions_info__user_content_f,on=['content_id'],how='left')
    
    # 特征加入训练集
    train_data = train_data.append(last_records)
    print(len(train_data))


# In[106]:


# 删除不需要的字段
remove_columns = ['user_id','row_id','content_type_id','user_answer','answered_correctly','filter_row']
features_columns = [c for c in train_data.columns if c not in remove_columns]

# 得到最终验证集
X_test, y_test = valid_data[features_columns].values, valid_data['answered_correctly'].values

# 得到最终训练集
X_train, y_train = train_data[features_columns].values, train_data['answered_correctly'].values

# 设置lgb模型训练参数
params = {
    'task': 'train',
    'boosting_type': 'gbdt',
    'objective': 'binary',
    'metric': 'auc',
    'num_leaves': 9,
    'learning_rate': 0.3,
    'feature_fraction_seed': 2,
    'feature_fraction': 0.9,
    'bagging_fraction': 0.8,
    'bagging_freq': 5,
    'min_data': 20,
    'min_hessian': 1,
    'verbose': -1,
    'silent': 0
    }

# 设置lgb训练参数
lgb_train = lgb.Dataset(X_train, y_train)

# 设置lgb模型测试参数
lgb_eval = lgb.Dataset(X_test, y_test, reference=lgb_train)

# 开始训练和测试
# lgb_train就是输入之前设置的训练集
# valid_sets就是输入之前的验证集
gbm = lgb.train(
            params,
            lgb_train,
            num_boost_round=10000,
            valid_sets=lgb_eval,
            early_stopping_rounds=20
            )


# In[ ]:


'''这一部分是用于kaggle提交的'''
# import pickle
# import riiideducation
# env = riiideducation.make_env()
# with open('gbm.pickle', 'rb') as fr:
#     gbm = pickle.load(fr)
#     iter_test = env.iter_test()
#
#     for (test_df, sample_prediction_df) in iter_test:
#
#         test_questions = test_df[test_df['content_type_id']==0]
#         test_questions_info = pd.merge(
#                 left=test_questions,
#                 right=questions,
#                 how='left',
#                 left_on='content_id',
#                 right_on='question_id'
#                 )
#
#         test_questions_info['prior_question_had_explanation'] = test_questions_info['prior_question_had_explanation'].map({'True':1,'False':0}).fillna(-1).astype(np.int8)
#
#         test_questions_info = test_questions_info.merge(test_lectures_info__user_f,on=['user_id'],how='left')
#         test_questions_info = test_questions_info.merge(test_questions_info__user_f,on=['user_id'],how='left')
#         test_questions_info = test_questions_info.merge(test_questions_info__user_content_f,on=['content_id'],how='left')
#
#         # 修改
#         #remove_columns = ['user_id','row_id','content_type_id','user_answer','answered_correctly','filter_row']
#         #features_columns = [c for c in train_data.columns if c not in remove_columns]
#
#
#         X_test = test_questions_info[features_columns].values
#
#         test_questions_info['answered_correctly'] =  gbm.predict(X_test)
#
#         env.predict(test_questions_info.loc[test_questions_info['content_type_id'] == 0, ['row_id', 'answered_correctly']])

