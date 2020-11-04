import pandas as pd
import numpy as np
import gc
from my_feature import question_base_correct_feature



def get_lecture_basic_features__user(train_lectures_info):
    gb_columns = ['user_id']
    gb_suffixes = 'lecture_' + '_'.join(gb_columns)
    part_lectures_columns = [column for column in train_lectures_info.columns if column.startswith('part')]
    types_of_lectures_columns = [column for column in train_lectures_info.columns if column.startswith('type_of_')]
    user_lecture_stats_part = train_lectures_info.groupby('user_id')[part_lectures_columns + types_of_lectures_columns].sum()
    for column in user_lecture_stats_part.columns:
        bool_column = column + '_boolean'
        user_lecture_stats_part[bool_column] = (user_lecture_stats_part[column] > 0).astype(int)

    agg_func = {
        'lecture_id': [np.size],
        'task_container_id': [lambda x: len(set(x))],
        'tag': [lambda x: len(set(x))]
    }
    columns = [
        gb_suffixes + '_size_lecture_id',
        gb_suffixes + '_unique_task_container_id',
        gb_suffixes + '_unique_tag'
    ]
    train_lectures_info__user_f = train_lectures_info.groupby(gb_columns).agg(agg_func).reset_index()
    train_lectures_info__user_f.columns = gb_columns + columns
    train_lectures_info__user_f=train_lectures_info__user_f.merge(user_lecture_stats_part,on=['user_id'],how='left')
    return train_lectures_info__user_f


def get_lecture_basic_features__user_tag(train_lectures_info):
    gb_columns = ['user_id', 'part']
    gb_suffixes = 'lecture_' + '_'.join(gb_columns)
    agg_func = {
        'lecture_id': [np.size],
        'task_container_id': [lambda x: len(set(x))],
        'tag': [lambda x: len(set(x))],

        # part 展开
        'type_of': [lambda x: len(set(x))],
    }
    columns = [
        gb_suffixes + '_size_lecture_id',
        gb_suffixes + '_unique_task_container_id',
        gb_suffixes + '_unique_tag',
        gb_suffixes + '_unique_type_of'
    ]
    train_lectures_info__user_tag_f = train_lectures_info.groupby(gb_columns).agg(agg_func).reset_index()
    train_lectures_info__user_tag_f.columns = gb_columns + columns
    gb_columns = ['user_id', 'type_of']
    gb_suffixes = 'lecture_' + '_'.join(gb_columns)
    agg_func = {
        'lecture_id': [np.size],
        'task_container_id': [lambda x: len(set(x))],
        'tag': [lambda x: len(set(x))],

        # part 展开
        'part': [lambda x: len(set(x))],
    }
    columns = [
        gb_suffixes + '_size_lecture_id',
        gb_suffixes + '_unique_task_container_id',
        gb_suffixes + '_unique_tag',
        gb_suffixes + '_unique_part'
    ]
    train_lectures_info__user_tag_f = train_lectures_info.groupby(gb_columns).agg(agg_func).reset_index()
    train_lectures_info__user_tag_f.columns = gb_columns + columns
    gb_columns = ['user_id', 'tag']
    gb_suffixes = 'lecture_' + '_'.join(gb_columns)
    agg_func = {
        'lecture_id': [np.size],
        'task_container_id': [lambda x: len(set(x))],
        'type_of': [lambda x: len(set(x))],

        # part 展开
        'part': [lambda x: len(set(x))],
    }
    columns = [
        gb_suffixes + '_size_lecture_id',
        gb_suffixes + '_unique_task_container_id',
        gb_suffixes + '_unique_tag',
        gb_suffixes + '_unique_part'
    ]
    train_lectures_info__user_tag_f = train_lectures_info.groupby(gb_columns).agg(agg_func).reset_index()
    train_lectures_info__user_tag_f.columns = gb_columns + columns
    return train_lectures_info__user_tag_f


# 问答类函数
def get_questions_basic_features__user(train_questions_info):
    gb_columns = ['user_id']
    gb_suffixes = 'question_' + '_'.join(gb_columns)
    agg_func = {
        'answered_correctly': [np.mean, np.sum, np.std],

        'question_id': [np.size],
        'task_container_id': [lambda x: len(set(x))],

        'prior_question_elapsed_time': [np.mean, np.max, np.min],

        'prior_question_had_explanation': [np.mean, np.sum, np.std],

        'bundle_id': [lambda x: len(set(x))],

        # part 展开
        'part': [lambda x: len(set(x))],
        'tags': [lambda x: len(set(x))],
        'timestamp1':[np.mean, np.sum, np.std]
    }
    columns = [
        gb_suffixes + '_answered_correctly_mean',
        gb_suffixes + '_answered_correctly_max',
        gb_suffixes + '_answered_correctly_min',

        gb_suffixes + '_size_question_id',
        gb_suffixes + '_unique_task_container_id',
        gb_suffixes + '_prior_question_elapsed_time_mean',
        gb_suffixes + '_prior_question_elapsed_time_max',
        gb_suffixes + '_prior_question_elapsed_time_min',

        gb_suffixes + '_prior_question_had_explanation_mean',
        gb_suffixes + '_prior_question_had_explanation_max',
        gb_suffixes + '_prior_question_had_explanation_min',

        gb_suffixes + '_unique_bundle_id',
        gb_suffixes + '_unique_part',
        gb_suffixes + '_unique_tags',
        gb_suffixes + '_prior_timestamp1_mean',
        gb_suffixes + '_prior_timestamp1_max',
        gb_suffixes + '_prior_timestamp1_min'
    ]
    train_questions_info__user_f = train_questions_info.groupby(gb_columns).agg(agg_func).reset_index()
    train_questions_info__user_f.columns = gb_columns + columns
    train_questions_info__user_f[gb_suffixes + '_prior_timestamp1_min'].fillna(0,inplace=True)
    train_questions_info__user_f[gb_suffixes + '_prior_question_had_explanation_min'].fillna(0, inplace=True)
    train_questions_info__user_f[gb_suffixes + '_prior_question_elapsed_time_min'].fillna(0, inplace=True)
    train_questions_info__user_f[gb_suffixes + '_answered_correctly_min'].fillna(0, inplace=True)

    return train_questions_info__user_f


def get_questions_basic_features__user_part(train_questions_info):
    gb_columns = ['user_id', 'part']
    gb_suffixes = 'question_' + '_'.join(gb_columns)
    agg_func = {
        'answered_correctly': [np.mean, np.sum, np.std],

        'question_id': [np.size],
        'task_container_id': [lambda x: len(set(x))],

        'prior_question_elapsed_time': [np.mean, np.max, np.min],

        'prior_question_had_explanation': [lambda x: len(set(x))],

        'bundle_id': [lambda x: len(set(x))],

        'tags': [lambda x: len(set(x))],
        'timestamp1': [np.mean, np.sum, np.min]
    }
    columns = [
        gb_suffixes + '_answered_correctly_mean',
        gb_suffixes + '_answered_correctly_max',
        gb_suffixes + '_answered_correctly_min',

        gb_suffixes + '_size_question_id',
        gb_suffixes + '_unique_task_container_id',
        gb_suffixes + '_prior_question_elapsed_time_mean',
        gb_suffixes + '_prior_question_elapsed_time_max',
        gb_suffixes + '_prior_question_elapsed_time_min',

        gb_suffixes + '_unique_prior_question_had_explanation',

        gb_suffixes + '_unique_bundle_id',
        gb_suffixes + '_unique_tags',
        gb_suffixes + '_prior_timestamp1_mean',
        gb_suffixes + '_prior_timestamp1_max',
        gb_suffixes + '_prior_timestamp1_min'
    ]
    train_questions_info__user_part_f = train_questions_info.groupby(gb_columns).agg(agg_func).reset_index()
    train_questions_info__user_part_f.columns = gb_columns + columns
    train_questions_info__user_part_f[gb_suffixes + '_prior_timestamp1_min'].fillna(0, inplace=True)
    train_questions_info__user_part_f[gb_suffixes + '_prior_question_elapsed_time_min'].fillna(0, inplace=True)
    train_questions_info__user_part_f[gb_suffixes + '_answered_correctly_min'].fillna(0, inplace=True)




    return train_questions_info__user_part_f


def get_questions_basic_features__content(train_questions_info):
    gb_columns = ['content_id']
    gb_suffixes = 'question_' + '_'.join(gb_columns)
    agg_func = {
        'answered_correctly': [np.mean, np.sum, np.std],

        'user_id': [np.size],

        'prior_question_elapsed_time': [np.mean, np.max, np.min],

        'prior_question_had_explanation': [lambda x: len(set(x))],
    }
    columns = [
        gb_suffixes + '_answered_correctly_mean',
        gb_suffixes + '_answered_correctly_max',
        gb_suffixes + '_answered_correctly_min',

        gb_suffixes + '_size_user_id',
        gb_suffixes + '_prior_question_elapsed_time_mean',
        gb_suffixes + '_prior_question_elapsed_time_max',
        gb_suffixes + '_prior_question_elapsed_time_min',

        gb_suffixes + '_unique_prior_question_had_explanation',
    ]

    train_questions_info__user_content_f = train_questions_info.groupby(gb_columns).agg(agg_func).reset_index()
    train_questions_info__user_content_f.columns = gb_columns + columns
    train_questions_info__user_content_f[gb_suffixes + '_prior_question_elapsed_time_min']=0
    train_questions_info__user_content_f[gb_suffixes + '_answered_correctly_min'] = 0
    train_questions_info__user_content_f[gb_suffixes + '_prior_question_elapsed_time_min'] = 0

    return train_questions_info__user_content_f


def train_val_split(train, questions, lectures):
    train_lectures = train[train['content_type_id'] == 1]
    train_questions = train[train['content_type_id'] == 0]
    del train
    gc.collect()
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
    question_base_correct_feature(train_questions_info, questions)

    test_lectures_info__user_f = get_lecture_basic_features__user(train_lectures_info)
    # test_lectures_info__user_tag_f = get_lecture_basic_features__user_tag(train_lectures_info)
    test_questions_info__user_f = get_questions_basic_features__user(train_questions_info)
    test_questions_info__user_part_f = get_questions_basic_features__user_part(train_questions_info)
    test_questions_info__user_content_f = get_questions_basic_features__content(train_questions_info)
    test_lectures_info__user_f.to_pickle('myfeature/test_lectures_info__user_f.pkl.xz',compression='xz')
    # test_lectures_info__user_tag_f.to_pickle('test_lectures_info__user_tag_f.pkl.xz',compression='xz')
    test_questions_info__user_f.to_pickle('myfeature/test_questions_info__user_f.pkl.xz',compression='xz')
    test_questions_info__user_part_f.to_pickle('myfeature/test_questions_info__user_part_f.pkl.xz',compression='xz')
    test_questions_info__user_content_f.to_pickle('myfeature/test_questions_info__user_content_f.pkl.xz',compression='xz')

    valid_data = pd.DataFrame()
    for i in range(3):
        # 获取训练标签数据(每个用户的最后一条记录)
        last_records = train_questions_info.drop_duplicates('user_id', keep='last')

        # 获取训练标签以前的数据
        # zip函数将对应的数据组成元组
        # dict函数将元组转化成字典
        # 得到的是user_id:row_id，这里的row_id是每个用户的最后一条数据
        map__last_records__user_row = dict(zip(last_records['user_id'], last_records['row_id']))

        # 如果train_questions_info和train_lectures_info中user_id为某一值
        # 则新建字段filter_row，其值为该user的最后一条数据的row_id
        train_questions_info['filter_row'] = train_questions_info['user_id'].map(map__last_records__user_row)
        train_lectures_info['filter_row'] = train_lectures_info['user_id'].map(map__last_records__user_row)

        # 如果train_questions_info和train_lectures_info中每一条记录，row_id<filter_row
        # 则留下这条记录，其余记录删除
        # 实际上就是删除了每个用户的最后一条记录
        train_questions_info = train_questions_info[train_questions_info['row_id'] < train_questions_info['filter_row']]
        train_lectures_info = train_lectures_info[train_lectures_info['row_id'] < train_lectures_info['filter_row']]

        # 对删除了最后一条记录的新的train_questions_info和train_lectures_info再一次提取特征
        train_lectures_info__user_f = get_lecture_basic_features__user(train_lectures_info)
        # train_lectures_info__user_tag_f = get_lecture_basic_features__user_tag(train_lectures_info)
        train_questions_info__user_f = get_questions_basic_features__user(train_questions_info)
        train_questions_info__user_part_f = get_questions_basic_features__user_part(train_questions_info)
        train_questions_info__user_content_f = get_questions_basic_features__content(train_questions_info)

        # 将提取到的特征与用户的最后一条记录合并
        last_records = last_records.merge(train_lectures_info__user_f, on=['user_id'], how='left')
        # last_records = last_records.merge(train_lectures_info__user_tag_f, on=['user_id','tag'], how='left')
        last_records = last_records.merge(train_questions_info__user_f, on=['user_id'], how='left')
        last_records = last_records.merge(train_questions_info__user_part_f, on=['user_id','part'], how='left')
        last_records = last_records.merge(train_questions_info__user_content_f, on=['content_id'], how='left')

        # 将提取到的特征加入valid_data
        valid_data = valid_data.append(last_records)
        print(len(valid_data))
    train_data = pd.DataFrame()
    for i in range(10):
        # 获取训练标签数据
        last_records = train_questions_info.drop_duplicates('user_id', keep='last')

        # 获取训练标签以前的数据
        map__last_records__user_row = dict(zip(last_records['user_id'], last_records['row_id']))

        train_questions_info['filter_row'] = train_questions_info['user_id'].map(map__last_records__user_row)
        train_lectures_info['filter_row'] = train_lectures_info['user_id'].map(map__last_records__user_row)

        train_questions_info = train_questions_info[train_questions_info['row_id'] < train_questions_info['filter_row']]
        train_lectures_info = train_lectures_info[train_lectures_info['row_id'] < train_lectures_info['filter_row']]

        # 获取特征
        train_lectures_info__user_f = get_lecture_basic_features__user(train_lectures_info)
        # train_lectures_info__user_tag_f = get_lecture_basic_features__user_tag(train_lectures_info)
        train_questions_info__user_f = get_questions_basic_features__user(train_questions_info)
        train_questions_info__user_part_f = get_questions_basic_features__user_part(train_questions_info)
        train_questions_info__user_content_f = get_questions_basic_features__content(train_questions_info)

        last_records = last_records.merge(train_lectures_info__user_f, on=['user_id'], how='left')
        # last_records = last_records.merge(train_lectures_info__user_tag_f, on=['user_id','tag'], how='left')
        last_records = last_records.merge(train_questions_info__user_f, on=['user_id'], how='left')
        last_records = last_records.merge(train_questions_info__user_part_f, on=['user_id','part'], how='left')
        last_records = last_records.merge(train_questions_info__user_content_f, on=['content_id'], how='left')

        # 特征加入训练集
        train_data = train_data.append(last_records)
        print(len(train_data))
        # 删除不需要的字段

    questions_answered_correctly_feature=pd.read_pickle('myfeature/questions_answered_correctly_feature.pkl.xz',compression='xz')
    train_data=train_data.merge(questions_answered_correctly_feature,on='content_id',how='left')
    valid_data=valid_data.merge(questions_answered_correctly_feature,on='content_id',how='left')


    remove_columns = ['user_id', 'row_id', 'content_type_id', 'user_answer', 'answered_correctly', 'filter_row','timestamp1',
                      'question_id','bundle_id_y', 'part_y', 'tags_x','tags_y','part_1_y', 'part_2_y', 'part_3_y', 'part_4_y', 'part_5_y',
                      'part_6_y','part_7_y','tags_count_y']




    train_data.fillna(0,inplace=True)
    valid_data.fillna(0, inplace=True)
    features_columns = [c for c in train_data.columns if c not in remove_columns]
    with open('myfeature/feature_columns.txt','w') as f:
        f.write(str(features_columns))

        # 得到最终验证集

    X_test, y_test = valid_data[features_columns], valid_data['answered_correctly']

        # 得到最终训练集
    X_train, y_train = train_data[features_columns], train_data['answered_correctly']


    return X_train, y_train, X_test, y_test
