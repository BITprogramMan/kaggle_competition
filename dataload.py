import pandas as pd
import numpy as np

def dataload():
    dir_path = '../input/riiid-test-answer-prediction/'
    file_train = 'train.csv'
    file_questions = 'questions.csv'
    file_lectures = 'lectures.csv'
    nrows = 100 * 10000

    train = pd.read_csv(
        dir_path + file_train,
        # nrows=nrows,
        usecols=['row_id', 'timestamp', 'user_id', 'content_id',
                 'content_type_id', 'task_container_id', 'answered_correctly',
                 'prior_question_elapsed_time', 'prior_question_had_explanation'],
        dtype={
            'row_id': 'int64',
            'timestamp': 'int64',
            'user_id': 'int32',
            'content_id': 'int16',
            'content_type_id': 'int8',
            'task_container_id': 'int16',
            'answered_correctly': 'int8',
            'prior_question_elapsed_time': 'float32',
            'prior_question_had_explanation': 'str'
        }
    )
    # train.loc[94522438,'row_id']='94522438'
    # train['row_id']=train['row_id'].values.astype(np.int64)

    lectures = pd.read_csv(
        dir_path + file_lectures,
        usecols=['lecture_id', 'tag', 'part', 'type_of'],
        # nrows=nrows,
        dtype={
            'lecture_id': 'int16',
            'tag': 'int16',
            'part': 'int8',
            'type_of': 'str'
        }
    )
    questions = pd.read_csv(
        dir_path + file_questions,
        # nrows=nrows,
        usecols=['question_id', 'bundle_id', 'part', 'tags'],
        dtype={
            'question_id': 'int16',
            'bundle_id': 'int16',
            'part': 'int8',
            'tags': 'str'
        }
    )
    # max_num = 3000
    # train = train.groupby(['user_id']).tail(max_num)
    train['timestamp1']=train.groupby('user_id')['timestamp'].diff().fillna(method='bfill')
    train['prior_question_elapsed_time'].fillna(value=train['prior_question_elapsed_time'].mean(),inplace=True)
    train['timestamp1'].fillna(method='ffill',inplace=True)
    train = train.sort_values(['timestamp'], ascending=True).reset_index(drop=True)
    train['prior_question_had_explanation'] = train['prior_question_had_explanation'].map(
        {'True': 1, 'False': 0}).fillna(0).astype(np.int8)


    lectures['type_of'] = lectures['type_of'].map(
        {'concept': 0, 'intention': 1, 'solving question': 2, 'starter': 3}).astype(np.int8)
    lectures=pd.get_dummies(lectures,columns=['part','type_of'])




    questions['tags'].fillna('188',inplace=True)

    questions['tags_count'] = questions['tags'].map(lambda x: len(str(x).split(' ')))
    # for i in range(189):
    #     questions['tags_' + str(i)] = 0
    #
    # for i in range(len(questions)):
    #     tags=questions['tags'][i].split()
    #     for tag in tags:
    #         questions['tags_'+tag][i]=1

    return train, questions, lectures
