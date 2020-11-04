# '''这一部分是用于kaggle提交的'''
import pickle
import riiideducation

env = riiideducation.make_env()
import pandas as pd

questions = pd.read_csv(
    '../input/riiid-test-answer-prediction/questions.csv',
    usecols=['question_id', 'bundle_id', 'part', 'tags'],
    dtype={
        'question_id': 'int16',
        'bundle_id': 'int16',
        'part': 'int8',
        'tags': 'str'
    }
)
questions['part_1'] = questions['part'].map(lambda x: 1 if x == 1 else 0)
questions['part_2'] = questions['part'].map(lambda x: 1 if x == 2 else 0)
questions['part_3'] = questions['part'].map(lambda x: 1 if x == 3 else 0)
questions['part_4'] = questions['part'].map(lambda x: 1 if x == 4 else 0)
questions['part_5'] = questions['part'].map(lambda x: 1 if x == 5 else 0)
questions['part_6'] = questions['part'].map(lambda x: 1 if x == 6 else 0)
questions['part_7'] = questions['part'].map(lambda x: 1 if x == 7 else 0)

questions['tags'] = questions['tags'].fillna('188')

questions['tags_count'] = questions['tags'].map(lambda x: len(str(x).split(' ')))
test_lectures_info__user_f = pd.read_pickle('../input/test-data/test_lectures_info__user_f.pkl.xz', compression='xz')
test_questions_info__user_f = pd.read_pickle('../input/test-data/test_questions_info__user_f.pkl.xz', compression='xz')
test_questions_info__user_part_f = pd.read_pickle('../input/test-data/test_questions_info__user_part_f.pkl.xz',
                                                  compression='xz')
test_questions_info__user_content_f = pd.read_pickle('../input/test-data/test_questions_info__user_content_f.pkl.xz',
                                                     compression='xz')
questions_answered_correctly_feature = pd.read_pickle('../myfeature/questions_answered_correctly_feature.pkl.xz',
                                                      compression='xz')

f = open('feature_columns.txt')
features_columns = eval(f.readline())
f.close()

with open('../input/test-data/lgb.pickle', 'rb') as fr:
    gbm = pickle.load(fr)
    iter_test = env.iter_test()

    for (test_df, sample_prediction_df) in iter_test:
        test_questions = test_df[test_df['content_type_id'] == 0]
        test_questions_info = pd.merge(
            left=test_questions,
            right=questions,
            how='left',
            left_on='content_id',
            right_on='question_id'
        )

        test_questions_info['prior_question_had_explanation'] = test_questions_info[
            'prior_question_had_explanation'].map({'True': 1, 'False': 0}).fillna(-1).astype(np.int8)

        test_questions_info = test_questions_info.merge(test_lectures_info__user_f, on=['user_id'], how='left')

        test_questions_info = test_questions_info.merge(test_questions_info__user_f, on=['user_id'], how='left')
        test_questions_info = test_questions_info.merge(test_questions_info__user_part_f, on=['user_id'], how='left')

        test_questions_info = test_questions_info.merge(test_questions_info__user_content_f, on=['content_id'],
                                                        how='left')
        test_questions_info = test_questions_info.merge(questions_answered_correctly_feature, on=['content_id'],
                                                        how='left')

        X_test = test_questions_info[features_columns]

        test_questions_info['answered_correctly'] = gbm.predict(X_test)

        env.predict(
            test_questions_info.loc[test_questions_info['content_type_id'] == 0, ['row_id', 'answered_correctly']])
