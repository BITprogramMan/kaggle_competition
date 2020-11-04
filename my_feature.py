import pandas as pd
import numpy as np
def question_base_correct_feature(train_questions_info, questions):
    question_part = train_questions_info[['row_id', 'user_id', 'content_id', 'answered_correctly', 'part','timestamp1','tags']]
    question_duplicated = question_part[question_part.duplicated(subset=['user_id', 'content_id'])]
    question_duplicated['delete'] = 1
    question_part1 = pd.merge(question_part, question_duplicated[['row_id', 'delete']], on='row_id', how='left')
    question_part1.drop(question_part1[question_part1['delete'] == 1].index, axis=0, inplace=True)
    questions_content_id_answered_correctly_feature = question_part1.groupby('content_id').agg(
        {'answered_correctly': [np.mean, np.sum, np.std],
         'timestamp1':[np.mean, np.sum, np.std]
         }).reset_index()
    content_id_columns = ['question_id',
                          'q' + '_answered_correctly_mean',
                          'q' + '_answered_correctly_max',
                          'q' + '_answered_correctly_min',
                          'q' + '_timestamp1_correctly_mean',
                          'q' + '_timestamp1_correctly_max',
                          'q' + '_timestamp1_correctly_min']

    questions_content_id_answered_correctly_feature.columns = content_id_columns
    questions_content_id_answered_correctly_feature['question_id'] = \
        questions_content_id_answered_correctly_feature['question_id'].values.astype(np.int16)
    questions = pd.merge(questions, questions_content_id_answered_correctly_feature, on='question_id', how='left')
    questions['q_answered_correctly_min'].fillna(0,inplace=True)
    questions['q_timestamp1_correctly_min'].fillna(0,inplace=True)
    questions_part_answered_correctly_feature = question_part1.groupby('part').agg(
        {'answered_correctly': [np.mean, np.sum, np.std],
         'timestamp1':[np.mean, np.sum, np.std]}).reset_index()
    part_columns = ['part',
                    'question_part_mean',
                    'question_part_max',
                    'question_part_min',
                    'question_timestamp1_mean',
                    'question_timestamp1_max',
                    'question_timestamp1_min']
    questions_part_answered_correctly_feature.columns = part_columns
    questions = pd.merge(questions, questions_part_answered_correctly_feature, on='part', how='left')
    questions['content_id']=questions['question_id']
    questions.drop('question_id',axis=1,inplace=True)




    # for i in range(189):
    #     questions_tags_answered_correctly_feature = question_part1.groupby('tags_' + str(i)).agg(
    #         {'answered_correctly': [np.mean, np.sum, np.std],
    #          'timestamp1': [np.mean, np.sum, np.std]
    #          }).reset_index()
    #     tag_columns = ['tags_' + str(i),
    #                     'question_part_'+'tags_' + str(i)+'_mean',
    #                     'question_part_'+'tags_' + str(i)+'_max',
    #                     'question_part_'+'tags_' + str(i)+'_min',
    #                     'question_timestamp1_'+'tags_' + str(i)+'_mean',
    #                     'question_timestamp1_'+'tags_' + str(i)+'_max',
    #                     'question_timestamp1_'+'tags_' + str(i)+'_min']
    #     questions_tags_answered_correctly_feature.columns = tag_columns
    #     questions = pd.merge(questions, questions_tags_answered_correctly_feature, on='tags_' + str(i), how='left')

    questions.to_pickle('myfeature/questions_answered_correctly_feature.pkl.xz', compression='xz')
    print(questions.shape)









