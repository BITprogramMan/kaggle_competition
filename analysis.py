import matplotlib.pyplot as plt
import pickle
import lightgbm as lgb


with open('myfeature/lgb.pickle', 'rb') as fr:
    gbm = pickle.load(fr)
# plt.figure(figsize=(100,80))
# lgb.plot_importance(gbm,max_num_features=20)
# plt.savefig('feature_importance.eps',dpi=600,format='eps')
# plt.show()
#
importance = list(gbm.feature_importance())
names = list(gbm.feature_name())
importance_dict={}
for key,value in zip(names,importance):
    importance_dict[key]=value
importance_sorted=list(sorted(importance_dict.items(),key=lambda item:item[1], reverse=True))
with open('feature_importance.txt', 'w') as file:
    for index, im in importance_sorted:
        string = index + ', ' + str(im) + '\n'
        file.write(string)







