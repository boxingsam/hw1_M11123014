import pandas as pd
import numpy as np
from sklearn import preprocessing
import graphviz
df=pd.read_csv("C:/Users/MB20705/Desktop/data mining/adult.csv",names=["age","workclass","fnlwgt", "education", "education-num","marital-status","occupation","relationship","race","sex","capital-gain","capital-loss","hours-per-week","native-country","salary"])  



#刪除重複和問號的值
df.drop_duplicates(inplace=True)
df = df.replace(" ?", np.nan)
print("重複的值有幾筆",df[pd.isnull(df).any(axis=1)].shape)
print("原始比數：",df.shape)
df.dropna(inplace=True)
print("刪除重複的值：",df.shape)



features=list(df.columns[:14])
df_X=df[features]
outcome=list(df.columns[14:])
df_y=df[outcome]


from sklearn.model_selection import train_test_split



X_train,X_test,y_train,y_test=train_test_split(df_X,df_y,test_size=0.3, random_state=1000)
n=pd.DataFrame(X_test)
n['salary']=y_test

df_X=df_X.drop('education-num', axis=1)

normalize_columns = ['age', 'fnlwgt', 'capital-gain','capital-loss','hours-per-week']
categorical_columns = ['workclass','education','marital-status','occupation','relationship','race','sex','native-country']

df_y=df_y['salary'].map({' <=50K':'0',' >50K':'1'})

#將類別資料轉成 One hot encoding 
def convert_to_onehot(data,columns):
  dummies = pd.get_dummies(data[columns])
  data = data.drop(columns, axis=1)
  data = pd.concat([data, dummies], axis=1)
  return data 
df_X= convert_to_onehot(df_X,categorical_columns)

def normalize(columns):
  scaler = preprocessing.StandardScaler()
  df_X[columns] = scaler.fit_transform(df_X[columns])
normalize(normalize_columns)
#print(df_X)  
#print(df_y)

print("經過one hot encoding：",df_X.shape) 

print("{:=^130s}".format(""))


#經過one hot 所以特徵有變多
xfeatures=list(df_X.columns[:])


from sklearn.model_selection import train_test_split
X,y=df_X.values,df_y.values

test_size=0.3
X_train,X_test,y_train,y_test=train_test_split(X,y,test_size=0.3, random_state=1000)
print(X_train.shape)
print(X_test.shape)

from sklearn import tree

train_list=[]
test_list=[]
for i in range(1,18):
    model= tree.DecisionTreeClassifier(criterion='entropy',max_depth=i)  
    model.fit(X_train, y_train)
    score_test=model.score(X_test, y_test)
    score_train=model.score(X_train, y_train)
    test_list.append(score_test)
    train_list.append(score_train)
result={
    "train":train_list,
    "test":test_list
}
df_result=pd.DataFrame(result)    


import matplotlib.pyplot as plt
plt.plot(train_list,label='train_data')
plt.plot(test_list,label='test_data')

plt.ylabel("Accuracy")
plt.xlabel("Depth_of_tree")
plt.title("Adult_dataset_accuracy")
plt.xlim([0,16])
plt.grid(True)
plt.legend()
plt.show()
print("每個max_depth的準確率：")
print(df_result,"\n最好的準確率：",df_result.max())

print("{:=^130s}".format(""))


#建立模型
from sklearn import tree
model=tree.DecisionTreeClassifier(criterion='entropy',max_depth=6)
model=model.fit(X_train,y_train)


importances = list(model.feature_importances_)
df_importance = pd.DataFrame({'feature':xfeatures,'feature_importance':importances})
df_importance = df_importance.sort_values(by=['feature_importance'],ascending=False).reset_index(drop=True)
print("每個特徵值的重要性：")
print(df_importance.loc[df_importance['feature_importance']!=0])
print("{:=^130s}".format(""))
print('訓練準確度',model.score(X_train, y_train))
print('測試準確度',model.score(X_test, y_test))


#預測，評估模型好壞；使用訓練資料當測試資料
pred=model.predict(X_train)

#輸出混亂矩陣，顯示準確率
from sklearn.metrics import confusion_matrix,classification_report
print("輸出混亂矩陣，顯示準確率：使用訓練資料")
print(confusion_matrix(y_train,pred))
print(classification_report(y_train,pred))


pred=model.predict(X_test)
#輸出混亂矩陣，顯示準確率
from sklearn.metrics import confusion_matrix,classification_report
print("輸出混亂矩陣，顯示準確率：使用測試資料")
print(confusion_matrix(y_test,pred))
print(classification_report(y_test,pred))




#產出Excel(Test data)
n['pre_class'] =pred
y1=n['pre_class'].map({'0':'<=50K','1':'>50K'})
n['pre_class']=y1
n.to_csv('ResultOfTestdata')




# Visualize tree
dot_data = tree.export_graphviz(model, out_file=None,  
  feature_names=xfeatures,                             
  filled=True, rounded=True,  
  special_characters=True)  
graph = graphviz.Source(dot_data)  
graph.format='png'
graph.render('Dtree')

import matplotlib.pyplot as plt#plt用於顯示圖片
import matplotlib.image as mping#mping用於讀取圖片

lena=mping.imread('Dtree.png')#讀取和程式碼處於同一目錄下的lean.png
#此時 lena 就已經是一個np.array了，可以對它進行任意處理
lena.shape#(512,512,3)
plt.imshow(lena)#顯示圖片
plt.axis('off')#不顯示座標軸
plt.show()

plt.savefig('Dtree.png')
from PIL import Image
im=Image.open('Dtree.png')
im.show()





