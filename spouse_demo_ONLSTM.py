# %% [markdown]
# # Detecting spouse mentions in sentences

# %% [markdown]
# In this tutorial, we will see how Snorkel can be used for Information Extraction. We will walk through an example text classification task for information extraction, where we use labeling functions involving keywords and distant supervision.
# ### Classification Task
# <img src="imgs/sentence.jpg" width="700px;" onerror="this.onerror=null; this.src='/doks-theme/assets/images/sentence.jpg';" align="center" style="display: block; margin-left: auto; margin-right: auto;">
#
# We want to classify each __candidate__ or pair of people mentioned in a sentence, as being married at some point or not.
#
# In the above example, our candidate represents the possible relation `(Barack Obama, Michelle Obama)`. As readers, we know this mention is true due to external knowledge and the keyword of `wedding` occuring later in the sentence.
# We begin with some basic setup and data downloading.
#
# %% {"tags": ["md-exclude"]}
# %matplotlib inline

import os
import pandas as pd
# from snorkel.utils import probs_to_preds

import pickle
import numpy as np


if os.path.basename(os.getcwd()) == "snorkel-tutorials":
    os.chdir("alloy")

# %%
from utils import load_data

((df_dev, Y_dev), df_train, (df_test, Y_test)) = load_data()
Y_dev=np.array(Y_dev)
Y_test=np.array(Y_test)



# %% [markdown]
# **Input Data:** `df_dev`, `df_train`, and `df_test` are `Pandas DataFrame` objects, where each row represents a particular __candidate__. For our problem, a candidate consists of a sentence, and two people mentioned in the sentence. The DataFrames contain the fields `sentence`, which refers to the sentence of the candidate, `tokens`, the tokenized form of the sentence, and `person1_word_idx` and `person2_word_idx`, which represent `[start, end]` indices in the tokens at which the first and second person's name appear, respectively.
#
# We also have certain **preprocessed fields**, that we discuss a few cells below.

# %% {"tags": ["md-exclude"]}

# Don't truncate text fields in the display
pd.set_option("display.max_colwidth", 0)

# print("This is train")
# print(df_train.iloc[0])



from preprocessors import get_person_text

#df_train 使用列表
candidate = df_train.iloc[0]
# print(type(candidate))
# print(candidate)
person_names = get_person_text(candidate).person_names

# print("Sentence: ", candidate["sentence"])
# print("m: ", person_names[0])
# print("v: ", person_names[1])
# print(get_len(candidate,candidate["person1_word_idx"][0],candidate["person1_word_idx"][1],"good"))
# # print(get_len(candidate,candidate["person1_word_idx"],"a"))
# print(get_len_twoWords(candidate['text_between'],"of","good"))


#将df写入xls.
# print(type(df_train))
df_train.to_csv("./output/train_data.csv",sep=',',index=False,header=True,encoding='utf_8_sig')
df_dev.to_csv("./output/dev_data.csv",sep=',',index=False,header=True,encoding='utf_8_sig')
df_test.to_csv("./output/test_data.csv",sep=',',index=False,header=True,encoding='utf_8_sig')
# # %% [markdown]
# # Let's look at a candidate in the development set:
#
# # %%
#
#
# #以下分段执行
# from preprocessors import get_person_text
# #
# # candidate = df_dev.loc[2]
# # print(type(candidate))
# # person_names = get_person_text(candidate).person_names
# #
# # print("Sentence: ", candidate["sentence"])
# # print("Person 1: ", person_names[0])
# # print("Person 2: ", person_names[1])
# #
# # # %% [markdown]
# # # ### Preprocessing the Data
# # #
# # # In a real application, there is a lot of data preparation, parsing, and database loading that needs to be completed before we generate candidates and dive into writing labeling functions. Here we've pre-generated candidates in a pandas DataFrame object per split (train,dev,test).
# #
# # # %% [markdown]
# # # ### Labeling Function Helpers
# # #
# # # When writing labeling functions, there are several functions you will use over and over again. In the case of text relation extraction as with this task, common functions include those for fetching text between mentions of the two people in a candidate, examing word windows around person mentions, and so on. We will wrap these functions as `preprocessors`.
# #
# # # %%
# # from snorkel.preprocess import preprocessor
# #
# #
# # @preprocessor()
# # def get_text_between(cand):
# #     """
# #     Returns the text between the two person mentions in the sentence for a candidate
# #     """
# #     start = cand.person1_word_idx[1] + 1
# #     end = cand.person2_word_idx[0]
# #     cand.text_between = " ".join(cand.tokens[start:end])
# #     return cand
# #
# #
# # # %% [markdown]
# # # ### Candidate PreProcessors
# # #
# # # For the purposes of the tutorial, we have three fields (`between_tokens`, `person1_right_tokens`, `person2_right_tokens`) preprocessed in the data, which can be used when creating labeling functions. We also provide the following set of `preprocessor`s for this task in `preprocessors.py`, along with the fields these populate.
# # # * `get_person_text(cand)`: `person_names`
# # # * `get_person_lastnames(cand)`: `person_lastnames`
# # # * `get_left_tokens(cand)`: `person1_left_tokens`, `person2_left_tokens`
# #
# # %%
from preprocessors import *
#

# 1 对 0 错，  -1 弃权
POSITIVE = 1
NEGATIVE = 0
ABSTAIN = -1
#
# # %%

# 开始写标签函数
from snorkel.labeling import labeling_function
#
#检查 两个词之间的与一些设计温度的词的交集
# # Check for the `spouse` words appearing between the person mentions
spouses = {"solvus", "temperature", "γ′", "γ"}
#
#
@labeling_function(resources=dict(spouses=spouses))
def temperature_words(x, spouses):
    # print(type(x))
    # print(x.iloc[0])
    # print(x[0])

    return POSITIVE if len(spouses.intersection(set(x.text_between))) > 0 else ABSTAIN

#
# # %%
# # 考虑 材料名称与括号内属性值的关系。 材料名称右边的第二个或第三个值与属性值有相交部分，则是positive。
#考虑距离 小于4
@labeling_function()
def lf_temperature_words_left_window(x):
    # print(type(x))
    # print(x)
    lenNV = get_len_NameValue(x.person1_word_idx[0], x.person1_word_idx[1], x.person2_word_idx[0], x.person2_word_idx[1])
    if lenNV<=3 and "(" in x.text_between and ")" in x.person2_right_first_tokens:
        return POSITIVE
    else:
        return ABSTAIN
#
#
# # %%
# 考虑 属性值 for 合金名称
# # 考虑一个合金名称和属性值的情况。属性值 for the 合金名称，没有“分别”的情况
@labeling_function()
def lf_for(x):
    #获取name和value的距离
    len=get_len_NameValue(x.person1_word_idx[0],x.person1_word_idx[1],x.person2_word_idx[0],x.person2_word_idx[1])
    #value 和for的距离为1
    # len1=get_len(x.text_between,x.person2_word_idx[0],x.person2_word_idx[1],"for")
    if "for" in x.text_between and "respectively" not in x.person1_right_tokens and len<5 :
        return POSITIVE
    # 考虑两个合金名称和属性值的情况
    # 属性值and属性值 was measured for 合金名称 and合金名称, 分别。
    elif("measured" in x.text_between and "for" in x.text_between and "respectively"  in x.person1_right_tokens and len==6):
        return POSITIVE
    # 属性值and属性值 for 合金名称 and合金名称, 分别。
    elif("respectively"  in x.person1_right_tokens and "for" in x.text_between and len==5):
        return POSITIVE
    #合金名称and合金名称 were 属性值 and 属性值，分别
    elif("respectively"  in x.person2_right_tokens and "were" in x.text_between and len==4):
        return POSITIVE
    #考虑三种合金与属性值的情况
    #合金名称 合金名称 and 合金名称  are about 属性值 and 属性值，分别
    elif("respectively" in x.person2_right_tokens and "are" in x.text_between and len==6):
        return POSITIVE
    else:
        return ABSTAIN
#
#
# # %%
# 考虑一种合金有两种表达方式的写法. 合金名称右边第二个词为还为合金名称
#通过观察，大部分只涉及到一个温度值
# Co–30Ni–10Al–5V–4Ta–2Ti ( 30Ni4Ta2Ti )，既单一关系

#获取所有合金名称集合
nameSet=set()
nameOneSet=set()
for index in range(len(df_train)):
    # print(index)
    cand=df_train.iloc[index]
    # print(cand)
    names = get_person_text(cand).person_names
    name1=names[0]

    names2 = get_person_textOneWord(cand).person_oneWordNames
    name2 = names2[0]
    # print(name1)
    # name2=person_names[1]
    nameSet.add(name1)
    nameOneSet.add(name2)
    # nameSet.append(name2)
# print(nameSet)
nameOneSet.discard(".")
nameOneSet.discard("(")
nameOneSet.discard("alloy")
nameOneSet.discard("and")
nameOneSet.discard(")")
nameOneSet.discard("at")
nameOneSet.discard("that")
nameOneSet.discard("the")
nameOneSet.discard("base")
nameOneSet.discard("alloys")
nameOneSet.discard("ternary")
nameOneSet.discard("between")
nameOneSet.discard("superalloys")
nameOneSet.discard("superalloy")
nameOneSet.discard("with")
nameOneSet.discard("has")
nameOneSet.discard("containing")
nameOneSet.discard("by")
nameOneSet.discard("4")
nameOneSet.discard("8")
nameOneSet.discard("1")
nameOneSet.discard("20")
nameOneSet.discard("heat-treatment")
nameOneSet.discard("would")
nameOneSet.discard("0")

@labeling_function()
def alloy_twoExpress(x):
    if "(" in x.person1_right_first_tokens and ")" in x.person1_right_third_tokens and len(set(x.text_between).intersection(nameSet)) > 0:
        return POSITIVE
    elif("(" in x.person1_left_first_tokens and ")" in x.person1_right_first_tokens and len(set(x.person1_left_second_tokens).intersection(nameSet))>0):
        return POSITIVE
    else:
        return ABSTAIN

#
# # %%
# Check for words that refer to `family` relationships between and to the left of the person mentions
# amily = {
#     "father",
#     "mother",
#     "sister",
#     "brother",
#     "son",
#     "daughter",
#     "grandfather",
#     "grandmother",
#     "uncle",
#     "aunt",
#     "cousin",
# }
# family = family.union({f + "-in-law" for f in family})
# f
#
#检查from to 格式 :
# from 华氏温度(温度) in 合金名称 to....
#温度值
#考虑与特定单词的距离
# @labeling_function()
# def lf_from_to(x):
#     person2LeftList=x.person2_left_tokens
#     person2RightList=x.person2_right_tokens
#     person1LeftList=x.person1_left_tokens
#     #value值左边与from的距离小于5
#     #考虑in的作用
#
#     len1=get_len(person2LeftList,x.person2_word_idx[0],x.person2_word_idx[1],"from")
#     len2=get_len(person2RightList,x.person2_word_idx[0],x.person2_word_idx[1],"in")
#     len3=get_len(person1LeftList,x.person1_word_idx[0],x.person1_word_idx[1],"in")
#     len4=get_len_NameValue(x.person1_word_idx[0],x.person1_word_idx[1],x.person2_word_idx[0],x.person2_word_idx[1])
#     len5 = get_len(person2LeftList, x.person2_word_idx[0], x.person2_word_idx[1], "to")
#     if (len1<=4 and len2<=5 and len3<=3 and len4<=10)or(len4<=7 and len3<3) or (len5<=4 and "in" in x.text_between and len4<10 and len2<=5 and len3<=3):
#         return POSITIVE
#     else:
#         return ABSTAIN

#获取所有合金名称集合
valueSet=set()
valueOneSet=set()
for index in range(len(df_train)):
    # print(index)
    cand=df_train.iloc[index]
    # print(cand)
    values = get_person_text(cand).person_names
    value1=values[1]

    values2 = get_person_textOneWord(cand).person_oneWordNames
    value2 = values2[1]
    # print(name1)
    # name2=person_names[1]
    valueSet.add(value1)
    valueOneSet.add(value2)
    # nameSet.append(name2)

valueOneSet.discard('a')
valueOneSet.discard("the")
valueOneSet.discard("to")
valueOneSet.discard("can")
valueOneSet.discard("of")
valueOneSet.discard("for")
valueOneSet.discard("temperature")
valueOneSet.discard(".")
valueOneSet.discard("(")
valueOneSet.discard(")")
valueOneSet.discard("and")
valueOneSet.discard("is")
valueOneSet.discard("with")
valueOneSet.discard("on")
valueOneSet.discard("1")
valueOneSet.discard("25")





#两种温度表达方式，一种合金，in用法

#和in用法
@labeling_function()
def lf_in(x):
    person2LeftList=x.person2_left_tokens
    person2RightList=x.person2_right_tokens
    person1LeftList=x.person1_left_tokens
    #value值左边与from的距离小于5
    #考虑in的作用

    len1=get_len(person2LeftList,x.person2_word_idx[0],x.person2_word_idx[1],"from")
    len2=get_len(person2RightList,x.person2_word_idx[0],x.person2_word_idx[1],"in")
    len3=get_len(person1LeftList,x.person1_word_idx[0],x.person1_word_idx[1],"in")
    len4=get_len_NameValue(x.person1_word_idx[0],x.person1_word_idx[1],x.person2_word_idx[0],x.person2_word_idx[1])
    len5 = get_len(person2LeftList, x.person2_word_idx[0], x.person2_word_idx[1], "to")

    # 考虑 1255 K ( 982 °C ) in the 0Ti
    # if(len(x.))
    if (len(set(x.person2_right_second_tokens).intersection(valueOneSet)) > 0) and len4 < 8:
        return POSITIVE
    # st1 = x.person2_left_third_tokens[0] + " " + x.person2_left_second_tokens[0]
    # stlist1 = st1.split(":")
    if "in" in x.text_between and len(set(x.person2_left_third_tokens).intersection(valueOneSet)) > 0 and len4 < 8 :
        return POSITIVE
    if (len1 <= 4 and len2 <= 5 and len3 <= 3 and len4 <= 10) or (len4 <= 7 and len3 < 3) or (
            len5 <= 4 and "in" in x.text_between and len4 < 10 and len2 <= 5 and len3 <= 3):
        return POSITIVE

    #考虑1222℃ was observerd in the 合金
    if "observed" in x.person2_right_second_tokens and "in" in x.person2_right_third_tokens and len4<6:
        return POSITIVE
    #考虑 value in 合金
    elif "in" in x.text_between and len4<6:
        return POSITIVE
    else:
        return ABSTAIN




#
#考虑 equal
# 1403 K ( 1130 °C ) , equal to the above Co-7Al-8W-4Ti-1Ta
@labeling_function()
def lf_equal(x):
    lenNV = get_len_NameValue(x.person1_word_idx[0], x.person1_word_idx[1], x.person2_word_idx[0],
                              x.person2_word_idx[1])
    if lenNV <7 and "equal" in x.text_between and "to" in x.person2_right_third_tokens:
        return POSITIVE
    if len(x.person2_right_second_tokens)> 0 and len(x.person2_right_third_tokens)> 0:
        # li = x.person2_right_second_tokens[0] + " " + x.person2_right_third_tokens[0]
        # lists = li.split(":")
        if "equal" in x.text_between and len(set(x.person2_right_second_tokens[0]).intersection(valueSet))> 0 and lenNV < 10:
            return POSITIVE
        else:
            return ABSTAIN
    else:
        return ABSTAIN

# 合金名称 has a solvus temperature value of 温度值
@labeling_function()
def lf_hasTem(x):
    lenNV = get_len_NameValue(x.person1_word_idx[0], x.person1_word_idx[1], x.person2_word_idx[0],
                              x.person2_word_idx[1])
    #考虑 has a temperrate of句式
    if "has" in x.text_between and "a" in x.text_between and "solvus" in x.text_between and "temperature" in  x.text_between and   "of" in  x.text_between and lenNV< 10 :
        return POSITIVE
    #考虑只有has的
    elif "has" in x.person2_left_first_tokens and lenNV<4:
        return POSITIVE
    #has a r solves temperature
    elif "has" in x.text_between and "a" in x.text_between and "γ′" in  x.text_between and  "solvus" in x.text_between and "temperature" in  x.text_between and lenNV< 12:
        return POSITIVE
    elif "has" in x.text_between and "a" in x.text_between and "temperature" in  x.text_between and  "solvus" in x.text_between and len(set(x.text_between).intersection(nameSet))<=0 and  lenNV< 21 :
        return POSITIVE
    if "has" in x.text_between and "a" in x.text_between and "temperature" in  x.text_between and  "solvus" in x.text_between and len(set(x.tokens).intersection(nameSet))==1 and len(set(x.tokens).intersection(valueSet))==1:
        return POSITIVE
    else:
        return ABSTAIN

#   the solvus temperature is  between and ... for
@labeling_function()
def lf_between_and(x):
    lenNV = get_len_NameValue(x.person1_word_idx[0], x.person1_word_idx[1], x.person2_word_idx[0],
                              x.person2_word_idx[1])
    lenv=get_len(candidate,candidate["person2_word_idx"][0],candidate["person2_word_idx"][1],"between")
    lenv2 = get_len(candidate, candidate["person2_word_idx"][0], candidate["person2_word_idx"][1], "and")
    lentwo=get_len_twoWords(x.tokens,"between","and")

    if "between" in x.person2_left_first_tokens and "and" in x.person2_right_first_tokens and lenNV<=5:
        return POSITIVE
    # elif "between" in x.person2_left_third_tokens and "and" in x.person2_left_first_tokens:
    #     return POSITIVE
    # elif "and" in x.person2_left_first_tokens and "between" in x.person2_left_tokens:
    #     return POSITIVE
    # if lenv<=3 or lenv2<=3:
    #     return POSITIVE
    if "between" in x.person2_left_first_tokens and lentwo<=7:
        return POSITIVE
    elif "between" in x.person2_left_tokens and "and" in x.person2_right_second_tokens and lentwo<=7:
        return POSITIVE
    elif "between" in x.person2_left_tokens and "and" in x.person2_left_tokens:
        return POSITIVE
    elif "between" in x.person2_left_tokens and "and" in x.person2_left_first_tokens:
        return POSITIVE
    else:
        return ABSTAIN

#考虑一个候选集与其他候选集的关系问题
@labeling_function()
def lf_oneMatch(x):
    allpd=df_train.append(df_dev)
    everylen=0
    for index, row in allpd.iterrows():
        if x.sentence==row["sentence"]:
            everylen+=1

    if everylen==1:
        return POSITIVE
    else:
        return ABSTAIN



#考虑 合金 were value值
@labeling_function()
def lf_be(x):
    lenNV = get_len_NameValue(x.person1_word_idx[0], x.person1_word_idx[1], x.person2_word_idx[0],
                              x.person2_word_idx[1])

    if "were" in x.text_between and lenNV<=5:
        return POSITIVE
    elif "is" in x.text_between and lenNV<=5:
        return POSITIVE
   # 考虑 0Cr and 4Cr alloys was measured as 965 °C for both alloys
    elif  "was" in x.text_between and "measured" in x.text_between  and "both" in x.person2_right_second_tokens and lenNV<=7:
        return POSITIVE
    #考虑 合金名称 was measured as value值
    # elif "was" in x.text_between and "measured" in x.text_between and "as" in x.text_between and lenNV<6:
    #     return POSITIVE
    else:
        return ABSTAIN














#
#
# # %%
# # Check for `other` relationship words between person mentions
# other = {"boyfriend", "girlfriend", "boss", "employee", "secretary", "co-worker"}
#
#
# @labeling_function(resources=dict(other=other))
# def lf_other_relationship(x, other):
#     return NEGATIVE if len(other.intersection(set(x.between_tokens))) > 0 else ABSTAIN
#
#
# # %% [markdown]
# # ### Distant Supervision Labeling Functions
# #
# # In addition to using factories that encode pattern matching heuristics, we can also write labeling functions that _distantly supervise_ data points. Here, we'll load in a list of known spouse pairs and check to see if the pair of persons in a candidate matches one of these.
# #
# # [**DBpedia**](http://wiki.dbpedia.org/): Our database of known spouses comes from DBpedia, which is a community-driven resource similar to Wikipedia but for curating structured data. We'll use a preprocessed snapshot as our knowledge base for all labeling function development.
# #
# # We can look at some of the example entries from DBPedia and use them in a simple distant supervision labeling function.
# #
# # Make sure `dbpedia.pkl` is in the `spouse/data` directory.
#
# # %%
# with open("data/dbpedia.pkl", "rb") as f:
#     known_spouses = pickle.load(f)
#
# list(known_spouses)[0:5]
#
#
# # %%
# @labeling_function(resources=dict(known_spouses=known_spouses), pre=[get_person_text])
# def lf_distant_supervision(x, known_spouses):
#     p1, p2 = x.person_names
#     if (p1, p2) in known_spouses or (p2, p1) in known_spouses:
#         return POSITIVE
#     else:
#         return ABSTAIN
#
#
# # %%
# from preprocessors import last_name
#
# # Last name pairs for known spouses
# last_names = set(
#     [
#         (last_name(x), last_name(y))
#         for x, y in known_spouses
#         if last_name(x) and last_name(y)
#     ]
# )
#
#
# @labeling_function(resources=dict(last_names=last_names), pre=[get_person_last_names])
# def lf_distant_supervision_last_names(x, last_names):
#     p1_ln, p2_ln = x.person_lastnames
#
#     return (
#         POSITIVE
#         if (p1_ln != p2_ln)
#         and ((p1_ln, p2_ln) in last_names or (p2_ln, p1_ln) in last_names)
#         else ABSTAIN
#     )
#
#
# # %% [markdown]
# # #### Apply Labeling Functions to the Data
# # We create a list of labeling functions and apply them to the data
#
# # %%
from snorkel.labeling import PandasLFApplier
#标签函数集合
lfs = [
    temperature_words,
    lf_temperature_words_left_window,
    lf_for,
    alloy_twoExpress,
    lf_in,
    lf_equal,
    lf_hasTem,
    lf_between_and,
    lf_oneMatch,
    lf_be,
]
applier = PandasLFApplier(lfs)
#
# # %% {"tags": ["md-exclude-output"]}
from snorkel.labeling import LFAnalysis
#
L_dev = applier.apply(df_dev)
L_train = applier.apply(df_train)
L_test=applier.apply(df_test)
# print(L_dev)
#
# # %%

#根据dev开发集分析结果
#打印分析结果
print(LFAnalysis(L_dev, lfs).lf_summary(Y_dev))
#
# # %% [markdown]
# # ### Training the Label Model
# #
# # Now, we'll train a model of the LFs to estimate their weights and combine their outputs. Once the model is trained, we can combine the outputs of the LFs into a single, noise-aware training label set for our extractor.
#
# # %% {"tags": ["md-exclude-output"]}
from snorkel.labeling import LabelModel
# #from metal.label_model import LabelModel
#
label_model = LabelModel(cardinality=2, verbose=True)
label_model.fit(L_train, lr=0.0001, n_epochs=500, log_freq=500, seed=1234)
# label_model.predict()
# label_model.predict_proba()
label_model.save("./model/saved_label_model.pkl")

# %% [markdown]
# ### Label Model Metrics
# Since our dataset is highly unbalanced (91% of the labels are negative), even a trivial baseline that always outputs negative can get a high accuracy. So we evaluate the label model using the F1 score and ROC-AUC rather than accuracy.

# %%
from snorkel.analysis import metric_score
from snorkel.utils import probs_to_preds

probs_test=label_model.predict_proba(L_test)
preds_test=probs_to_preds(probs_test)

col_name=df_test.columns.values.tolist()                   # 将数据框的列名全部提取出来存放在列表里

# for i in col_name:
#     print(i)
# print(type(col_name))
col_name.insert(0,'predict')                      # 在列索引为2的位置插入一列,列名为:city，刚插入时不会有值，整列都是NaN
col_name.insert(1,'Y_test')
df1=df_test.reindex(columns=col_name)              # DataFrame.reindex() 对原行/列索引重新构建索引值

df1['predict']=preds_test  # 给city列赋值
df1['Y_dev']=Y_test  # 给city列赋值
df1.to_csv('./output/result_test.csv',index=False,header=True,encoding='utf_8_sig')


# # %% [markdown]
# # ### Label Model Metrics
# # Since our dataset is highly unbalanced (91% of the labels are negative), even a trivial baseline that always outputs negative can get a high accuracy. So we evaluate the label model using the F1 score and ROC-AUC rather than accuracy.
#
# # %%



print(
    f"Label model f1 score: {metric_score(Y_test, preds_test, probs=probs_test, metric='f1')}"
)
print(
    f"Label model roc-auc: {metric_score(Y_test, preds_test, probs=probs_test, metric='roc_auc')}"
)
print(
    f"Label model precision: {metric_score(Y_test, preds_test, probs=probs_test, metric='precision')}"
)
print(
    f"Label model recall: {metric_score(Y_test, preds_test, probs=probs_test, metric='recall')}"
)

#
# # %% [markdown]
# # ### Part 4: Training our End Extraction Model
# #
# # In this final section of the tutorial, we'll use our noisy training labels to train our end machine learning model. We start by filtering out training data points which did not recieve a label from any LF, as these data points contain no signal.
# #
# # %%

from snorkel.labeling import filter_unlabeled_dataframe
# #
probs_train = label_model.predict_proba(L_train)
df_train_filtered, probs_train_filtered = filter_unlabeled_dataframe(
    X=df_train, y=probs_train, L=L_train
)
#
# # %% [markdown]
# # Next, we train a simple [LSTM](https://en.wikipedia.org/wiki/Long_short-term_memory) network for classifying candidates. `tf_model` contains functions for processing features and building the keras model for training and evaluation.
#
# # %% {"tags": ["md-exclude-output"]}
from tf_model import get_model, get_feature_arrays
from utils import get_n_epochs
# #
X_train = get_feature_arrays(df_train_filtered)
#将训练集词对应数字
batch_size = 64

model = get_model()

model.fit(X_train, probs_train_filtered, batch_size=batch_size, epochs=get_n_epochs())
#
# # %% [markdown]
# # Finally, we evaluate the trained model by measuring its F1 score and ROC_AUC.
#
# # %%
X_test = get_feature_arrays(df_test)
probs_test = model.predict(X_test)
preds_test = probs_to_preds(probs_test)
#
# #将结果写入文件
#
col_name=df_test.columns.tolist()                   # 将数据框的列名全部提取出来存放在列表里
col_name.insert(0,'predict')                      # 在列索引为2的位置插入一列,列名为:city，刚插入时不会有值，整列都是NaN
col_name.insert(1,'Y_test')
df1=df_test.reindex(columns=col_name)              # DataFrame.reindex() 对原行/列索引重新构建索引值
#
df1['predict']=preds_test  # 给city列赋值
df1['Y_test']=Y_test
df1.to_csv('./output/result.csv',index=False)
#
#
#
#
#
#
#
#
#
#
print(
    f"Test F1 when trained with soft labels: {metric_score(Y_test, preds=preds_test, metric='f1')}"
)
print(
    f"Test ROC-AUC when trained with soft labels: {metric_score(Y_test, probs=probs_test, metric='roc_auc')}"
)

# %% [markdown]
# ## Summary
# In this tutorial, we showed how Snorkel can be used for Information Extraction. We demonstrated how to create LFs that leverage keywords and external knowledge bases (distant supervision). Finally, we showed how a model trained using the probabilistic outputs of the Label Model can achieve comparable performance while generalizing to all data points.



#写了三个标签函数，输出的结果格式不正确，然后写不同的函数，提高准确率。

#---------------------------使用XLNet训练模型----------------------------------------#
#
# from snorkel.labeling import filter_unlabeled_dataframe
#
# import tensorflow as tf
# from absl import flags
# FLAGS = flags.FLAGS
#
# if FLAGS.use_tpu:
#     estimator = tf.contrib.tpu.TPUEstimator(
#         use_tpu=FLAGS.use_tpu,
#         model_fn=model_fn,
#         config=run_config,
#         train_batch_size=FLAGS.train_batch_size,
#         predict_batch_size=FLAGS.predict_batch_size,
#         eval_batch_size=FLAGS.eval_batch_size)
# else:
#     estimator = tf.estimator.Estimator(
#         model_fn=model_fn,
#         config=run_config)
#
#
#
#
# probs_train = label_model.predict_proba(L_train)
# df_train_filtered, probs_train_filtered = filter_unlabeled_dataframe(
#     X=df_train, y=probs_train, L=L_train
# )
#
# from tf_model import get_model, get_feature_arrays
# from utils import get_n_epochs
#
# X_train = get_feature_arrays(df_train_filtered)
# model = get_model()
# batch_size = 64
# model.fit(X_train, probs_train_filtered, batch_size=batch_size, epochs=get_n_epochs())
#
# # %% [markdown]
# # Finally, we evaluate the trained model by measuring its F1 score and ROC_AUC.
#
# # %%
# X_test = get_feature_arrays(df_test)
# probs_test = model.predict(X_test)
# preds_test = probs_to_preds(probs_test)
#
# #将结果写入文件
#
# col_name=df_test.columns.tolist()                   # 将数据框的列名全部提取出来存放在列表里
# col_name.insert(0,'predict')                      # 在列索引为2的位置插入一列,列名为:city，刚插入时不会有值，整列都是NaN
# col_name.insert(1,'Y_test')
# df1=df_test.reindex(columns=col_name)              # DataFrame.reindex() 对原行/列索引重新构建索引值
#
# df1['predict']=preds_test  # 给city列赋值
# df1['Y_test']=Y_test
# df1.to_csv('./output/result.csv',index=False)










# print(
#     f"Test F1 when trained with soft labels: {metric_score(Y_test, preds=preds_test, metric='f1')}"
# )
# print(
#     f"Test ROC-AUC when trained with soft labels: {metric_score(Y_test, probs=probs_test, metric='roc_auc')}"
# )

