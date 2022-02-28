# +
from typing import Optional

from snorkel.preprocess import preprocessor
from snorkel.types import DataPoint

# -

@preprocessor()
def get_person_text(cand: DataPoint) -> DataPoint:
    """
    Returns the text for the two person mentions in candidate
    """
    person_names = []
    # person_oneWordNames = []
    for index in [1, 2]:
        field_name = "person{}_word_idx".format(index)
        start = cand[field_name][0]
        end = cand[field_name][1] + 1
        person_names.append(" ".join(cand["tokens"][start:end]))
        # person_oneWordNames.append(" ".join(cand["tokens"][start]))
    cand.person_names = person_names
    # cand.person_oneWordNames=person_oneWordNames
    return cand

@preprocessor()
def get_person_textOneWord(cand: DataPoint) -> DataPoint:
    """
    Returns the text for the two person mentions in candidate
    """
    # person_names = []
    person_oneWordNames = []
    for index in [1, 2]:
        field_name = "person{}_word_idx".format(index)
        start = cand[field_name][0]
        # end = cand[field_name][1] + 1
        # person_names.append(" ".join(cand["tokens"][start:end]))
        # print(type(cand["tokens"]))
        # print(len(cand["tokens"]))
        # print(start)
        # print(end)
        person_oneWordNames.append(" ".join(cand["tokens"][start:start+1]))
    # cand.person_names = person_names
    cand.person_oneWordNames=person_oneWordNames
    return cand


@preprocessor()
def get_person_last_names(cand: DataPoint) -> DataPoint:
    """
    Returns the last names for the two person mentions in candidate
    """
    cand = get_person_text(cand)
    person1_name, person2_name = cand.person_names
    person1_lastname = (
        person1_name.split(" ")[-1] if len(person1_name.split(" ")) > 1 else None
    )
    person2_lastname = (
        person2_name.split(" ")[-1] if len(person2_name.split(" ")) > 1 else None
    )
    cand.person_lastnames = [person1_lastname, person2_lastname]
    return cand


@preprocessor()
def get_text_between(cand: DataPoint) -> DataPoint:
    """
    Returns the text between the two person mentions in the sentence
    """
    # print(cand)
    # print(type(cand))
    # print(cand.person1_word_idx[1])
    # print(cand.person2_word_idx[0])
    if cand.person1_word_idx[1]<=cand.person2_word_idx[0]:
        start = cand.person1_word_idx[1] + 1
        end = cand.person2_word_idx[0]
        #cand.text_between = " ".join(cand.tokens[start:end])
        cand.text_between = cand.tokens[start:end]
    else:
        start=cand.person2_word_idx[1]+1
        end=cand.person1_word_idx[0]
        #cand.text_between=" ".join(cand.tokens[start:end])
        cand.text_between = cand.tokens[start:end]


    # print(cand.text_between)
    return cand


@preprocessor()
def get_left_tokens(cand: DataPoint) -> DataPoint:
    """
    Returns tokens in the length 10 window to the left of the person mentions
    """
    # TODO: need to pass window as input params
    window = 10

    end = cand.person1_word_idx[0]
    cand.person1_left_tokens = cand.tokens[0:end][-1 - window : -1]

    end = cand.person2_word_idx[0]
    cand.person2_left_tokens = cand.tokens[0:end][-1 - window : -1]
    return cand

@preprocessor()
def get_right_tokens(cand: DataPoint) -> DataPoint:
    """
    Returns tokens in the length 10 window to the left of the person mentions
    """
    # TODO: need to pass window as input params
    window = 10

    start = cand.person1_word_idx[1]
    cand.person1_right_tokens = cand.tokens[start+1:start+window+1]

    end = cand.person2_word_idx[1]
    cand.person2_right_tokens = cand.tokens[start+1:start+window+1]
    return cand


@preprocessor()
def get_right_first_tokens(cand: DataPoint) -> DataPoint:
    """
    Returns tokens in the length 1 window to the right of the person mentions
    """
    # TODO: need to pass window as input params
    window = 1

    end = cand.person1_word_idx[1]
    cand.person1_right_first_tokens = cand.tokens[end+1:end+window+1]

    end = cand.person2_word_idx[1]
    cand.person2_right_first_tokens = cand.tokens[end+1:end+window+1]

    # print(cand.person1_right_one_tokens)
    return cand

@preprocessor()
def get_right_second_tokens(cand: DataPoint) -> DataPoint:
    """
    Returns tokens in the length 1 window to the right of the person mentions
    """
    # TODO: need to pass window as input params
    window = 2

    end = cand.person1_word_idx[1]
    cand.person1_right_second_tokens = cand.tokens[end+1:end+window+1][-1::]

    end = cand.person2_word_idx[1]
    cand.person2_right_second_tokens = cand.tokens[end+1:end+window+1][-1::]

    # print(cand.person1_right_second_tokens)
    return cand


@preprocessor()
def get_right_third_tokens(cand: DataPoint) -> DataPoint:
    """
    Returns tokens in the length 1 window to the right of the person mentions
    """
    # TODO: need to pass window as input params
    window = 3

    end = cand.person1_word_idx[1]
    cand.person1_right_third_tokens = cand.tokens[end+1:end+window+1][-1::]

    end = cand.person2_word_idx[1]
    cand.person2_right_third_tokens = cand.tokens[end+1:end+window+1][-1::]

    # print(cand.person1_right_second_tokens)
    return cand

@preprocessor()
def get_left_first_tokens(cand: DataPoint) -> DataPoint:
    """
    Returns tokens in the length 1 window to the right of the person mentions
    """
    # TODO: need to pass window as input params
    window = 1
    end = cand.person1_word_idx[0]
    cand.person1_left_first_tokens = cand.tokens[end-window:end]

    end = cand.person2_word_idx[0]
    cand.person2_left_first_tokens = cand.tokens[end-window:end]

    # print(cand.person1_right_one_tokens)
    return cand

@preprocessor()
def get_left_second_tokens(cand: DataPoint) -> DataPoint:
    """
    Returns tokens in the length 1 window to the right of the person mentions
    """
    # TODO: need to pass window as input params
    window = 2

    end = cand.person1_word_idx[0]
    cand.person1_left_second_tokens = cand.tokens[end-window:end][0:1]

    end = cand.person2_word_idx[0]
    cand.person2_left_second_tokens = cand.tokens[end-window:end][0:1]

    # print(cand.person1_right_second_tokens)
    return cand


@preprocessor()
def get_left_third_tokens(cand: DataPoint) -> DataPoint:
    """
    Returns tokens in the length 1 window to the right of the person mentions
    """
    # TODO: need to pass window as input params
    window = 3

    end = cand.person1_word_idx[1]
    cand.person1_left_third_tokens = cand.tokens[end-window:end][0:1]

    end = cand.person2_word_idx[1]
    cand.person2_left_third_tokens = cand.tokens[end-window:end][0:1]

    # print(cand.person1_right_second_tokens)
    return cand

# Helper function to get last name for dbpedia entries.
def last_name(s: str) -> Optional[str]:
    name_parts = s.split(" ")
    return name_parts[-1] if len(name_parts) > 1 else None

#计算datapoint中合金名称或者属性值与普通字之间的距离

def get_len(sentence,truple1,truple2,word) -> int:
    """
    sentence为tokens 列表
    nameValue为元组(truple1,truple2)
    word为某个单词
    若不存在，则返回-1.
    否则返回词之间的距离
    """
    #
    # sentence=cand.tokens
    # print(type(sentence))
    if  word in sentence:
        # print("报错")
        position = sentence.index(word)
        if position>truple2:
            return position-truple2
        else:
            return truple1-position
    else:
        return -1

#两个普通字之间的距离
def get_len_twoWords(tokensList,word1,word2):
    """
    tokensList为分词列表
    word为某个单词
    若不存在，则返回-1.
    否则返回词之间的距离
    """
    #
    # sentence=cand.tokens
    if word1 in tokensList and word2 in tokensList:
        position1 = tokensList.index(word1)
        position2=tokensList.index(word2)
        if position1>position2:
            return position1-position2
        else:
            return position2-position1
    else:
        return -1

#合金名称和属性值之间的距离
def get_len_NameValue(truple1_1,truple1_2,truple2_1,truple2_2):
    """
    tokensList为分词列表
    例如: 计算(2,3)和(7,9)之间的距离
    (7,9)(2,3)
    若不存在，则返回-1.
    否则返回词之间的距离
    """
    #
    # sentence=cand.tokens
    if truple1_2<=truple2_1:
        return truple2_1-truple1_2
    else:
        return truple1_1-truple2_2





