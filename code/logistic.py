'''
逻辑回归垃圾邮件分类
'''
import os
import pandas as pd
import numpy as np
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.linear_model.logistic import LogisticRegression
from sklearn.model_selection import train_test_split,cross_val_score

def readtxt(path,encoding):
    with open(path, 'r', encoding = encoding) as f:
        lines = f.readlines()
    return ''.join(lines)

def fileWalker(path):
    fileArray = []
    for root, dirs, files in os.walk(path):
        for fn in files:
            eachpath = str(root+'\\'+fn)
            fileArray.append(eachpath)
    return fileArray

def email_parser(email_path):
    content_list = readtxt(email_path, 'utf8')
    return content_list


def get_word(email_file):
    word_list = []
    word_set = []
    punctuations = """,.<>()*&^%$#@!'";~`[]{}|、\\/~+_-=?"""
    email_paths = fileWalker(email_file)
    for email_path in email_paths:
        # content_list = readtxt(email_path, 'utf8')
        # content = (' '.join(content_list)).replace(
        #     '\r\n', ' ').replace('\t', ' ')
        # for punctuation in punctuations:
        #     # content = content.replace(punctuation, '').replace('  ', ' ')
        #     content = (' '.join(content.split(punctuation))).replace('  ', ' ')
        # clean_word = [word.lower()
        #               for word in content.split(' ') if len(word) > 2]
        clean_word = email_parser(email_path)
        word_list.append(clean_word)
        word_set.extend(clean_word)
    return word_list, set(word_set)

def load(ham_list, spam_list):
    sum_list = ham_list + spam_list
    a = ['ham'] * len(ham_list)
    b = ['spam'] * len(spam_list)
    c = a+b
    diction = {}
    diction['type'] = c
    diction['label'] = sum_list
    return diction

def load_test(paths):
    diction = {}
    label = []
    type = []
    testpaths = fileWalker(paths)
    for testpath in testpaths:
        type.append(testpath.split('\\')[-1])
        label.append(email_parser(testpath))
    diction['type'] = type
    diction['label'] = label
    return diction

if __name__ == '__main__':
    ham_file = r'..\email\ham'
    spam_file = r'..\email\spam'
    test_file = r'..\email\test'
    ham_list, ham_set = get_word(ham_file)
    spam_list, spam_set = get_word(spam_file)
    diction = load(ham_list,spam_list)
    df = pd.DataFrame(data=diction)
    print("垃圾邮件个数：%s" % df[df['type'] == 'spam']['type'].count())
    print("正常邮件个数：%s" % df[df['type'] == 'ham']['type'].count())
    vectorizer = TfidfVectorizer()#词频-逆向文件频率
    X = df['label'].values
    y = df['type'].values

    train_x = vectorizer.fit_transform(X)#拟合数据
    lr = LogisticRegression()
    lr.fit(train_x,y)#分类

    testdf = pd.DataFrame(data=load_test(test_file))
    xtest = vectorizer.transform(testdf['label'].values)#标准化数据

    predictions = lr.predict(xtest)#预测
    for i ,prediction in enumerate(predictions):
        print("预测为 %s ,信件为 %s" % (prediction, testdf['type'].values[i]))