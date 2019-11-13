from sklearn.linear_model import LogisticRegression
from sklearn.datasets import load_breast_cancer
from sklearn.model_selection import train_test_split
import numpy as np
'''
自己写的,拟合度不咋地
'''

# 激活函数
def sigmoid(z):
    return 1.0 / (1.0 + np.exp(-z))


# 从文件中获取特征值和y值
def get_data(path):
    file = open(path)
    character_data = []  # 特征
    y_data = []  # y值
    for line in file.readlines():
        character = line.strip().split()
        character_data.append(list(map(float, character[0:-1])))
        y_data.append(float(character[-1]))
    return np.mat(character_data), np.mat(y_data)


# 代价函数
def cost(theta, character_data, y_data):
    x = np.multiply(-y_data.T, np.log(sigmoid(character_data * theta.T) + 1e-5))  # 代价函数第一项
    y = np.multiply((1 - y_data.T), np.log(1 - sigmoid(character_data * theta.T) + 1e-5))  # 代价函数第二项
    return np.sum(x - y) / len(x)


# 梯度下降
def gradient_descent(theta, character_data, y_data, alpha):
    m, n = np.shape(character_data)
    for j in range(1000):
        index = list(range(m))
        for i in range(m):
            alpha = 4 / (1.0 + j + i) + 0.0001
            randindex = int(np.random.uniform(0, len(index)))
            h = sigmoid(np.sum(character_data[randindex] * theta))
            error = y_data[randindex] - h
            theta = theta + alpha * error * character_data[randindex]
            del (index[randindex])
    return theta

    # model = LogisticRegression()
    # model.fit(character_data,y_data.T)
    # train_score = model.score(character_data,y_data.T)
    # y_pred = model.predict(text_x)
    # #print(np.c_[y_pred,text_y.T])
    # result = np.equal(y_pred,text_y)
    # print(sum(result.T))
    # print('比率: {0}/{1}'.format(int(sum(result.T)),text_y.T.shape[0]))

    # print(model.predict(character_data)) 俺也不知道为啥，拟合出来的玩意特别离谱
    # print(model.score(character_data, y_data.T))
    # temp = np.mat(np.zeros(theta.shape))
    # for i in range(100000):
    #   h = sigmoid(character_data * theta.T) - y_data.T
    #   for j in range(theta.shape[1]):
    #     t = np.multiply(h,character_data[:,j])
    #     if j == 0:
    #       temp[0,j] = theta[0,j] - ((alpha/len(character_data)) * np.sum(t))
    #     else:
    #       temp[0,j] = theta[0,j] - ((alpha/len(character_data)) * np.sum(t)) + ((alpha/len(character_data)) * theta[:,j])
    #   theta = temp
    # return theta


def classifier(theta, x):
    # for i,j in zip(x,y.T):
    #   result = np.sum(theta.T * i)
    #   print(result)
    #   if sigmoid(result) > 0.5:
    #     print('result = 1','true = '+str(j))
    #   else:
    #     print('result = 0','true = '+str(j))
    p = sigmoid(sum(x * theta))
    if p > 0.5:
        return 1
    else:
        return 0


if __name__ == '__main__':
    character_data, y_data = get_data('horseColicTraining.txt')
    # character_data = np.c_[np.ones(character_data.shape[0]),character_data]#左边添一溜1，给theta0用
    theta = np.ones(character_data.shape[1])  # 初始化theta值
    # theta = np.mat(theta)
    print(theta)
    # x,y = get_data('/content/drive/My Drive/test/horseColicTest.txt')
    f = open('horseColicTest.txt')
    final = gradient_descent(theta, np.array(character_data), y_data.A.tolist()[0],1)  # theta,character_data,y_data,0.0005
    for line in f.readlines():
      currLine = line.strip().split('\t')
      arrayline = []
      for i in range(21):
        arrayline.append(float(currLine[i]))
      print(classifier(final,np.array(arrayline)))
      print(int(float(currLine[21])))
      print(classifier(final,np.array(arrayline))==int(float(currLine[21])))
      print('---------------------------------------')