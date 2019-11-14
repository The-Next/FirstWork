import os

def get_label(filepath):
    file = open(filepath)
    character_data = []  # 特征
    for line in file.readlines():
        character = line.strip().split()
        character_data.append(list(map(float,character)))
    health = [i for i in character_data  if i[-1] == 1]
    ill = [i for i in character_data  if i[-1] == 0]
    health = list(map(lambda x: x[0:-1], health))
    ill = list(map(lambda x: x[0:-1], ill))
    health_list = []
    ill_list = []
    for i in range(21):
        a = [j[i] for j in health]
        b = [j[i] for j in ill]
        health_list.append(a)
        ill_list.append(b)
    unicom = []
    for i in range(21):
        a = [j[i] for j in character_data]
        unicom.append(list(set(a)))
    return health_list,ill_list,unicom



def count_word_prob(list, union_set):#对每一个特征值分别求频率，竖着求
    prob_list = []
    for i in range(21):
        word_prob = {}
        for word in union_set[i]:
            word_prob[word] = list[i].count(word)/len(list[i])
        prob_list.append(word_prob)
    return prob_list


def filter(ham_word_pro, spam_word_pro, test_file):
    f = open(test_file)
    for line in f.readlines():
        character = line.strip().split()
        character = list(map(float,character))
        email_spam_prob = 0.0
        spam_prob = 0.5
        ham_prob = 0.5
        file_name = character[-1]
        prob_dict = {}
        words = character[0:-1]#获取测试集中单词信息

        for i in range(21):
            Psw = 0.0  # P(S|W)
            if words[i] not in spam_word_pro[i]:
                Psw = 0.4
            else:
                Pws = spam_word_pro[i][words[i]]  # P(W|S),频率代替概率
                Pwh = ham_word_pro[i][words[i]]  # P(W|H)
                Psw = spam_prob * (Pws / (Pwh * ham_prob + Pws * spam_prob))  # P(W|S)P(S)/(P(W|S)P(S)+P(W|H)P(H))
            prob_dict[words[i]] = Psw
        numerator = 1
        denominator_h = 1
        for k, v in prob_dict.items():
            numerator *= v#P1P2P3......
            denominator_h *= (1-v)#(1-p1)(1-p2)(1-p3)......
        if numerator > denominator_h:
            print(file_name,'0')
        else:
            print(file_name,'1')


def main():
    ham_file = r'..\email\ham'
    spam_file = r'..\email\spam'
    train_file = 'horseColicTraining.txt'
    test_file = 'horseColicTest.txt'
    health_list,ill_list,unicom_set = get_label(train_file)
    health_word_pro = count_word_prob(health_list, unicom_set)
    ill_word_pro = count_word_prob(ill_list, unicom_set)
    filter(health_word_pro, ill_word_pro, test_file)


if __name__ == '__main__':
    main()
