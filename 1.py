# Три параметра:
# 1. ЖЕНЩИНА
# 2. БЫДЛОКОДЕР
# 3. Вкусно готовит
# Если (1 И 2) ИЛИ ничего = 0
# Если 3 = 1
# Если (1 xor 2) И 3 = 1

import numpy as np
import sys

class POMOGITE(object):

    def sigmoid(self, x):
        return 1 / (1 + np.exp(-x))

    def __init__(self, learn_rate):
        self.w1 = np.random.normal(0.0, 2 ** -0.5, (2, 3))
        self.w2 = np.random.normal(0.0, 1, (1, 2))
        self.smertelniy_nomer = np.vectorize(self.sigmoid)
        self.learn_rate = np.array([learn_rate])

    def predict(self, inputs):
        inputs1 = np.dot(self.w1, inputs)
        outputs1 = self.smertelniy_nomer(inputs1)
        inputs2 = np.dot(self.w2, outputs1)
        outputs2 = self.smertelniy_nomer(inputs2)
        return outputs2
        
    def GYM(self, inputs, expect):
        inputs1 = np.dot(self.w1, inputs)
        outputs1 = self.smertelniy_nomer(inputs1)
        inputs2 = np.dot(self.w2, outputs1)
        outputs = self.smertelniy_nomer(inputs2)

        pred = outputs[0]

        # Дальше файлы смерти:

        error_lay2 = np.array([pred - expect])
        gradient2 = pred * (1 - pred) # Не зря же учила таблицу производных
        wdelta2 = error_lay2 - gradient2
        self.w2 -= np.dot(wdelta2, outputs1.reshape(1, len(outputs1))) * self.learn_rate

        error_lay1 = wdelta2 * self.w2
        gradient1 = outputs1 * (1 - outputs1)
        wdelta1 = error_lay1 * gradient1
        self.w1 -= np.dot(inputs.reshape(len(inputs), 1), wdelta1).T * self.learn_rate
        
# Бог покинул чат

def MSE(o, O):
    return np.mean((o - O) ** 2)
    
BOSS = [
    ([0, 0, 0], 0),
    ([0, 0, 1], 1),
    ([0, 1, 0], 0),
    ([0, 1, 1], 1),
    ([1, 0, 0], 0),
    ([1, 0, 1], 1),
    ([1, 1, 0], 0),
    ([1, 1, 1], 0)
]

# Ниже проверка работоспособности 

#print('Кол-во эпох:', end=" ")
epoX = 5000 #int(input())
#print('learning rate:', end=" ")
learn_rate = 0.07 #float(input())

HELP = POMOGITE(learn_rate=learn_rate)
for i in range(epoX):
    inputs_ = []
    a_ty_ne_poslushal = []
    for inputst, a_ya_govorila in BOSS:
        HELP.GYM(np.array(inputst), a_ya_govorila)
        inputs_.append(np.array(inputst))
        a_ty_ne_poslushal.append(np.array(a_ya_govorila))

    train_loss = MSE(HELP.predict(np.array(inputs_).T), np.array(a_ty_ne_poslushal))
    sys.stdout.write("\Poluchil: {}, I messed up: {}".format(str(100 * i / float(epoX))[:4], str(train_loss)[:5]))

# Поздравляю, это закончилось

for inp, correct in BOSS:
    print("В случае: {} получили: {}, ожидали: {}".format(str(inp), str(HELP.predict(np.array(inp)) > 0.5), str(correct == 1)))
