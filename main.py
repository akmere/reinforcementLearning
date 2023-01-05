import pandas as pd
import numpy as np
import matplotlib.pyplot as plt


def bandit(qw, n, alpha, e, change, rewardNoise = 0, isUcb=False):
    q = np.copy(qw)
    k = len(q)
    Q = np.zeros(k)
    c = 2
    optimalChoices = 0
    optimality = np.zeros(n)
    reps = np.zeros(k)
    choice = 0
    rewards = np.zeros(n)
    for i in range(n):
        if (isUcb):
            # print (Q + c * np.sqrt(np.log(reps)/reps))
            # print(np.sqrt(np.log(reps)/reps))
            if (reps.min() == 0):
                choice = reps.argmin(0)
            else:
                choice = (Q + c * np.sqrt(np.log(i + 1)/reps)).argmax(0)
        elif (np.random.uniform(0, 1) > e):
            choice = Q.argmax(0)
        else:
            choice = np.random.choice(k)
        optimalChoice = q.argmax(0)
        if (choice == optimalChoice):
            optimalChoices+= 1
        reps[choice] += 1
        reward = np.random.normal(q[choice], rewardNoise)
        if (alpha == 0):
            Q[choice] = Q[choice] + (reward-Q[choice])/reps[choice]
        else:
            Q[choice] = Q[choice] + alpha * (reward - Q[choice])
        rewards[i] = reward
        for y in range(k):
            q[y] += change * np.random.normal(0, 1)
        if(i != 0): 
            optimality[i] = optimalChoices/i
    # print(rewards)
    return [rewards, optimalChoices, n, optimality]


nn = 100
nr = 1000
valuesLen = 10
winners = np.zeros(4)
procents = np.zeros(4)
ens = np.zeros(4)
sums = np.zeros(4)
optimalities1 = np.zeros(nr)
optimalities2 = np.zeros(nr)
optimalities3 = np.zeros(nr)
optimalities4 = np.zeros(nr)
averageRewards1 = np.zeros(nr)
averageRewards2 = np.zeros(nr)
averageRewards3 = np.zeros(nr)
averageRewards4 = np.zeros(nr)
for p in range(nn):
    qv = np.random.normal(0, 1, valuesLen)
    # qv = [0.1, 0.11, 0.12, 0.13, 0.14, 0.09, 0.08, 0.07, 0.06, 0.05]
    r1 = bandit(qv, nr, alpha=0.3, e=0.01, change=0, rewardNoise=0)
    r2 = bandit(qv, nr, alpha=0, e=0.1, change=0, rewardNoise=0)
    r3 = bandit(qv, nr, alpha=0.1, e=0, change=0, rewardNoise=0)
    r4 = bandit(qv, nr, alpha=0.2, e=0, change=0, rewardNoise=0, isUcb=True)
    optimalities1 += r1[3]
    optimalities2 += r2[3]
    optimalities3 += r3[3]
    optimalities4 += r4[3]
    averageRewards1 += r1[0]/nn
    averageRewards2 += r2[0]/nn
    averageRewards3 += r3[0]/nn
    averageRewards4 += r4[0]/nn
    # print(averageRewards2)
    # print(r2[0])
# print(averageRewards4)
# print(optimalities1/nn)
# print(optimalities4/nn)

# print(averageRewards4)

# plt.plot(optimalities1/nn, 'r')
# plt.plot(optimalities2/nn, 'g')
# plt.plot(optimalities3/nn, 'm')
# plt.plot(optimalities4/nn, 'b')
# plt.show()
# print(averageRewards4)
plt.plot(averageRewards1, 'r')
plt.plot(averageRewards2, 'g')
plt.plot(averageRewards3, 'm')
plt.plot(averageRewards4, 'b')
plt.show()