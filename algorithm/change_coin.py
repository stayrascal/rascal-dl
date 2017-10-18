#! /usr/bin/env python
# coding:utf-8


def change_coin(money):
    coin = [1, 2, 5, 10, 20, 50, 100]
    coin.sort(reverse=True)
    money = money * 100
    change = {}

    for one in coin:
        num_coin = money // one
        if num_coin > 0:
            change[one] = num_coin
        money = money % one
        if money == 0:
            break
    return change


def coinChange(centsNeeded, coinValues):
    # coinValues = [1, 2, 5, 10, 20, 50, 100]
    minCoins = [[0 for j in range(centsNeeded + 1)] for i in range(len(coinValues))]
    minCoins[0] = list(range(centsNeeded + 1))
    print(minCoins)
    for i in range(1, len(coinValues)):
        print(minCoins)
        for j in range(0, centsNeeded + 1):
            if j < coinValues[i]:
                minCoins[i][j] = minCoins[i - 1][j]
            else:
                minCoins[i][j] = min(minCoins[i - 1][j], 1 + minCoins[i][j - coinValues[i]])
    return minCoins[-1][-1]


if __name__ == '__main__':
    money = 3.42
    coin = [1, 2, 5, 10, 20, 50, 100]
    num_coin = change_coin(money)
    result = [(key, num_coin[key]) for key in sorted(num_coin.keys())]
    print('You have {} RMB'.format(money))
    print('I had to change you:')
    print('Coin Number')
    for i in result:
        if i[0] == 100:
            print('Yuan {}  {}'.format(i[0] / 100, i[1]))
        if i[0] < 10:
            print('Fen  {}  {}'.format(i[0], i[1]))
        else:
            print('Jiao {}  {}'.format(i[0] / 10, i[1]))

    print()
    num2 = coinChange(5, coin)
    print(num2)
