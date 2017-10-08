# -*- coding: utf-8 -*-
import csv
import numpy
import matplotlib.pyplot as plt

price, size = numpy.loadtxt('houses.csv', delimiter='|', usecols=(1, 2), unpack=True)

print(price)
print(size)

price_mean = numpy.mean(price)
size_mean = numpy.mean(size)
price_var = numpy.var(price)
size_var = numpy.var(size)

print('Price Average: ', price_mean, 'Price Variance: ', price_var)
print('Size Average: ', size_mean, 'Size Variance: ', size_var)

plt.figure()
plt.subplot(211)
plt.title('/ 10000RMB')
plt.hist(price, bins=20)

plt.subplot(212)
plt.title('/ m**2')
plt.hist(size, bins=20)

plt.figure(2)
plt.title('price')
plt.plot(price)
plt.show()
