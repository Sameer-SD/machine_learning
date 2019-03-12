from statistics import mean
import numpy as np
import matplotlib.pyplot as plt
from matplotlib import style
import random

style.use('fivethirtyeight')

# xs=np.array([1,2,3,4,5,6],dtype=np.float64)
# ys=np.array([5,4,5,4,6,7],dtype=np.float64)


def create_dataset(hm, variance, step=2, correlation=False):

    val = 1
    ys = []
    for i in range(hm):
        y=val+random.randrange(-variance, variance)
        ys.append(y)
        if correlation and correlation == 'pos':
            val += step
        elif correlation and correlation == 'neg':
            val -= step
    xs = [i for i in range(len(ys))]

    return np.array(xs, dtype=np.float64), np.array(ys, dtype=np.float64)


def best_fit_slope(xs, ys):
    m = ((mean(xs)*mean(ys)) - mean(xs*ys))/(mean(xs)*mean(xs) - mean(xs*xs))
    return m


xs, ys = create_dataset(40, 5, 2, 'pos')

m = best_fit_slope(xs, ys)
print(m)


def best_fit_y_intercept(xs, ys):
    b = mean(ys)-m*mean(xs)
    return b


b = best_fit_y_intercept(xs, ys)
print(b)


def squared_error(ys_orig, ys_line):
    return sum((ys_line-ys_orig)**2)


def coefficient_of_det(ys_orig, ys_line):
    y_mean_line = [mean(ys_orig) for y in ys_orig]
    squared_error_reg = squared_error(ys_orig, ys_line)
    squared_error_y_mean = squared_error(ys_orig, y_mean_line)
    return 1-(squared_error_reg/squared_error_y_mean)


line = [(m*x+b) for x in xs]
accuracy = coefficient_of_det(ys, line)
print(accuracy)

plt.scatter(xs, ys)
plt.plot(xs, line)
plt.show()
