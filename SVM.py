import matplotlib.pyplot as plt
from matplotlib import style
import numpy as np
style.use('ggplot')


class SVM:
    def __init__(self, visualization=True):
        self.visualization = visualization
        self.colors = {1: 'r', -1: 'b'}
        if self.visualization:
            self.fig = plt.figure()
            self.ax = self.fig.add_subplot(1, 1, 1)

    def fit(self, data):
        self.max_value = data[1][0][0]
        self.min_value = data[1][0][0]
        self.data = data
        opt_dict = {}
        transforms = [[1, 1], [1, -1], [-1, 1], [-1, -1]]
        for i in data:
            for j in data[i]:
                for k in j:
                    if k > self.max_value:
                        self.max_value = k
                    elif k < self.min_value:
                        self.min_value = k
        steps = [self.max_value*0.1, self.max_value*0.01, self.max_value*0.001]
        b_step = 5
        b_multiple = 5
        opt = self.max_value*10
        for step in steps:
            w = np.array([opt, opt])
            optimized = False
            while not optimized:
                for b in np.arange(-self.max_value*b_step, self.max_value*b_step, step*b_multiple):
                    for t in transforms:
                        w_t = w*t
                        found = True
                        for i in self.data:
                            for xi in self.data[i]:
                                yi = i
                                if yi*(np.dot(xi, w_t)+b) < 1:
                                    found = False
                                    break
                            if not found:
                                break
                        if found:
                            opt_dict[np.linalg.norm(w_t)] = [w_t, b]
                if w[0] < 0:
                    optimized = True
                    print("Optimized")
                else:
                    w = w-step
            norms = sorted([n for n in opt_dict])
            opt_choice = opt_dict[norms[0]]
            self.w = opt_choice[0]
            self.b = opt_choice[1]
            opt = opt_choice[0][0]+step*2

    def predict(self, points):
        # sign( x.w+b )
        prediction = np.sign(np.dot(np.array(points), self.w) + self.b)
        if prediction != 0 and self.visualization:
            self.ax.scatter(points[0], points[1], s=200, marker='*', c=self.colors[prediction])
        return prediction

    def visualize(self):    #for plotting the points and the Support vector plane found
        [[self.ax.scatter(x[0], x[1], s=100, color=self.colors[i]) for x in data_dict[i]] for i in data_dict]  

        # hyperplane = x.w+b
        # v = x.w+b
        # psv = 1
        # nsv = -1
        # dec = 0
        def hyperplane(x, w, b, v):
            return (-w[0] * x - b + v) / w[1]

        datarange = (self.min_feature_value * 0.9, self.max_feature_value * 1.1)
        hyp_x_min = datarange[0]
        hyp_x_max = datarange[1]

        # (w.x+b) = 1
        # positive support vector hyperplane
        psv1 = hyperplane(hyp_x_min, self.w, self.b, 1)
        psv2 = hyperplane(hyp_x_max, self.w, self.b, 1)
        self.ax.plot([hyp_x_min, hyp_x_max], [psv1, psv2], 'k')

        # (w.x+b) = -1
        # negative support vector hyperplane
        nsv1 = hyperplane(hyp_x_min, self.w, self.b, -1)
        nsv2 = hyperplane(hyp_x_max, self.w, self.b, -1)
        self.ax.plot([hyp_x_min, hyp_x_max], [nsv1, nsv2], 'k')

        # (w.x+b) = 0
        # positive support vector hyperplane
        db1 = hyperplane(hyp_x_min, self.w, self.b, 0)
        db2 = hyperplane(hyp_x_max, self.w, self.b, 0)
        self.ax.plot([hyp_x_min, hyp_x_max], [db1, db2], 'y--')

        plt.show()

data_dict = {-1: np.array([[1, 7], [2, 8], [3, 8], ]), 1: np.array([[5, 1], [6, -1], [7, 3], ])}
svm = SVM()
svm.fit(data=data_dict)

predict = [[2, 4], [5, 6], [7, 9], [3, 4], [1, 2], [4, 5]]

for i in predict:
    svm.predict(i)

svm.visualize()
