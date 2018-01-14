import numpy as np
import matplotlib.pyplot as plt
from tensorflow.examples.tutorials.mnist import input_data


mnist = input_data.read_data_sets('MNIST_data', one_hot=True)

# test_data = mnist.test.images[5].reshape(28, 28)
# for j in range(28):
#     showList = ''
#     for i in range(28):
#         if test_data[i + 28 * j] > 0:
#             showList += '1 '
#         else:
#             showList += '0 '
#     print showList

# for j in range(28):
#     showList = ''
#     for i in range(28):
#         if test_data[j][i] > 0:
#             showList += '1 '
#         else:
#             showList += '0 '
#     print showList

fig = plt.figure()
ax = fig.add_subplot(111)
plt.ion()
plt.show()

for pN in range(100):

    try:
        ax.lines.remove(scatter_points[0])
    except Exception:
        pass

    test_data = (mnist.test.images[pN]).reshape(28, 28)
    xs = []
    ys = []
    for j in range(28):
        for i in range(28):
            if test_data[j][i] > 0:
                xs.append(np.float32(i))
                ys.append(np.float32((28 - j)))

    scatter_points = ax.plot(xs, ys, 'ro')

    plt.pause(0.1)



