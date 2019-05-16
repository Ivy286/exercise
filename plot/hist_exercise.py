import numpy as np
import matplotlib.pyplot as plt

# Fixing random state for reproducibility
np.random.seed(19680801)

mu, sigma = 100, 15
x = mu + sigma * np.random.randn(10000)

# the histogram of the data
n, bins, patches = plt.hist(x, 50, density=True, facecolor='g', alpha=0.75)

fontdict = {'family': 'Times New Roman', 'size': 12}

# plt.xlabel('Smarts', fontweight='semibold')
plt.xlabel('Smarts', fontdict)
plt.ylabel('Probability', fontweight='semibold', family='Times New Roman')
plt.title('Histogram of IQ', fontweight='semibold', size=12, family='Times New Roman')
plt.text(60, .025, r'$\mu=100,\ \sigma=15$')
plt.axis([40, 160, 0, 0.03])

plt.xticks(fontproperties='Times New Roman', size=10)
plt.yticks(fontproperties='Times New Roman', size=10)

plt.grid(linestyle='--')
plt.show()
