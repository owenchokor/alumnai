import matplotlib.pyplot as plt

# Data for two scatter plots
x1 = [1, 2, 3, 4, 5]
y1 = [10, 15, 8, 20, 18]

x2 = [1, 2, 3, 4, 5]
y2 = [5, 7, 6, 10, 9]

# Scatter plots
plt.scatter(x1, y1, color='blue', label='Dataset 1')
plt.scatter(x2, y2, color='red', label='Dataset 2')

# Auto-scaled axes
plt.legend()
plt.title("Auto-Scaled Scatter Plot")
plt.show()
