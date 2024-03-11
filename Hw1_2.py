import numpy as np
import matplotlib.pyplot as plt
import sklearn as sk
import pandas as pd
import random
import math
from sklearn.preprocessing import StandardScaler
import statistics

# read Kmean_3D_data File and preprocessing
cust_df = pd.read_csv('Customers.csv')
cust_df.head()
cust_df = cust_df.values[:, [3, 4]]
cust_df_income = cust_df[:, 0]
cust_df_Score = cust_df[:, 1]

# select 5 random center point
center_points = cust_df[np.random.choice(
    cust_df.shape[0], size=5, replace=False)]

# define Distance of 2 point function


def distance_3D(center_point, node):
    return math.sqrt(((center_point[0]-node[0])**2)+((center_point[1]-node[1])**2))


n = 10
total = 1
while (n > 0 and total != 0):
    # create a array : cluster of Data
    clustering_arra = []

    for node in cust_df:
        arra = []
        for center_point in center_points:

            a = distance_3D(center_point, node)
            arra.append(a)
        clustering_arra.append(arra.index(min(arra)))

    # append clustering_arra array to cus_df
    cust_df = np.c_[cust_df, clustering_arra]

    # define 5 cluster array
    cluster0 = cust_df[cust_df[:, -1] == 0, :]
    cluster1 = cust_df[cust_df[:, -1] == 1, :]
    cluster2 = cust_df[cust_df[:, -1] == 2, :]
    cluster3 = cust_df[cust_df[:, -1] == 3, :]
    cluster4 = cust_df[cust_df[:, -1] == 4, :]
    total = 0
    old_center_points = center_points

    # Update the center_points
    center_points = np.array([[statistics.mean(cluster0[:, 0]), statistics.mean(cluster0[:, 1])],
                              [statistics.mean(cluster1[:, 0]),
                               statistics.mean(cluster1[:, 1])],
                              [statistics.mean(cluster2[:, 0]),
                               statistics.mean(cluster2[:, 1])],
                              [statistics.mean(cluster3[:, 0]),
                               statistics.mean(cluster3[:, 1])],
                              [statistics.mean(cluster4[:, 0]),
                               statistics.mean(cluster4[:, 1])]
                              ])

    Diff_matrix = np.subtract(old_center_points, center_points)
    total = np.sum(Diff_matrix[:, 0])+np.sum(Diff_matrix[:, 1])
    n -= 1


fig = plt.figure()
ax = fig.add_subplot()
plt.scatter(cluster0[:, 0], cluster0[:, 1],
            color='red', label='Group 1', marker='.')
plt.scatter(cluster1[:, 0], cluster1[:, 1],
            color='yellow', label='Group 2', marker='.')
plt.scatter(cluster2[:, 0], cluster2[:, 1],
            color='blue', label='Group 3', marker='.')
plt.scatter(cluster3[:, 0], cluster3[:, 1],
            color='green', label='Group 4', marker='.')
plt.scatter(cluster4[:, 0], cluster4[:, 1],
            color='orange', label='Group 5', marker='.')

ax.scatter(center_points[:, 0], center_points[:, 1], marker='o', color="black")
plt.show(block=True)
