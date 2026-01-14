import matplotlib.pyplot as plt
from sklearn.cluster import DBSCAN
from sklearn.datasets import make_circles
from sklearn.preprocessing import StandardScaler

X,y = make_circles(
    n_samples = 500,
    noise = 0.05,
    factor = 0.55,
)

X = StandardScaler().fit_transform(X)

try:
    for i in range(1,101):

        dbs = DBSCAN(
            eps =  0.005*i,
            min_samples = 5
        )

        y_pred = dbs.fit_predict(X)

        plt.clf()
        plt.scatter(
            x = X[:,0],
            y = X[:,1],
            c = y_pred,
            label = f'k = {dbs.eps}'
        )

        plt.legend()
        plt.pause(interval=1)
        

except Exception as e :
    print(f'The animation was stopped : {e}')


plt.show()
