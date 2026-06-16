# Import the necessary libraries for DSBCAN ,plotting and making circles.
import matplotlib.pyplot as plt
from sklearn.cluster import DBSCAN
from sklearn.datasets import make_circles
from sklearn.preprocessing import StandardScaler

# Make circles and store X(Variables) and y(labels) values 
X,y = make_circles( 
    n_samples = 500, # The number of rows
    noise = 0.05, # The noise in the sample to create circles (Tune it to your needs, recommended : 0.001 ~ 0.01)
    factor = 0.55, # The scale factor / distance between inner and outer circle.
)

# Scale X as DBSCAN is sensitive to distance.
X = StandardScaler().fit_transform(X)

# Now we wrap the whole animation in try except to handle Errors.
try:
    # The i is a factor which is multiplied to epsilon in each iteration.
    for i in range(1,101):

        # Instantiate the model with the contant min_samples and the current epsilon value.
        dbs = DBSCAN(
            eps =  0.005*i, # Value increases each loop
            min_samples = 5
        )

        y_pred = dbs.fit_predict(X) # Predict the labels

        plt.clf() # The clear function which helps in clearing the graph after each iteration
        plt.scatter(
            x = X[:,0], # The 1'st X column
            y = X[:,1], # The 2'nd X column
            c = y_pred, # The predicted color for X
            label = f'k = {dbs.eps}' # Current K value
        )

        plt.legend() # The legend to display the label
        plt.pause(interval=1) # Pauses the animation each time for the given interval time in seconds
         

except Exception as e :
    print(f'The animation was stopped : {e}') # Handles any error


plt.show() # Shows the animation one by one (interval by interval) after the whole process is completed.
