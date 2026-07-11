import seaborn as sns
import matplotlib.pyplot as plt

def get_numeric_continous_evaluation(df,numeric_continous_columns):
    for x in numeric_continous_columns:
        print(f'--->{x}<---\n')
        print(f'The Description of the column : \n{df[x].describe()}\n')
        print(f'The Skewness of the column : {df[x].skew():.2f}')
        print(f'The Kurtosis of the column : {df[x].kurt():.2f}\n')

        data = df[x]
        Q1 = data.quantile(0.25)
        Q3 = data.quantile(0.75)
        IQR = Q3 - Q1
        lower_bound = Q1 - (1.5*IQR)
        upper_bound = Q3 + (1.5*IQR)
        no_of_outlier = ((df[x] > upper_bound ) | (df[x] < lower_bound )).sum()
        percentage_outlier = no_of_outlier/len(df)*100

        print(f'{'='*50}IQR{'='*50}')
        print(f'The Lower bound : {lower_bound:.2f}')
        print(f'The Upper bound : {upper_bound:.2f}')
        print(f'The Number of outlier for this column by IQR : {no_of_outlier}')
        print(f'The Percentage of outlier for this column by IQR : {percentage_outlier}')
        print(f'='*103)

        print(f'The Graphical Analysis : \n')

        figure , axes = plt.subplots(1,3,figsize=(14,5))

        sns.histplot(
                        data = df,
                        x = x,
                        ax = axes[0],
                        color = "#D93A3A"
        )
        axes[0].set_title(f'Histplot of {x}')

        sns.boxplot(
                        data = df,
                        y = x,
                        ax = axes[1],
                        color = "#B22A2A"
        )
        axes[1].set_title(f'Boxplot of {x}')

        sns.kdeplot(
                        data = df,
                        x = x,
                        ax = axes[2],
                        color = "#A81010"
        )
        axes[2].set_title(f'Kdeplot of {x}')

        plt.tight_layout()
        plt.grid()
        plt.show()