class eda:

    def __init__(self,data):
        self.data = data
    
    def overview(self):
        print(f'The Head of the dataframe : \n{self.data.head(5).to_string()}\n')
        print(f'The Tail of the dataframe : \n{self.data.tail(5).to_string()}\n')
        print(f'Random samples from the dataframe : \n{self.data.sample(5).to_string()}\n')

    def summary(self):
        print(f'Summary : \nRows : {self.data.shape[0]} \tColumns : {self.data.shape[1]}\n')
        print(f'The List of the columns : \n{self.data.columns.tolist()}\n')
        print(f'Columns with dtype : int \n{self.data.select_dtypes('int').columns.tolist()}\n')    
        print(f'Columns with dtype : float \n{self.data.select_dtypes('float').columns.tolist()}\n')   
        print(f'Columns with dtype : object/str \n{self.data.select_dtypes('str').columns.tolist()}\n')   
        print(f'The number of missing values in each column : \n{self.data.isna().sum()}\n')
        print(f'The number of duplicate values in each column : \n{self.data.duplicated().sum()}\n')

    def description(self):
        print(f'\nThe Column dtypes and non-null count :\n')
        print(f'\n{self.data.info()}\n')
        print(f'The Description of the dataframe : \n{self.data.describe().to_string()}\n')
        print('\nThe following description given below  is for numeric values only\n')
        print(f'The Correlation between numeric columns : \n{self.data.corr(numeric_only=True).to_string()}\n')
        print(f'The Covariance between numeric columns : \n{self.data.cov(numeric_only=True).to_string()}\n')
        print(f'The Skewness in numeric columns : \n{self.data.skew(numeric_only=True).to_string()}\n')
        print(f'The Kurtosis in numeric columns : \n{self.data.kurtosis(numeric_only=True).to_string()}\n')