def column_cardinality_analysis(df,threshold=5):

    print(f'Threshold : {threshold}\n')

    rare_columns = []

    for i in df.select_dtypes('object').columns:     # target column checking skipped as it is already not included in for loop(is a numeric value)

        x = df[i].value_counts()/len(df)*100

        rare_unique_values = x[x < threshold].index.to_list()
        common_values = x[x >= threshold].index.to_list()

        if rare_unique_values:

            rare_columns.append(i)

            print(f'--->{i}<---\nRare Unique Values{f'({len(rare_unique_values)})'} : {rare_unique_values}\nCommon Unique Values{f'({len(common_values)})'} : {common_values}\n')
            print(f'Rare Categorical Percentage : {x[x < threshold ]}\n')
            print(f'Total Percentage of Rare Columns : {x[x < threshold ].sum()}\n')
            print(f'Common:Rare Column Ratio :   {x[x >= threshold].sum():.2f} : {x[x < threshold].sum():.2f}\n')

    print(f'List of columns Having rare values : {rare_columns} ')
    
    return rare_columns
