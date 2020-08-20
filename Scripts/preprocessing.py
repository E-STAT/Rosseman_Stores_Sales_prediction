import numpy as np
import pandas as pd


if __name__ == '__main__':
    train = pd.read_csv('../Supermarket/Data/rossmann-store-sales/train.csv')
    test = pd.read_csv('../Supermarket/Data/rossmann-store-sales/test.csv')
    store = pd.read_csv('../Supermarket/Data/rossmann-store-sales/store.csv')


    def Merge():
        df_train = pd.merge(train, store, on = 'Store')
        df_test = pd.merge(test, store, on = 'Store')

        return df_train, df_test


    def create_feature(dataset):
        """
        The create feature function would help create features using the date
        column and also other features
        """

        # i would map some categorical variable to numerical if
        # that variable present it is replaced
        mappings = {'0': 0, 'a': 1, 'b': 2, 'c': 3, 'd': 4}
        dataset.StoreType.replace(mappings, inplace=True)
        dataset.Assortment.replace(mappings, inplace=True)
        dataset.StateHoliday.replace(mappings, inplace=True)

        # converting some categorical variables to object
        categ = ['DayOfWeek', 'Open', 'Promo', 'StateHoliday', 'SchoolHoliday']
        for i in categ:
            dataset = dataset.astype({i: 'object'})

        # convert date column to datetime
        dataset['Date'] = pd.to_datetime(dataset.Date)

        # Feature creation
        dataset['Year'] = dataset.Date.dt.year
        dataset['Month'] = dataset.Date.dt.month
        dataset['Day'] = dataset.Date.dt.day
        dataset['DayOfWeek'] = dataset.Date.dt.dayofweek
        dataset['WeekOfYear'] = dataset.Date.dt.weekofyear
        dataset = dataset.set_index('Date')
        dataset = dataset.sort_index()
        return dataset


    def Fill_missing():
        df_train, df_test = Merge()
        df_train = create_feature(df_train)
        df_test = create_feature(df_test)
        df_train.fillna(value = 0, inplace = True)
        df_test.fillna(value = 0, inplace = True)
        return df_train, df_test

    df_train, df_test = Fill_missing()

    #Export cleaned data to csv and will be used in building model pipeline
    df_train.to_csv('../Supermarket/Data/cleaned_train.csv', index=False)
    df_test.to_csv('../Supermarket/Data/cleaned_test.csv', index = False)

    print('############## THE END OF SCRIPTS######################')
