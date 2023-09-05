from sklearn.preprocessing import MinMaxScaler, StandardScaler, RobustScaler, Normalizer, MaxAbsScaler, PowerTransformer, QuantileTransformer
import pandas as pd
import numpy as np

class Normalise_Engine():
    '''
        Normalise_Engine Creates a stored instance of the Standardising method from Sklearn (Consistent instance)
        Also holds methods related to its function (normalise and inverse)
    '''

    processed_columns = {}
        
    def __init__(self, include_cols=[]):
        self.engine = MaxAbsScaler()

    ''' Log and reverse log functions '''
    def log_data(self, df, include_cols=[]):
        ''' 
            Take a dataframe and logs every column except specified ones. Replaces all NaN values with 0 before returning 
            Params:
                ► df as DataFrame : dataframe to convert to log values
                ► exclude as list : List of column names to exclude from logging → Add string value columns here
                ► display as Boolean : True/False whether to display the print lines in the terminal
        '''
        log_df = pd.DataFrame()
        for column in df:
            if column in include_cols:
                df[column] = df[column].replace(0, np.nan)
                log_df[column] = np.log10(df[column].replace(0, np.nan))
                self.processed_columns.update({column:"Logged"})
            else:
                log_df[column] = df[column]
                self.processed_columns.update({column:"Unchanged"})

        log_df = log_df.replace(np.nan, 0)

        return log_df


    ''' Reverse Column Logging '''
    def reverse_log_data(self, log_df, include_cols=[]):
        ''' 
            Reverses logging the data
            Params:
                ► df as DataFrame : dataframe to convert to log values
                ► exclude as list : List of column names to exclude from logging → Add string value columns here
                ► display as Boolean : True/False whether to display the print lines in the terminal
        '''
        df = pd.DataFrame()
        for column in log_df:
            if column in include_cols:
                df[column] = 10 ** log_df[column]
            else:
                df[column] = log_df[column]

        df = df.replace(np.nan, 0)

        return df

    def normalise(self, data, include_cols):
        '''
            normalise method takes raw data and a list of columns to normalise and returns the same dataset with specified columns normalised
            Params:
                ► data as Pandas DataFrame : Raw data to normalise
                ► include_cols as list : List of column names to normalise
        '''
        target_data = data[include_cols]

        if len(target_data.shape) == 1:
            target_data = target_data.values.reshape(-1, 1)

        normalised_data = self.engine.fit_transform(target_data)
        normalised_data = pd.DataFrame(normalised_data, columns=include_cols)

        data_difference = data[list(set(data.columns).difference(include_cols))].reset_index(drop=True)

        return pd.concat([data_difference, normalised_data], axis=1)

    def inverse(self, data, include_cols):
        '''
            inverse method reverses the normalisation so that data can be returned to a normal state
            params:
                ► data as Pandas DataFrame : Raw data to normalise
                ► include_cols as list : List of column names to normalise
        '''
        
        inversed_data = self.engine.inverse_transform(data[include_cols])
        inversed_data = pd.DataFrame(inversed_data, columns=include_cols)

        data_difference = data[list(set(data.columns).difference(include_cols))].reset_index(drop=True)

        return pd.concat([data_difference, inversed_data], axis=1)
