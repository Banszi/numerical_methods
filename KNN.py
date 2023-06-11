'''
Algorithm: KNN - K nearest neighbors
Dataset: CSV file with following columns: Id, SepalLengthCm, SepalWidthCm, PetalLengthCm, PetalWidthCm, Species
'''


import pandas as pd
import numpy as np
from pathlib import Path
import random
from collections import Counter
from matplotlib import pyplot as plt


__author__ = 'Sebastian Stach'


CSV_FILE_PATH = Path(r'Iris.csv')
DATA_SET_TEST_SIZE = 0.2
K = 5

IRIS_MAP = {
    'Iris-setosa': 0,
    'Iris-versicolor': 1,
    'Iris-virginica': 2
}


class KNN:

    def __init__(self):
        self.k = K

    def set_training_data(self, X_train, y_train):
        self.X_train = X_train
        self.y_train = y_train

    def predict(self, X_test):
        predictions = [self._predict_helper(x_test) for x_test in X_test]
        return predictions

    def _predict_helper(self, x_test):
        # Calculate euclidean distance between two points
        distances = [self.calculate_euclidean_distance(x_test, x_train) for x_train in self.X_train]

        # Sort distances from low to high and only this values which are lower than K parameter
        k_indexes = np.argsort(distances)[:self.k]
        k_nearest_labels = [self.y_train[idx] for idx in k_indexes]

        # Search for the most common values
        most_common = Counter(k_nearest_labels).most_common()
        # Take only value of most common elements from tuple of tuples: ((x,y), (x,y) ...)
        # Take first tuple and first element in this tuple
        return most_common[0][0]

    def calculate_euclidean_distance(self, x1, x2):
        distance = np.sqrt(np.sum((x1-x2)**2))
        return distance


def create_df_from_csv_file(csv_file_path: Path) -> pd.DataFrame:
    '''
    Read CSV file and create DataFrame using all data
    '''
    df = pd.read_csv(csv_file_path)
    df.drop(columns=['Id'], inplace=True)
    return df


def divide_to_data_and_target(df: pd.DataFrame) -> tuple:
    '''
    Separate DataFrame to data df and target df
    Targer is set in last column in input DataFrame
    return: (data, target)
    '''
    data = df[df.columns[:-1]]
    target = df[df.columns[-1]]
    target = target.replace('Iris-setosa', IRIS_MAP['Iris-setosa'])
    target = target.replace('Iris-versicolor', IRIS_MAP['Iris-versicolor'])
    target = target.replace('Iris-virginica', IRIS_MAP['Iris-virginica'])

    return data, target


def divide_data_to_test_and_train_sets(data_df: pd.DataFrame, target_df: pd.DataFrame, test_size: float) -> tuple:
    '''
    Divide data to test and train data sets by given test size factor
    test_size must be bigger than 0.0 and lower than 1.0
    return: (test_data_df, train_data_df, test_target_df, train_target_df)
    '''
    df_len = len(data_df)
    how_many_indexes_to_test_set = int(test_size * df_len)
    random_indexes_list = random.sample(range(0, df_len), how_many_indexes_to_test_set)

    test_data_df = data_df.loc[data_df.index.isin(random_indexes_list)]
    train_data_df = data_df.loc[~data_df.index.isin(random_indexes_list)]

    test_target_df = target_df.loc[target_df.index.isin(random_indexes_list)]
    train_target_df = target_df.loc[~target_df.index.isin(random_indexes_list)]

    return train_data_df, test_data_df, train_target_df, test_target_df


def convert_data_into_arrays(X_train_df, X_test_df, y_train_df, y_test_df) -> tuple:
    '''
    Convert given DataFrames into arrays data type
    '''

    X_train = X_train_df.values
    X_test = X_test_df.values
    y_train = y_train_df.values
    y_test = y_test_df.values

    return X_train, X_test, y_train, y_test


def prepare_scatter_plot(df, col_name_1, col_name_2, target) -> None:
    '''
    Draw a scatter plot for two data column and target
    '''
    plt.figure()
    scatter = plt.scatter(df[col_name_1], df[col_name_2], c=target)
    plt.xlabel(col_name_1)
    plt.ylabel(col_name_2)
    plt.legend(handles=scatter.legend_elements()[0], labels=IRIS_MAP.keys())
    plt.title(f'Scatter plot for {col_name_1} and {col_name_2}')
    plt.show()


def prepare_comparison_scatter_plot(X_test_df, y_test, predictions, col_name_1, col_name_2) -> None:
    '''
    Draw a comparison scatter plot with original data and predicted data
    '''
    f, (ax1, ax2) = plt.subplots(1, 2)
    scatter_original = ax1.scatter(X_test_df[col_name_1], X_test_df[col_name_2], c=y_test)
    ax1.legend(handles=scatter_original.legend_elements()[0], labels=IRIS_MAP.keys())
    ax1.set(title='Original data from csv file')
    scatter_predicted = ax2.scatter(X_test_df[col_name_1], X_test_df[col_name_2], c=predictions)
    ax2.legend(handles=scatter_predicted.legend_elements()[0], labels=IRIS_MAP.keys())
    ax2.set(title='Predicted data by KNN algorithm')
    plt.show()


def calculate_prediction_ratio(y_test, predictions) -> float:
    prediction_ratio = (np.sum(y_test == predictions) / len(y_test)) * 100

    return prediction_ratio


if __name__ == '__main__':

    iris_df = create_df_from_csv_file(CSV_FILE_PATH)
    data_df, target_df = divide_to_data_and_target(iris_df)

    prepare_scatter_plot(data_df, 'SepalLengthCm', 'PetalLengthCm', target_df)

    X_train_df, X_test_df, y_train_df, y_test_df = divide_data_to_test_and_train_sets(data_df, target_df, test_size=DATA_SET_TEST_SIZE)
    X_train, X_test, y_train, y_test = convert_data_into_arrays(X_train_df, X_test_df, y_train_df, y_test_df)

    knn = KNN()
    knn.set_training_data(X_train, y_train)
    predictions = knn.predict(X_test)

    prepare_comparison_scatter_plot(X_test_df, y_test, predictions, 'SepalLengthCm', 'PetalLengthCm')

    print(f'Original test set output:  {list(y_test)}')
    print(f'Predicted test set output: {list(predictions)}')

    prediction_ratio = calculate_prediction_ratio(y_test, predictions)

    print(f'Prediction ratio: {prediction_ratio:.2f}%')



