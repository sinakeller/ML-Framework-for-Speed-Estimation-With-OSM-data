import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.experimental import enable_iterative_imputer
from sklearn.impute import IterativeImputer

import framework.global_data


def load_data(path, feature_columns, target_column, imputer=None, drop_nan=True):
    print("Load Data...")
    ds = pd.read_pickle(path)
    col = feature_columns
    col.append(target_column)
    ds = ds[col]
    y = ds[target_column]
    if imputer is not None:
        print("# Impute missing Data")
        imputer.fit(ds)
        ds_imp = imputer.transform(ds)
        ds[:] = ds_imp
        ds[target_column] = y
    old_size = ds.shape[0]
    if drop_nan:
        print("# Drop entries containing NaN")
        ds.dropna(inplace=True)
        print("#  {} entries droped".format(old_size-ds.shape[0]))
    print('# Finished loading dataset from "{:s}" with shape {}'.format(
        path, ds.shape))
    print('')
    framework.global_data._ds = ds
    framework.global_data._y = ds[target_column]
    framework.global_data._X = ds.drop([target_column], axis=1)


def scale_data(feature_scaler, column_names):
    print("Scale Data...")
    if not set(column_names).issubset(framework.global_data._X.columns):
        print("Not (all) column_names exist in dataframe.")
        return
    if feature_scaler:
        framework.global_data._X_scaled = pd.DataFrame(feature_scaler.fit_transform(
            framework.global_data._X[column_names]), columns=column_names)
    else:
        framework.global_data._X_scaled = pd.DataFrame(
            framework.global_data._X[column_names], columns=column_names)
    print('')


def split_data(test_size, random_state, shuffle=True):
    print("Split Data...")
    framework.global_data._X_train_scaled, framework.global_data._X_test_scaled, framework.global_data._y_train, framework.global_data._y_test = train_test_split(
        framework.global_data._X_scaled, framework.global_data._y, test_size=test_size, random_state=random_state, shuffle=shuffle)
    print("# X_train_scaled shape:", framework.global_data._X_train_scaled.shape)
    print("# y_train shape:", framework.global_data._y_train.shape)
    print("# X_test_scaled shape:", framework.global_data._X_test_scaled.shape)
    print("# y_test shape:", framework.global_data._y_test.shape)
    print('')


def save_datasets(path):
    from pathlib import Path
    Path(path).mkdir(parents=True, exist_ok=True)
    framework.global_data._ds.to_csv(path+'/ds.csv')
    framework.global_data._X.to_csv(path+'/X.csv')
    framework.global_data._y.to_csv(path+'/y.csv')
