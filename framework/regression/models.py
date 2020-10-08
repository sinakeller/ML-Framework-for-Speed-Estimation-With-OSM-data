import timeit
import numpy as np
import pandas as pd

from sklearn.linear_model import LinearRegression
from sklearn.ensemble import RandomForestRegressor, BaggingRegressor, ExtraTreesRegressor, AdaBoostRegressor, GradientBoostingRegressor
from sklearn.neural_network import MLPRegressor
from sklearn.neighbors import KNeighborsRegressor
from sklearn.cross_decomposition import PLSRegression
from sklearn.svm import SVR
from sklearn import metrics

from joblib import dump, load

import framework.global_data


def train(X, y, model):
    model.fit(X, y)

    return model


def predict(X, y, model, stats):
    start_time = timeit.default_timer()
    y_pred = model.predict(X)
    elapsed = timeit.default_timer() - start_time
    stats['prediction time'] = elapsed
    stats['prediction RMSE'] = np.sqrt(metrics.mean_squared_error(y, y_pred))
    stats['prediction R2'] = metrics.r2_score(y, y_pred)
    scores = {'explained_variance': metrics.explained_variance_score(y, y_pred),
              'max_error': metrics.max_error(y, y_pred),
              'mean_absolute_error': metrics.mean_absolute_error(y, y_pred),
              'mean_squared_error': metrics.mean_squared_error(y, y_pred),
              'root_mean_squared_error': np.sqrt(metrics.mean_squared_error(y, y_pred)),
              'neg_median_absolute_error': metrics.median_absolute_error(y, y_pred),
              'r2':  metrics.r2_score(y, y_pred)
              }

    stats['scores'] = scores
    return y_pred


def add_som_clusters(num_cluster):
    import susi
    from sklearn.cluster import KMeans

    # SOM clusters
    som = susi.SOMClustering(
        # n_jobs=4,
        n_rows=40,
        n_columns=40,
        learning_rate_start=0.001,
        n_iter_unsupervised=2000000
    )
    som.fit(framework.global_data._X_scaled)

    print("### Adding Som Clusters: X_scaled")
    clusters = som.get_clusters(framework.global_data._X_scaled.values)
    kmeans = KMeans(n_clusters=num_cluster, random_state=0).fit(clusters)
    bmu = som.get_bmus(framework.global_data._X_scaled.values)
    row = kmeans.cluster_centers_[kmeans.labels_, 0]
    row = (row - row.mean()) / row.std()
    col = kmeans.cluster_centers_[kmeans.labels_, 1]
    col = (col - col.mean()) / col.std()
    framework.global_data._X_scaled = framework.global_data._X_scaled.assign(
        som_row_clustered=row)
    framework.global_data._X_scaled = framework.global_data._X_scaled.assign(
        som_column_clustered=col)
    lis = np.array(bmu[:])
    row = lis[:, 0]
    row = (row - row.mean()) / row.std()
    col = lis[:, 1]
    col = (col - col.mean()) / col.std()
    framework.global_data._X_scaled = framework.global_data._X_scaled.assign(
        som_bmu_row=row)
    framework.global_data._X_scaled = framework.global_data._X_scaled.assign(
        som_bmu_column=col)

    print("### Adding Som Clusters: X_train_scaled")
    clusters = som.get_clusters(framework.global_data._X_train_scaled.values)
    km = KMeans(n_clusters=num_cluster, random_state=0).fit(clusters)
    bmu = som.get_bmus(framework.global_data._X_train_scaled.values)
    row = km.cluster_centers_[km.labels_, 0]
    row = (row - row.mean()) / row.std()
    col = km.cluster_centers_[km.labels_, 1]
    col = (col - col.mean()) / col.std()
    framework.global_data._X_train_scaled = framework.global_data._X_train_scaled.assign(
        som_row_clustered=row)
    framework.global_data._X_train_scaled = framework.global_data._X_train_scaled.assign(
        som_column_clustered=col)
    lis = np.array(bmu[:])
    row = lis[:, 0]
    row = (row - row.mean()) / row.std()
    col = lis[:, 1]
    col = (col - col.mean()) / col.std()
    framework.global_data._X_train_scaled = framework.global_data._X_train_scaled.assign(
        som_bmu_row=row)
    framework.global_data._X_train_scaled = framework.global_data._X_train_scaled.assign(
        som_bmu_column=col)

    print("### Adding Som Clusters: X_test_scaled")
    clusters = som.get_clusters(framework.global_data._X_test_scaled.values)
    km = kmeans.predict(clusters)
    bmu = som.get_bmus(framework.global_data._X_test_scaled.values)
    row = kmeans.cluster_centers_[km, 0]
    row = (row - row.mean()) / row.std()
    col = kmeans.cluster_centers_[km, 1]
    col = (col - col.mean()) / col.std()
    framework.global_data._X_test_scaled = framework.global_data._X_test_scaled.assign(
        som_row_clustered=row)
    framework.global_data._X_test_scaled = framework.global_data._X_test_scaled.assign(
        som_column_clustered=col)
    lis = np.array(bmu[:])
    row = lis[:, 0]
    row = (row - row.mean()) / row.std()
    col = lis[:, 1]
    col = (col - col.mean()) / col.std()
    framework.global_data._X_test_scaled = framework.global_data._X_test_scaled.assign(
        som_bmu_row=row)
    framework.global_data._X_test_scaled = framework.global_data._X_test_scaled.assign(
        som_bmu_column=col)


def add_pca(num_cluster):
    from sklearn import decomposition
    pca = decomposition.PCA(n_components=num_cluster)
    pca.fit(framework.global_data._X_scaled)
    framework.global_data._X_scaled = pca.transform(
        framework.global_data._X_scaled)
    framework.global_data._X_train_scaled = pca.transform(
        framework.global_data._X_train_scaled)
    framework.global_data._X_test_scaled = pca.transform(
        framework.global_data._X_test_scaled)


def train_models(results, n_esti=200, generate_som_clusters=False, som_only=False, generate_pca=False, pca_components=None, save=True):
    from sklearn.ensemble import RandomForestRegressor, BaggingRegressor, ExtraTreesRegressor, AdaBoostRegressor, GradientBoostingRegressor, VotingRegressor
    from sklearn.linear_model import LinearRegression, Ridge
    from sklearn.svm import SVR
    import susi
    import numpy as np
    print("Start training of Models...")

    num_cluster = len(framework.global_data._X['class_id'].value_counts())

    if generate_som_clusters:
        print("Generate SOM clusters")
        add_som_clusters(num_cluster)
        if som_only:
            print("Using only SOM clusters")
            som_cols = ['som_row_clustered', 'som_column_clustered',
                        'som_bmu_row', 'som_bmu_column']
            framework.global_data._X_train_scaled = framework.global_data._X_train_scaled.filter(
                som_cols, axis=1)
            framework.global_data._X_test_scaled = framework.global_data._X_test_scaled.filter(
                som_cols, axis=1)
    if generate_pca:
        print("Generate {} PCA components".format(pca_components))
        add_pca(pca_components)

    som = susi.SOMRegressor(
        # n_jobs=4,
        n_rows=40,
        n_columns=40,
        learning_rate_start=0.001,
        n_iter_unsupervised=100000,
        n_iter_supervised=100000
    )
    lr = LinearRegression()
    ab = AdaBoostRegressor(n_estimators=n_esti)
    et = ExtraTreesRegressor(n_estimators=n_esti, n_jobs=-1)
    br = BaggingRegressor(n_estimators=n_esti)
    gb = GradientBoostingRegressor(n_estimators=n_esti, loss='huber')
    rf = RandomForestRegressor(n_estimators=n_esti, n_jobs=-1)
    ridge = Ridge(alpha=1.0)
    svr = SVR(kernel='rbf', gamma='scale')
    models = [
        ('som', som),
        ('lr', lr),
        ('ab', ab),
        ('et', et),
        ('br', br),
        ('gb', gb),
        ('rf', rf),
        ('ridge', ridge),
        ('svr', svr)
    ]

    for (name, model) in models:
        print("# Training: {}".format(name))
        model = train(framework.global_data._X_train_scaled,
                      framework.global_data._y_train, model)
        if save:
            filename = 'framework/models/'+name+'.joblib'
            dump(model, filename)

    return results


def test_model(model, stats):
    y_pred = predict(framework.global_data._X_test_scaled,
                     framework.global_data._y_test, model, stats)
    return y_pred


def test_models(results, savepred=False):
    from pathlib import Path
    import os
    import re

    for entry in os.scandir('framework/models/'):
        if entry.path.endswith(".joblib"):
            name = re.sub(r'((\d-|\d_|\d)|.\w+$)', '', entry.name)
            model = load(entry.path)
            fi = {}
            if isinstance(framework.global_data._X_train_scaled, pd.DataFrame):
                for ind, col in enumerate(framework.global_data._X_train_scaled.columns):
                    if hasattr(model, 'feature_importances_'):
                        fi[col] = model.feature_importances_[ind]
            stats = {"feature importance": fi}
            y_pred = predict(framework.global_data._X_test_scaled,
                             framework.global_data._y_test, model, stats)
            if savepred:
                df = pd.DataFrame()
                df = df.assign(truth=framework.global_data._y_test)
                df['predictions'] = y_pred
                path = 'output/predictions'
                Path(path).mkdir(parents=True, exist_ok=True)
                df.to_csv(path+'/'+name+'.csv')
            results = results.append({'Method': name, 'Prediction Time': stats['prediction time'],
                                      'prediction RMSE': stats['prediction RMSE'], 'prediction R2': stats['prediction R2'], 'feature importance': stats['feature importance'], 'scores': stats['scores']}, ignore_index=True)
    return results
