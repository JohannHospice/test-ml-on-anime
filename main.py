import numpy as np
from sklearn.preprocessing import MultiLabelBinarizer, MinMaxScaler, RobustScaler, LabelEncoder, PolynomialFeatures
import csv
import matplotlib.pyplot as plt
from sklearn.neighbors import KNeighborsClassifier
from sklearn.linear_model import LinearRegression
from sklearn.model_selection import GridSearchCV
from sklearn.feature_selection import VarianceThreshold

ANIME_DATA_PATH = 'anime.csv'


def loadData():
    with open(ANIME_DATA_PATH, newline='', encoding="utf8") as csvfile:
        reader = csv.reader(csvfile)
        return [row for row in reader]


def binarizer(X):
    return MultiLabelBinarizer(sparse_output=True).fit_transform(X)


def encodeByLabel(X):
    return LabelEncoder().fit_transform(X)


def normalizeScore(X):
    return MinMaxScaler().fit_transform(X)


def splitNPArray(data):
    return np.array([string.split(', ') for string in data], dtype=object)


def strNPArray(data):
    return np.array([str(string) for string in data])


def intNPArray(data):
    return np.array([[int(string)] for string in data])


def floatNPArray(data):
    return np.array([[float(string)] for string in data])


def cleanData(data):
    cleanedData = [row for row in data[1:] if 'Unknown' not in row]
    dataset = {title: [row[i] for row in cleanedData]
               for i, title in enumerate(data[0])}

    dataset['Genders'] = splitNPArray(dataset['Genders'])
    dataset['Producers'] = splitNPArray(dataset['Producers'])
    dataset['Studios'] = splitNPArray(dataset['Studios'])
    dataset['Source'] = strNPArray(dataset['Source'])
    dataset['Type'] = strNPArray(dataset['Type'])
    dataset['Rating'] = strNPArray(dataset['Rating'])
    dataset['Score'] = floatNPArray(dataset['Score'])
    dataset['Episodes'] = intNPArray(dataset['Episodes'])
    dataset['Favorites'] = intNPArray(dataset['Favorites'])
    dataset['Popularity'] = intNPArray(dataset['Popularity'])
    dataset['Members'] = intNPArray(dataset['Members'])
    dataset['Watching'] = intNPArray(dataset['Watching'])
    dataset['Completed'] = intNPArray(dataset['Completed'])
    dataset['On-Hold'] = intNPArray(dataset['On-Hold'])
    dataset['Dropped'] = intNPArray(dataset['Dropped'])
    dataset['Plan to Watch'] = intNPArray(dataset['Plan to Watch'])

    return dataset, len(cleanedData)


def bestVariableVariance(dataset, names, length):
    normalizedData = [[t[0] for t in MinMaxScaler().fit_transform(dataset[name])]
                      for name in names]

    X = np.array(
        [[normalizedData[nameIndex][pointIndex]
            for nameIndex in range(len(names))] for pointIndex in range(length)]
    )

    print(X.var(axis=0))

    selector = VarianceThreshold(threshold=0.01)
    selector.fit_transform(X)
    selected = np.array(names)[selector.get_support()]
    return selected, X


if __name__ == '__main__':
    data = loadData()

    dataset, length = cleanData(data)

    # gendersEncoded = MultiLabelBinarizer(
    #     sparse_output=True).fit_transform(dataset['Genders'])
    # producerEncoded = MultiLabelBinarizer(
    #     sparse_output=True).fit_transform(dataset['Producers'])
    # studiosEncoded = MultiLabelBinarizer(
    #     sparse_output=True).fit_transform(dataset['Studios'])

    # sourcesEncoded = LabelEncoder().fit_transform(dataset['Source'])
    # typesEncoded = LabelEncoder().fit_transform(dataset['Type'])
    # maturityEncoded = LabelEncoder().fit_transform(dataset['Rating'])

    # popularityNormalized = MinMaxScaler().fit_transform(dataset['Popularity'])
    # scoresNormalized = MinMaxScaler().fit_transform(dataset['Score'])
    # episodesNormalized = MinMaxScaler().fit_transform(dataset['Episodes'])
    # membersNormalized = MinMaxScaler().fit_transform(dataset['Members'])

    # param = {"n_neighbors": np.arange(1, 20), 'metric': [
    #     'euclidean', 'manhattan']}

    # grid = GridSearchCV(KNeighborsClassifier(), param, cv=5)
    # grid.fit(gendersEncoded, typesEncoded)
    # print(grid.best_score_)
    # print(grid.best_params_)

    # model = grid.best_estimator_
    # print(model.score(gendersEncoded, typesEncoded))

    # X_poly = PolynomialFeatures(3).fit_transform(scoresNormalized)
    # model = LinearRegression().fit(X_poly, popularityNormalized)
    # yPred = model.predict(X_poly)
    # print(model.score(scoresNormalized, popularityNormalized))

    selected, X = bestVariableVariance(dataset, ['Popularity',
                                                 'Score',
                                                 'Episodes', 'Dropped', 'Members', 'Watching', 'Completed', 'On-Hold', 'Plan to Watch'], length)
    print(selected)

    plt.plot(X)

    # plt.plot({scoresNormalized, episodesNormalized, popularityNormalized})
    plt.show()

    # from sklearn.datasets import load_iris

    # iris = load_iris()
    # X = iris.data
    # print(X)
    # plt.plot(X)
