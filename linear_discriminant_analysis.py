import pandas as pd
import numpy as np
from matplotlib import pyplot as plt
from sklearn.datasets.samples_generator import make_blobs


class TwoClassFisherLDA:
    ## Class Variables ##
    X = X_f = C_n = mean = n_k = k_vec = None

    ## Constructors ##
    def __init__(self, final_dim):
        self.n_k = final_dim

    ## Methods ##
    def gen_data(self):
        X, y = make_blobs(n_samples=1000, n_features=3, centers=2, cluster_std=1)

        data_pair = [(X[i], y[i]) for i in range(len(y))]
        data_pair.sort(key=lambda pair: pair[1])

        self.X = {
            "label": np.array([data_pair[k][1] for k in range(len(y))])
            , "data": np.array([data_pair[k][0] for k in range(len(X))])
        }

        assert (len(self.X["data"][0]) >= self.n_k)

        self._init_cn()

    def parse_data(self, file_path, label_col, sep=','):
        df = pd.read_csv(file_path, sep=sep, header=None)
        self.X = {
            "label": np.array(df.pop(label_col))
            , "data": np.array(df)
        }

        assert (len(self.X["data"][0]) >= self.n_k)

        self._init_cn()

    def _init_cn(self):
        self.C_n = {}
        temp = {}
        C_i = 0

        for i, row in enumerate(self.X["data"]):
            if self.X["label"][i] not in temp:
                temp[self.X["label"][i]] = C_i
                self.C_n[C_i] = []
                C_i += 1

            self.C_n[temp[self.X["label"][i]]].append(row)

        for i in range(len(self.C_n.keys())):
            self.C_n[i] = np.array(self.C_n[i])

    def _between_scatter(self):
        self.mean = {}
        self.mean["overall"] = np.array([[np.mean(X_n)] for X_n in self.X['data'].T])
        self.mean['0'] = np.array([[np.mean(X_n)] for X_n in self.C_n[0].T])
        self.mean['1'] = np.array([[np.mean(X_n)] for X_n in self.C_n[1].T])

        f = lambda cn_m, N: N * (cn_m - self.mean["overall"]).dot((cn_m - self.mean["overall"]).T)

        return f(self.mean[str(0)], len(self.C_n[0])) + f(self.mean[str(1)], len(self.C_n[1]))

    def _within_scatter(self):
        s_w = np.zeros((len(self.C_n[0][0]), len(self.C_n[0][0])))

        for i in range(len(self.C_n.keys())):
            s_i = np.zeros((len(self.C_n[0][0]), len(self.C_n[0][0])))

            for j, entry in enumerate(self.C_n[i]):
                s_i += (entry - self.mean[str(i)]).dot((entry - self.mean[str(i)]).T)

            s_w += s_i

        return s_w

    def _calc_k_vec(self):
        s_b = self._between_scatter()
        s_w = self._within_scatter()

        val, vec = np.linalg.eig(
            np.linalg.inv(s_w).dot(s_b))

        # sort eigen
        eigen_pairs = [(np.abs(val[i]), vec[:, i]) for i in range(len(val))]
        eigen_pairs.sort(key=lambda pair: pair[0], reverse=True)

        # choose top k vectors
        self.k_vec = np.array([eigen_pairs[k][1] for k in range(self.n_k)])

    def reduce(self):
        self._calc_k_vec()
        self.X_f = self.k_vec.dot(self.X["data"].T)


def main():
    lda = TwoClassFisherLDA(final_dim=2)
    lda.gen_data()
    lda.reduce()

    plt.scatter(lda.X_f[0][0:len(lda.C_n[0])], lda.X_f[1][0:len(lda.C_n[0])], s=5, color='b')
    plt.scatter(lda.X_f[0][len(lda.C_n[0]):], lda.X_f[1][len(lda.C_n[0]):], s=5, color='r')
    plt.show()


if __name__ == "__main__":
    main()