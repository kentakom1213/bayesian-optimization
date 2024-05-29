"""ガウス過程回帰の実装
"""
import numpy as np
import matplotlib.pyplot as plt


def rbf(theta_1: float, theta_2: float):
    """動径基底関数カーネルを作成する

    $
    f(x, y) = θ1 * exp(- |x - y| / θ2)
    $

    Args:
        theta_1 (float): θ1
        theta_2 (float): θ2

    Returns:
        callable[(float, float), float]: 動径基底関数
    """
    return lambda x, y: theta_1 * np.exp(- (x - y) ** 2 / theta_2)


class GaussianProcessRegression:
    """ガウス過程回帰を行う
    """

    def __init__(self, kernel: callable):
        # カーネル関数
        self.kernel = kernel
        # 共分散行列
        self.K = None
        # 共分散行列の逆行列
        self.Kinv = None
        # trainデータ
        self.x_train = None
        self.y_train = None
        self.N = None

    def fit(self, x_train, y_train):
        # trainデータの保存
        self.x_train = x_train
        self.y_train = y_train
        self.N = len(x_train)

        K = np.zeros((self.N, self.N))

        # グリッドを生成
        row, col = np.meshgrid(x_train, x_train)

        # 共分散行列を計算
        K = self.kernel(row, col)

        # 逆行列を計算
        Kinv = np.linalg.inv(K)

        self.K = K
        self.Kinv = Kinv

    def predict(self, x):
        row, col = np.meshgrid(x, self.x_train)

        kx = self.kernel(row, col)

        # 予測値の平均
        mu = kx.T @ self.Kinv @ self.y_train

        # 予測値の分散
        var = self.kernel(x.T, x) - (kx.T @ self.Kinv @ kx)

        return mu, var


if __name__ == "__main__":
    import matplotlib.pyplot as plt

    # RBFカーネル
    kernel = rbf(1, 1)

    # GPR
    gpr = GaussianProcessRegression(kernel)

    # テストデータ
    SAMPLE = 7

    x = np.linspace(0, 2 * np.pi, SAMPLE) + np.random.normal(0, 0.2, SAMPLE)
    y = np.sin(x)

    gpr.fit(x, y)

    # 予測
    x_test = np.linspace(-1, 2 * np.pi + 1, 900)
    mu, var = gpr.predict(x_test)

    print(mu)
    print(var)

    sigma = np.sqrt(var.diagonal())

    fig, ax = plt.subplots(figsize=(8, 6), dpi=100)

    # サンプルパス
    samples = np.random.multivariate_normal(mu, var / 900, 5)

    ax.plot(x_test, mu, label="mean")
    ax.plot(x_test, samples.T, color="gray", alpha=0.5)
    ax.fill_between(x_test, mu - sigma, mu + sigma, alpha=0.5, label="std")
    ax.scatter(x, y, label="train")
    ax.legend()
    ax.grid()

    fig.savefig("gpr.png")
