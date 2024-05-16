"""ガウス過程回帰の実装
"""
import numpy as np


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
    return lambda x, y: theta_1 * np.exp(- np.abs(x - y) / theta_2)


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

    def fit(self, x_train, y_train):
        N = len(x_train)
        K = np.zeros((N, N))

        # グリッドを生成
        row, col = np.meshgrid(x_train, x_train)

        # 共分散行列を計算
        K = self.kernel(row, col)

        # 逆行列を計算
        Kinv = np.linalg.inv(K)

        self.K = K
        self.Kinv = Kinv

    def predict(self, x):
        pass


if __name__ == "__main__":
    # RBFカーネル
    kernel = rbf(1, 1)

    # GPR
    gpr = GaussianProcessRegression(kernel)

    # テストデータ
    x = np.linspace(0, 2 * np.pi, 100)
    y = np.sin(x)

    gpr.fit(x, y)
