import numpy as np
import torch

ITER = 200
K = 9

def get_data(data_path):
    f = open(data_path, 'r')
    n_data = len(f.readlines())
    clk = 0
    f.close()

    x_data = np.zeros([n_data, 3])
    y_data = np.zeros(n_data)

    f = open(data_path, 'r')
    for line in f.readlines():
        line_y = line.split(':')[0]
        line_x1 = line.split(':')[1]
        line_x2 = line_x1.split('(')[1]
        line_x3 = line_x2.split(')')[0]
        line_x4 = line_x3.split(' ')
        if line_y == 'A':
            y_data[clk] = 1
        else:
            y_data[clk] = 0
        for j in range(3):
            x_data[clk][j] = float(line_x4[j])
        clk += 1

    f.close()
    return x_data, y_data

def MLE_Gaussian(x_data):
    u = np.zeros(3)
    sigma = np.zeros([3, 3])
    for i in range(x_data.shape[0]):
        u = u + x_data[i]
    u = u / x_data.shape[0]
    for i in range(x_data.shape[0]):
        sigma = sigma + np.mat(x_data[i] - u).T * np.mat(x_data[i] - u)
    sigma = sigma / x_data.shape[0]
    return u, sigma

def func_Gaussian(x, u, sigma):
    # x [3,1] ty:np.matrix
    # u [3,1] ty:np.matrix
    # sigma [3,3] ty:np.matrix
    d = 3
    return 1 / ((2 * np.pi) ** (d / 3)) / (np.linalg.det(sigma) ** 0.5) * np.exp(-0.5 * ((x - u).T * sigma.I * (x - u)))

def func_GMM(x, pi, u, sigma, k):
    # x [3,1] ty:np.matrix
    # pi [k,] ty:np.array
    # u [k,3] ty:np.matrix
    # sigma [k,3,3] ty:torch.tensor
    d = 3
    res = 0
    for i in range(k):
        res += pi[i] / ((2 * np.pi) ** (d / 3)) / (np.linalg.det(np.mat(sigma[i])) ** 0.5) * np.exp(-0.5 * ((x - u[i].T).T * np.mat(sigma[i]).I * (x - u[i].T)))
    return res

def compute_LLE(pi, u, sigma, x_data, k):
    res = 0
    for i in range(x_data.shape[0]):
        res += np.log(func_GMM(np.mat(x_data[i]).T, pi, u, sigma, k))
    return res

def init_centroids(x_data, k):
    m, n = x_data.shape
    centroids = np.zeros([k, n])
    idx = np.random.randint(0, m, k)
    for i in range(k):
        centroids[i, :] = x_data[idx[i], :]
    return centroids

def find_closest_centroids(x_data, centroids):
    m = x_data.shape[0]
    k = centroids.shape[0]
    idx = np.zeros(m)
    for i in range(m):
        min_dist = 999999999
        for j in range(k):
            dist = np.sum((x_data[i, :] - centroids[j, :]) ** 2)
            if dist < min_dist:
                min_dist = dist
                idx[i] = j
    return idx

def compute_centroids(x_data, idx, k):
    m, n = x_data.shape
    centroids = np.zeros([k, n])
    for i in range(k):
        indices = np.where(idx == i)
        centroids[i] = (np.sum(x_data[indices[0], :], axis=0) / len(indices[0])).ravel()
    return centroids

def k_means(x_data, k):
    m, n = x_data.shape
    centroids = init_centroids(x_data, k)
    idx = np.zeros(m)
    for i in range(100): # 迭代100次
        idx = find_closest_centroids(x_data, centroids)
        centroids = compute_centroids(x_data, idx, k)
    return idx, centroids

def compute_gamma(pi, u, sigma, x_data):
    k = len(pi)
    n = x_data.shape[0]
    gamma = np.zeros([k, n])
    down = np.zeros(n)
    for i in range(n):
        for j in range(k):
            down[i] += pi[j] * func_Gaussian(np.mat(x_data[i]).T, np.mat(u[j]).T, np.mat(sigma[j]))
    for i in range(n):
        for j in range(k):
            up = pi[j] * func_Gaussian(np.mat(x_data[i]).T, np.mat(u[j]).T, np.mat(sigma[j]))
            gamma[j][i] = up / down[i]
    return gamma

def EM(x_data, k):
    N = x_data.shape[0]
    idx, centroids = k_means(x_data, k)
    pi = np.zeros(k)
    u = np.zeros([k, 3])
    sigma = torch.zeros([k, 3, 3])
    # 初始化参数pi,u,sigma
    for i in range(k):
        pi[i] = np.sum(idx == i) / len(idx)
        x_data_i = x_data[np.where(idx == i)[0], :]
        u[i, :], sigma_out = MLE_Gaussian(x_data_i)
        sigma[i, :] = torch.tensor(sigma_out)
    # E step
    gamma = compute_gamma(pi, u, sigma, x_data)
    # M step
    for n in range(ITER):
        Nk = np.zeros(k)
        for i in range(k):

            Nk[i] = np.sum(gamma[i])
            pi[i] = Nk[i] / N
            temp1 = np.zeros(3)
            temp2 = np.zeros([3, 3])
            for j in range(N):
                temp1 += gamma[i][j] * x_data[j]
            u[i] = 1 / Nk[i] * temp1
            for j in range(N):
                temp2 += gamma[i][j] * (np.mat(x_data[j] - u[i]).T * np.mat(x_data[j] - u[i]))
            sigma[i] = torch.tensor(1 / Nk[i] * temp2)

            gamma = compute_gamma(pi, u, sigma, x_data)

        # u = np.mat(u)
        # print(compute_LLE(pi, u, sigma, x_data, k))
    return pi, u, sigma

if __name__ == '__main__':
    x_train, y_train = get_data('train.txt')
    x_test, y_test = get_data('test.txt')
    y_testing = np.zeros(y_test.shape)

    n1 = int(np.sum(y_train))
    n2 = x_train.shape[0] - n1
    x_train1 = np.zeros([n1, 3])
    x_train2 = np.zeros([n2, 3])
    for i in range(n1):
        x_train1[i] = x_train[i]
    for i in range(n2):
        x_train2[i] = x_train[n1 + i]
    pi1, u1, sigma1 = EM(x_train1, K)
    pi2, u2, sigma2 = EM(x_train2, K)
    u1 = np.mat(u1)
    u2 = np.mat(u2)

    # 得到GMM模型
    for i in range(x_test.shape[0]):
        x_input = np.mat(x_test[i]).T
        if func_GMM(x_input, pi1, u1, sigma1, K) > func_GMM(x_input, pi2, u2, sigma2, K):
            y_testing[i] = 1
        else:
            y_testing[i] = 0

    print("测试集上的预测准确率为：", np.sum(y_test == y_testing) / y_test.shape)




