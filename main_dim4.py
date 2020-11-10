# coding:utf-8
import pandas as pd
import numpy as np
import torch
from torch import nn
import matplotlib.pyplot as plt

# 超参数
device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
lr = 0.01
batch_size = 4
num_epochs = 500
# 窗大小
window = 2


def data_cleaning():
    # 数据读取
    names = "统一编号	X	Y	地面高程	井口高程	地下水类型	1990年1月	1990年2月	1990年3月	1990年4月	1990年5月	1990年6月	1990年7月	1990年8月	1990年9月	1990年10月	1990年11月	1990年12月	1991年1月	1991年2月	1991年3月	1991年4月	1991年5月	1991年6月	1991年7月	1991年8月	1991年9月	1991年10月	1991年11月	1991年12月	1992年1月	1992年2月	1992年3月	1992年4月	1992年5月	1992年6月	1992年7月	1992年8月	1992年9月	1992年10月	1992年11月	1992年12月	1993年1月	1993年2月	1993年3月	1993年4月	1993年5月	1993年6月	1993年7月	1993年8月	1993年9月	1993年10月	1993年11月	1993年12月	1994年1月	1994年2月	1994年3月	1994年4月	1994年5月	1994年6月	1994年7月	1994年8月	1994年9月	1994年10月	1994年11月	1994年12月	1995年1月	1995年2月	1995年3月	1995年4月	1995年5月	1995年6月	1995年7月	1995年8月	1995年9月	1995年10月	1995年11月	1995年12月	1996年1月	1996年2月	1996年3月	1996年4月	1996年5月	1996年6月	1996年7月	1996年8月	1996年9月	1996年10月	1996年11月	1996年12月	1997年1月	1997年2月	1997年3月	1997年4月	1997年5月	1997年6月	1997年7月	1997年8月	1997年9月	1997年10月	1997年11月	1997年12月	1998年1月	1998年2月	1998年3月	1998年4月	1998年5月	1998年6月	1998年7月	1998年8月	1998年9月	1998年10月	1998年11月	1998年12月	1999年1月	1999年2月	1999年3月	1999年4月	1999年5月	1999年6月	1999年7月	1999年8月	1999年9月	1999年10月	1999年11月	1999年12月	2000年1月	2000年2月	2000年3月	2000年4月	2000年5月	2000年6月	2000年7月	2000年8月	2000年9月	2000年10月	2000年11月	2000年12月	2001年1月	2001年2月	2001年3月	2001年4月	2001年5月	2001年6月	2001年7月	2001年8月	2001年9月	2001年10月	2001年11月	2001年12月	2002年1月	2002年2月	2002年3月	2002年4月	2002年5月	2002年6月	2002年7月	2002年8月	2002年9月	2002年10月	2002年11月	2002年12月	2003年1月	2003年2月	2003年3月	2003年4月	2003年5月	2003年6月	2003年7月	2003年8月	2003年9月	2003年10月	2003年11月	2003年12月	2004年1月	2004年2月	2004年3月	2004年4月	2004年5月	2004年6月	2004年7月	2004年8月	2004年9月	2004年10月	2004年11月	2004年12月	2005年1月	2005年2月	2005年3月	2005年4月	2005年5月	2005年6月	2005年7月	2005年8月	2005年9月	2005年10月	2005年11月	2005年12月	2006年1月	2006年2月	2006年3月	2006年4月	2006年5月	2006年6月	2006年7月	2006年8月	2006年9月	2006年10月	2006年11月	2006年12月	2007年1月	2007年2月	2007年3月	2007年4月	2007年5月	2007年6月	2007年7月	2007年8月	2007年9月	2007年10月	2007年11月	2007年12月	2008年1月	2008年2月	2008年3月	2008年4月	2008年5月	2008年6月	2008年7月	2008年8月	2008年9月	2008年10月	2008年11月	2008年12月	2009年1月	2009年2月	2009年3月	2009年4月	2009年5月	2009年6月	2009年7月	2009年8月	2009年9月	2009年10月	2009年11月	2009年12月	2010年1月	2010年2月	2010年3月	2010年4月	2010年5月	2010年6月	2010年7月	2010年8月	2010年9月	2010年10月	2010年11月	2010年12月	2011年1月	2011年2月	2011年3月	2011年4月	2011年5月	2011年6月	2011年7月	2011年8月	2011年9月	2011年10月	2011年11月	2011年12月	2012年1月	2012年2月	2012年3月	2012年4月	2012年5月	2012年6月	2012年7月	2012年8月	2012年9月	2012年10月	2012年11月	2012年12月	2013年1月	2013年2月	2013年3月	2013年4月	2013年5月	2013年6月	2013年7月	2013年8月	2013年9月	2013年10月	2013年11月	2013年12月	2014年1月	2014年2月	2014年3月	2014年4月	2014年5月	2014年6月	2014年7月	2014年8月	2014年9月	2014年10月	2014年11月	2014年12月	2015年1月	2015年2月	2015年3月	2015年4月	2015年5月	2015年6月	2015年7月	2015年8月	2015年9月	2015年10月	2015年11月	2015年12月	2016年1月	2016年2月	2016年3月	2016年4月	2016年5月	2016年6月	2016年7月	2016年8月	2016年9月	2016年10月	2016年11月	2016年12月	2017年1月	2017年2月	2017年3月	2017年4月	2017年5月	2017年6月	2017年7月	2017年8月	2017年9月	2017年10月	2017年11月	2017年12月	2018年1月	2018年2月	2018年3月	2018年4月	2018年5月	2018年6月	2018年7月	2018年8月	2018年9月	2018年10月	2018年11月	2018年12月"
    col_names = names.split()
    data = pd.read_excel('data.xlsx', names=col_names)

    # 只要每个月地下水位高度数据
    data = data.iloc[:, 6:]
    # 保留至少包含28年数据的井，并对缺失值做线性插值处理
    data = data.dropna(thresh=12 * 27).interpolate(axis=1)

    # 归一化
    data_max = data.stack().max()
    data_min = data.stack().min()
    data_norm = (data - data_min) / (data_max - data_min)
    return data_norm


# 整理成有监督数据集
def split2dataset(data_raw, split_rate):
    m, n = data_raw.shape
    n_train = int(n * split_rate)

    train_x, train_y = [], []
    test_x, test_y = [], []

    for i in range(m):
        seq_train_x = []
        seq_train_y = []
        seq_test_x = []
        seq_test_y = []
        for j in range(n_train):
            if j + window > n_train:
                break
            temp = data_raw.iloc[i, j:j + window]
            seq_train_x.append(temp[:-1].values)
            seq_train_y.append(temp[-1])
        for k in range(n_train, n):
            if k + window > n:
                break
            temp = data_raw.iloc[i, k:k + window]
            seq_test_x.append(temp[:-1].values)
            seq_test_y.append(temp[-1])
        train_x.append(seq_train_x)
        train_y.append(seq_train_y)
        test_x.append(seq_test_x)
        test_y.append(seq_test_y)

    return train_x, train_y, test_x, test_y


# 定义网络模型
class LSTM(nn.Module):
    def __init__(self, input_size=window - 1, hidden_size=128, num_layers=2, output_size=1):
        super(LSTM, self).__init__()
        self.lstm = nn.LSTM(input_size=input_size, hidden_size=hidden_size, num_layers=num_layers)
        self.dropout = nn.Dropout(0.5)
        self.fc = nn.Sequential(
            nn.Linear(hidden_size, hidden_size),
            nn.Tanh(),
            nn.Linear(hidden_size, output_size)
        )

    def forward(self, x):
        out, _ = self.lstm(x)
        s, b, h = out.shape
        # print(out.shape)
        out = out.view(s * b, -1)
        out = self.dropout(out)
        out = self.fc(out)
        return out.view(s, b, -1)


# 定义训练过程
def train():
    print('training on', device)
    early_count = 0
    for epoch in range(num_epochs + 1):
        train_l_sum, batch_count = 0.0, 0
        net.train()
        for X, y in train_dataloader:
            X = X.reshape(X.shape[1], X.shape[0], -1)
            y = y.reshape(y.shape[1], y.shape[0], -1)
            X, y = X.to(device), y.to(device)
            # print(X.shape,y.shape)
            # print('y', y)
            y_hat = net(X)
            # print('y_hat', y_hat.shape)
            # print('y_hat', y_hat)
            l = criterion(y_hat, y)
            optimizer.zero_grad()
            l.backward()
            optimizer.step()

            train_l_sum += l.item()
            batch_count += 1
        # 出现10次loss<0.001，则提前结束训练
        if train_l_sum / batch_count < 0.001:
            early_count += 1
            if early_count >= 10:
                print('early stopping')
                break
        if epoch % 10 == 0:
            print('epoch %d,train_loss %.4f' % (epoch, train_l_sum / batch_count))


def valid():
    net.eval()

    i = 0
    for X, y in test_dataloader:
        i += 1
        real = np.array([[0]] * X.shape[0])
        prediction = np.array([[0]] * X.shape[0])

        X = X.reshape(X.shape[1], X.shape[0], -1)
        y = y.reshape(y.shape[1], y.shape[0], -1)
        X, y = X.to(device), y.to(device)
        y_hat = net(X)
        loss = criterion(y_hat, y).item()
        print('%d/%d MSELoss' % (i, (n_rows + 5) / batch_size), loss)
        y = y.reshape(y.shape[1], y.shape[0])
        y = y.cpu().data.numpy()
        real = np.concatenate((real, y), axis=1)
        y_hat = y_hat.reshape(y_hat.shape[1], y_hat.shape[0])
        y_hat = y_hat.cpu().data.numpy()
        prediction = np.concatenate((prediction, y_hat), axis=1)

        for index in range(real.shape[0]):
            plt.plot(prediction[index, 1:].T, 'r', label='prediction' if index == 0 else '')
            plt.plot(real[index, 1:].T, 'b', label='real' if index == 0 else '')

        plt.title('%d/%d MSELoss= %f' % (i, (n_rows + 5) / batch_size, loss))
        plt.legend(labels=('prediction', 'real'))
        plt.legend(loc='best', numpoints=1)
        plt.savefig('images/window%d_dim4_%d.jpg' % (window, i))
        plt.show()
        plt.figure()


if __name__ == '__main__':
    # 是否使用已保存的模型，默认路径为./model.pth
    use_saved_model = True
    saved_model_path = './model_window' + str(window) + '_dim4.pt'

    # 预处理后的数据
    data_norm = data_cleaning()
    print('保留的数据大小', data_norm.shape)
    print('保留的地下水井index', data_norm.index.values)
    # 共有多少口井被保留
    n_rows = data_norm.shape[0]

    # 划分数据集（训练集0.7，测试集0.3）
    split_rate = 0.7
    train_x, train_y, test_x, test_y = split2dataset(data_norm, split_rate=split_rate)
    train_x = torch.tensor(train_x, dtype=torch.float)
    train_y = torch.tensor(train_y, dtype=torch.float)
    test_x = torch.tensor(test_x, dtype=torch.float)
    test_y = torch.tensor(test_y, dtype=torch.float)

    print('train_x.shape:', train_x.shape, 'train_y.shape:', train_y.shape)
    print('test_x.shape:', test_x.shape, 'test_y.shape:', test_y.shape)

    # 加载数据集
    train_dataset = torch.utils.data.TensorDataset(train_x, train_y)
    train_dataloader = torch.utils.data.DataLoader(train_dataset, batch_size=batch_size)
    test_dataset = torch.utils.data.TensorDataset(test_x, test_y)
    test_dataloader = torch.utils.data.DataLoader(test_dataset, batch_size=batch_size)

    # 创建网络
    net = LSTM().to(device)
    # 损失函数 L2 Loss
    criterion = nn.MSELoss()
    # 优化器
    optimizer = torch.optim.Adam(params=net.parameters(), lr=lr)

    # 训练
    if not use_saved_model:
        train()
        torch.save(net.state_dict(), saved_model_path)
    else:
        net.load_state_dict(torch.load(saved_model_path, map_location=torch.device(device)))
        # print(net)

    # 验证
    valid()
