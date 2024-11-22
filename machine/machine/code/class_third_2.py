import os
import torch
import numpy as np
from matplotlib import pyplot as plt
from torch.testing._internal.data.network1 import Net
from torch.utils.data import DataLoader, TensorDataset

from pythonProject.machine.code.class_third_1 import draw

batch_size = 64
learning_rate = 0.0001
EPOCHS = 10


def main_predict(data_path, model_path):
    # 加载测试数据
    test_img = torch.FloatTensor(np.load(data_path))

    # 加载模型权重
    model.load_state_dict(torch.load(model_path, weights_only=True))

    # 转换数据
    data = test_img.squeeze().numpy()

    # 使用模型进行预测
    with torch.no_grad():
        output = model(test_img)

    # 获取预测标签
    _predict_label = torch.max(output, dim=1)
    predict_label = _predict_label[1].detach().numpy()  # 获取标签并转换为 numpy 数组

    # 打印数据和预测标签
    print(data.shape)
    print(predict_label)

    # 可视化结果
    draw(data, predict_label)


def main_train():
    # 使用相对路径加载训练数据和标签
    train_img = torch.FloatTensor(np.load('../data/mnist_npy_new/mnist_npy/train_data.npy'))
    test_img = torch.FloatTensor(np.load('../data/mnist_npy_new/mnist_npy/test_data.npy'))
    train_label = torch.FloatTensor(np.load('../data/mnist_npy_new/mnist_npy/train_label.npy'))
    test_label = torch.FloatTensor(np.load('../data/mnist_npy_new/mnist_npy/test_label.npy'))

    # 创建数据集
    train_dataset = TensorDataset(train_img, train_label)
    test_dataset = TensorDataset(test_img, test_label)

    # 输出数据集的形状
    print(train_label.shape)
    print(test_label.shape)

    # 创建数据加载器
    train_loader = DataLoader(train_dataset, batch_size=batch_size, shuffle=True)
    test_loader = DataLoader(test_dataset, batch_size=batch_size, shuffle=False)

    # 初始化测试集的准确率列表
    acc_list_test = []

    # 训练和测试循环
    for epoch in range(EPOCHS):
        train(epoch, train_loader)  # 训练模型
        acc_test = test(epoch, test_loader)  # 测试模型
        acc_list_test.append(acc_test)  # 记录每个epoch的测试集准确率

    # 绘制准确率曲线
    plt.plot(acc_list_test)
    plt.xlabel('Epoch')
    plt.ylabel('Accuracy on Testset')
    plt.show()


def test(epoch, test_loader):
    correct = 0
    total = 0

    with torch.no_grad():
        for data in test_loader:
            images, labels = data
            outputs = model(images)
            predicted = torch.max(outputs.data, dim=1)
            ground_truth = torch.max(labels.data, dim=1)

            total += labels.size(0)
            correct += (predicted == ground_truth).sum().item()

    acc = correct / total
    print('[%d/%d]: Accuracy on test set: %.1f%%' % (epoch + 1, EPOCHS, 100 * acc))

    os.makedirs('CNN_model', exist_ok=True)
    torch.save(model.state_dict(), 'CNN_model/' + 'epoch=' + str(epoch) + '.pth')

    return acc


model = Net()
criterion = torch.nn.CrossEntropyLoss()
optimizer = torch.optim.Adam(model.parameters(), lr=learning_rate)


def train(epoch, train_loader):
    running_loss = 0.0
    running_total = 0
    running_correct = 0

    for batch_idx, data in enumerate(train_loader):  # 遍历数据加载器
        inputs, target = data

        optimizer.zero_grad()  # 清零梯度
        outputs = model(inputs)  # 前向传播
        loss = criterion(outputs, target)  # 计算损失
        loss.backward()  # 反向传播
        optimizer.step()  # 更新参数

        running_loss += loss.item()  # 累加损失
        predicted = torch.max(outputs.data, dim=1)  # 预测结果
        _ground_truth = torch.max(target, dim=1)  # 获取真实标签

        running_total += inputs.shape[0]  # 累加样本数量
        running_correct += (predicted == _ground_truth).sum().item()  # 累加正确预测数量

        if batch_idx % 300 == 299:  # 每300个batch打印一次
            print('[%d, %5d]: loss: %.3f acc: %.2f%%' %
                  (epoch + 1, batch_idx + 1, running_loss / 300, 100 * running_correct / running_total))

            # 重置计数器
            running_loss = 0.0
            running_total = 0
            running_correct = 0


if __name__ == "__main__":
    main_train()
