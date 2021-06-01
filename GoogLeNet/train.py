import torch
import torch.nn as nn
from torchvision import transforms, datasets
import torchvision
import json
import matplotlib.pyplot as plt
import os
import torch.optim as optim
from model import GoogLeNet

device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
print(device)

data_transform = {
    "train": transforms.Compose([transforms.RandomResizedCrop(224),    # 随机剪裁
                                 transforms.RandomHorizontalFlip(),    # 随机水平翻转
                                 transforms.ToTensor(),         # 转换为tensor
                                 # ToTensor()的官方解释：Converts a PIL Image or numpy.ndarray (H x W x C) in the range
                                 #     [0, 255] to a torch.FloatTensor of shape (C x H x W) in the range [0.0, 1.0]
                                 transforms.Normalize((0.5, 0.5, 0.5), (0.5, 0.5, 0.5))]),   # 标准化，均值0，标准差1
    "val": transforms.Compose([transforms.Resize((224, 224)),
                               transforms.ToTensor(),
                               transforms.Normalize((0.5, 0.5, 0.5), (0.5, 0.5, 0.5))])}


data_dir='.'
train_dataset = datasets.ImageFolder(os.path.join(data_dir, 'train'),
                                     transform=data_transform["train"])
batch_size = 20
# DataLoader会把数据分成一批一批的，里面还有一些其他的设置参数
train_loader = torch.utils.data.DataLoader(train_dataset,
                                           batch_size=batch_size, shuffle=True,
                                           num_workers=0)
# 验证集的处理方式类似
validate_dataset = datasets.ImageFolder(os.path.join(data_dir, 'valid'),
                                        transform=data_transform["val"])

val_num = len(validate_dataset)
validate_loader = torch.utils.data.DataLoader(validate_dataset,
                                              batch_size=batch_size, shuffle=False,
                                              num_workers=0)

# test_data_iter = iter(validate_loader)
# test_image, test_label = test_data_iter.next()

# net = torchvision.models.googlenet(num_classes=5)
# model_dict = net.state_dict()
# pretrain_model = torch.load("googlenet.pth")
# del_list = ["aux1.fc2.weight", "aux1.fc2.bias",
#             "aux2.fc2.weight", "aux2.fc2.bias",
#             "fc.weight", "fc.bias"]
# pretrain_dict = {k: v for k, v in pretrain_model.items() if k not in del_list}
# model_dict.update(pretrain_dict)
# net.load_state_dict(model_dict)
# 实例化模型
net = GoogLeNet(num_classes=40, aux_logits=True, init_weights=True)
net.to(device)
# 定义损失函数，定义优化器
loss_function = nn.CrossEntropyLoss()
optimizer = optim.Adam(net.parameters(), lr=0.0003)
# 最佳准确率
best_acc = 0.0
# 保存权重的路径
save_path = './googleNet.pth'
for epoch in range(30):
    # train
    net.train()
    # 每个epoch清零损失
    running_loss = 0.0
    # 迭代器里面有索引和数据
    for step, data in enumerate(train_loader, start=0):
        # 数据是一个元组，有图像和标签
        images, labels = data
        # 每一个batch梯度清零
        optimizer.zero_grad()
        # 将图片输入网络，得到主分类器和两个辅助分类器的结果
        logits, aux_logits2, aux_logits1 = net(images.to(device))
        # 计算三个损失
        # 这里的loss是每个batch中的平均loss
        loss0 = loss_function(logits, labels.to(device))
        loss1 = loss_function(aux_logits1, labels.to(device))
        loss2 = loss_function(aux_logits2, labels.to(device))
        # 三个损失加权求和
        loss = loss0 + loss1 * 0.3 + loss2 * 0.3
        # 反向传播求导
        loss.backward()
        # 沿梯度方向下降
        optimizer.step()

        # print statistics
        # 累加每个batch的损失
        running_loss += loss.item()
        # print train process
        rate = (step + 1) / len(train_loader)
        a = "*" * int(rate * 50)
        b = "." * int((1 - rate) * 50)
        print("\rtrain loss: {:^3.0f}%[{}->{}]{:.3f}".format(int(rate * 100), a, b, loss), end="")
    print()

    # validate
    net.eval()
    acc = 0.0  # accumulate accurate number / epoch
    with torch.no_grad():   # 不再计算梯度
        for data_test in validate_loader:
            test_images, test_labels = data_test
            # 只有主分类器的结果
            outputs = net(test_images.to(device))  # eval model only have last output layer
            predict_y = torch.max(outputs, dim=1)[1]
            # 计算所有预测正确的个数
            acc += (predict_y == test_labels.to(device)).sum().item()
        # 准确率
        accurate_test = acc / val_num
        # 如果准确率为历史最高值，就保存这次的训练结果
        if accurate_test > best_acc:
            torch.save(net.state_dict(), save_path)
            best_acc = accurate_test      # 更新最佳准确率
        # 打印一次epoch之后的训练集平均损失和验证集的准确率
        print('[epoch %d] train_loss: %.3f  test_accuracy: %.3f' %
              (epoch + 1, running_loss / step, accurate_test))

print('Finished Training')

