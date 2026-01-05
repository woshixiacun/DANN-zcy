import pickle
from joblib import dump, load
import torch


def make_data_labels(dataframe, set):
    '''
        参数 dataframe: 数据框
        返回 x_data: 数据集     torch.tensor
            y_label: 对应标签值  torch.tensor
    '''
    # 信号值
    x_data = dataframe.iloc[:,0:-1]
    # 标签值
    y_label = dataframe.iloc[:,-1]
    # x_data = torch.tensor(x_data.values).float()
    # y_label = torch.tensor(y_label.values.astype('int64')) # 指定了这些张量的数据类型为64位整数，通常用于分类任务的类别标签

    x_data = x_data.values
    y_label = y_label.values

    data_infos = []
    for i in range(len(x_data)):
        temp_dict = {}
        temp_dict["value"] = x_data[i]
        temp_dict["label"] = y_label[i]

        data_infos.append(temp_dict)

    with open(f"dataset/1Ddata_convet_zcy/{set}.pkl", "wb") as f:
        pickle.dump(data_infos, f)

    # return x_data, y_label

# 加载数据
train_set = load('/mnt/d/Study_File/codezcy/DANN-zcy-main/dataset/1Ddata/train_set') 
val_set = load('/mnt/d/Study_File/codezcy/DANN-zcy-main/dataset/1Ddata/val_set') 
test_set = load('/mnt/d/Study_File/codezcy/DANN-zcy-main/dataset/1Ddata/test_set') 

# 制作标签
make_data_labels(train_set, "train")
make_data_labels(val_set, "val")
make_data_labels(test_set, "test")
