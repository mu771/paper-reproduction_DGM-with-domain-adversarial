import torch.utils.data as data
import torch
import scipy.io
import numpy as np

class GetLoader(data.Dataset):
    def __init__(self, data, label, environment_label, envir_onehot, domain_flag, domain_onehot):
        data = torch.from_numpy(data)
        label = torch.from_numpy(label)
        data = data.to(torch.float32)
        label = label.to(torch.float32)

        self.data = data.reshape(-1, 1, 8, 50)
        self.label = label.reshape(-1, 2)
        self.environment_label_one_hot = envir_onehot.reshape(-1, 6)
        self.environment_label = environment_label.reshape(-1, 1)
        self.domain_flag = domain_flag.reshape(-1, 1)
        self.domian_onehot = domain_onehot.reshape(-1, 2)
        self.len = self.data.shape[0]

    def __getitem__(self, indx):
        return self.data[indx], self.label[indx], self.environment_label[indx], self.domain_flag[indx]

    def __len__(self):
        return self.len

def LoadData(root, envir_label, domain_flag):
    data = scipy.io.loadmat(root)
    
    cir = np.abs(data['cir_data'][:, 0:50])
    label = data['label']
    label = np.repeat(label,3,axis=0)
    environment_label = np.full((int(cir.shape[0] / 8), 1), envir_label)
    envir_onehot = np.full((int(cir.shape[0] / 8), 6), 0)
    for i in range(int(cir.shape[0] / 8)):
        envir_onehot[i] = np.eye(6)[:, envir_label-1].reshape(-1, 1).squeeze()
        #print(environment_label[i])
    domain_flag_1 = np.full((int(cir.shape[0] / 8), 1), domain_flag)
    domain_onehot = np.full((int(cir.shape[0] / 8), 2), 0)
    for i in range(int(cir.shape[0] / 8)):
        domain_onehot[i] = np.eye(2)[:, domain_flag].reshape(-1, 1).squeeze()
        #print(environment_label[i])

    data = GetLoader(cir, label, environment_label, envir_onehot, domain_flag_1, domain_onehot)
    cir = data.data
    label = data.label
    envir_one_hot = data.environment_label_one_hot
    environment_label = data.environment_label
    domain_flag = data.domain_flag
    domain_onehot = data.domian_onehot

    return cir, label, environment_label, envir_one_hot, domain_flag, domain_onehot

def DataMerge():
    source_root_list = ['D:\\UWB-dataset\\material_CIR\\LOS\\LOS_data.mat',
                        'D:\\UWB-dataset\\material_CIR\\glass\\glass_data100.mat',
                        'D:\\UWB-dataset\\material_CIR\\wood\\wood_data100.mat',
                        'D:\\UWB-dataset\\material_CIR\\human\\human_data100.mat',
                        'D:\\UWB-dataset\\material_CIR\\baffle\\baffle_data.mat'
                        ]
    target_root_list = [
                        'D:\\UWB-dataset\\material_CIR\\stone\\stone_data100.mat'
                        ]

    source_cir_list = []
    source_label_list = []
    source_environment_label_list = []
    sour_envir_one_hot_list = []
    source_domain_list = []
    sour_domain_oh_list = []

    target_cir_list = []
    target_label_list = []
    target_environment_label_list = []
    tar_envir_one_hot_list = []
    target_domain_list = []
    tar_domain_oh_list = []
    # 循环遍历三个MAT文件的路径
    for i in range(4):
        source_root = source_root_list[i]
        target_root = target_root_list[0]
        
        # 调用LoadData()函数加载数据
        source_cir, source_label, source_environment_label, source_envir_one_hot, source_domain, sour_domian_oh = \
            LoadData(source_root, i + 1, 0)
        target_cir, target_label, target_environment_label, tar_envir_one_hot, target_domain, tar_domain_oh = \
            LoadData(target_root, 6, 1)

        #对原始cir进行归一化
        normalized_source_cir = torch.zeros_like(source_cir)  # 创建一个与原始数据相同形状的张量来存储归一化后的数据
        normalized_target_cir = torch.zeros_like(target_cir)

        for i in range(source_cir.shape[0]):  # 遍历每个样本
            sample = source_cir[i]
            min_val = torch.min(sample)
            max_val = torch.max(sample)
            normalized_sample = (sample - min_val) / (max_val - min_val)
            normalized_source_cir[i] = normalized_sample
        
        for i in range(target_cir.shape[0]):  # 遍历每个样本
            sample = target_cir[i]
            min_val = torch.min(sample)
            max_val = torch.max(sample)
            normalized_sample = (sample - min_val) / (max_val - min_val)
            normalized_target_cir[i] = normalized_sample

        # 将加载得到的数据存储到列表中
        source_cir_list.append(normalized_source_cir)
        source_label_list.append(source_label)
        source_environment_label_list.append(source_environment_label)
        sour_envir_one_hot_list.append(source_envir_one_hot)
        source_domain_list.append(source_domain)
        sour_domain_oh_list.append(sour_domian_oh)

        target_cir_list.append(normalized_target_cir)
        target_label_list.append(target_label)
        target_environment_label_list.append(target_environment_label)
        tar_envir_one_hot_list.append(tar_envir_one_hot)
        target_domain_list.append(target_domain)
        tar_domain_oh_list.append(tar_domain_oh)
    
    # 合并数据
    source_cir_merged = np.concatenate(source_cir_list, axis=0)
    source_label_merged = np.concatenate(source_label_list, axis=0)
    source_environment_label_merged = np.concatenate(source_environment_label_list, axis=0)
    sour_envir_oh_merged = np.concatenate(sour_envir_one_hot_list, axis=0)
    source_domain_merged = np.concatenate(source_domain_list, axis=0)
    sour_domain_oh_merged = np.concatenate(sour_domain_oh_list, axis=0)

    target_cir_merged = np.concatenate(target_cir_list, axis=0)
    target_label_merged = np.concatenate(target_label_list, axis=0)
    target_environment_label_merged = np.concatenate(target_environment_label_list, axis=0)
    tar_envir_oh_merged = np.concatenate(tar_envir_one_hot_list, axis=0)
    target_domain_merged = np.concatenate(target_domain_list, axis=0)
    tar_domain_oh_merged = np.concatenate(tar_domain_oh_list, axis=0)

    # 返回合并后的数据集
    return (source_cir_merged, source_label_merged, source_environment_label_merged, 
            sour_envir_oh_merged, source_domain_merged, sour_domain_oh_merged, 
            target_cir_merged, target_label_merged, target_environment_label_merged, 
            tar_envir_oh_merged, target_domain_merged, tar_domain_oh_merged)
