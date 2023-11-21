import torch
import torch.nn.functional as F
from torch.utils.data import DataLoader
from torch.optim import Adam
from model import DGM  
#from dataload import LoadData
from dataload_new import DataMerge
from dataload_2 import LoadData
import argparse
import torch.nn as nn
import torch.utils.data as data
from torch.utils.data.dataset import random_split

def Data_new(args):
    batch_size = args.batch_size
    source_cir, source_label, source_envir_label, sour_envir_onehot, source_domain, sour_domain_oh, \
        target_cir, target_label, target_environment_label, tar_envir_onehot, target_domain, tar_domain_oh = DataMerge()
    #source_cir, source_label, source_envir_label, source_domain, \
    #    target_cir, target_label, target_environment_label, target_domain = LoadData()

    # 创建 TensorDataset
    #将数据转换为张量的形式
    source_cir = torch.tensor(source_cir, dtype=torch.float32)
    source_label = torch.tensor(source_label)
    source_envir_label = torch.tensor(source_envir_label)
    sour_envir_onehot = torch.tensor(sour_envir_onehot)
    source_domain = torch.tensor(source_domain)
    sour_domain_oh = torch.tensor(sour_domain_oh)
    dataset_source = data.TensorDataset(source_cir, source_label, source_envir_label, 
                                        sour_envir_onehot, source_domain, sour_domain_oh)

    target_cir = torch.tensor(target_cir)
    target_label = torch.tensor(target_label)
    target_envir_label = torch.tensor(target_environment_label)
    tar_envir_onehot = torch.tensor(tar_envir_onehot)
    target_domain = torch.tensor(target_domain)
    tar_domain_oh = torch.tensor(tar_domain_oh)
    dataset_target = data.TensorDataset(target_cir, target_label, target_envir_label, 
                                        tar_envir_onehot, target_domain, tar_domain_oh)

    # 定义训练集和验证集的比例
    train_ratio = 0.8
    train_ratio_tar = 0.4 #用目标域40%的数据参与训练

    #源域划分训练集和验证集
    train_size_sour = int(train_ratio * len(dataset_source))
    val_size_sour = len(dataset_source) - train_size_sour

    train_dataset_sour, val_dataset_sour = random_split(dataset_source, [train_size_sour, val_size_sour])
    train_loader_sour = data.DataLoader(train_dataset_sour, batch_size=batch_size, shuffle=True)
    val_loader_sour = data.DataLoader(val_dataset_sour, batch_size=batch_size, shuffle=True)

    #目标域划分训练集和验证集
    a = len(dataset_target)
    tards_lenght = int(train_ratio_tar * len(dataset_target))
    b=a-tards_lenght
    dataset_target_new, ds_target_test = random_split(dataset_target, [tards_lenght, b])
    train_size_tar = int(train_ratio * len(dataset_target_new))
    val_size_tar =tards_lenght - train_size_tar

    train_dataset_tar, val_dataset_tar = random_split(dataset_target_new, [train_size_tar, val_size_tar])
    train_loader_tar = data.DataLoader(train_dataset_tar, batch_size=batch_size, shuffle=True)
    val_loader_tar = data.DataLoader(val_dataset_tar, batch_size=batch_size, shuffle=True)
    test_loader_tar = data.DataLoader(ds_target_test, batch_size=batch_size, shuffle=True)

    return train_loader_sour, val_loader_sour, train_loader_tar, val_loader_tar, test_loader_tar

def DGM_train_no_envir(model,train_loader_sour,eval_loader_sour,args):
    print('start DGM training without envir')
    criterion = nn.MSELoss()
    optimizer = Adam(model.parameters(), lr=args.learning_rate)

    losses = []
    accuracies_envir = []
    accuracies_dis = []

    for epoch in range(args.epochs):
        model.train()
        total_loss = 0.0

        for (inputs_source, label_source, _, _, _, _) in train_loader_sour:

            inputs_source = inputs_source.to(device)
            label_source = label_source.to(device)
            
            optimizer.zero_grad()

            #计算源域的loss
            reconstructe_signals, predict_labels, distances, _ = model(inputs_source)
            predict_labels = predict_labels.float()
            loss_1 = criterion(reconstructe_signals, inputs_source)
            #loss_2 = criterion(predict_labels, envir_label_source)
            loss_3 = criterion(distances, label_source)
            loss = loss_1 + loss_3
            loss.backward()
            optimizer.step()

            total_loss += loss.item()

        avg_loss = total_loss / len(train_loader_sour)
        losses.append(avg_loss)

        # Evaluation on validation set
        if (epoch + 1) % args.val_interval == 0:
            accuracy_envir, accuracy_dis, accuracy_domain = evaluate(model, eval_loader_sour,device)
            accuracies_envir.append(accuracy_envir)
            accuracies_dis.append(accuracy_dis)
            print(f"Epoch {epoch + 1}: Loss = {avg_loss:.3f}, "
                    f"dis error = {accuracy_dis:.3f}")
            
    torch.save(model.state_dict(), "model_weights_reconstruction_distance.pt")

def DGM_train_with_envir(model,train_loader_sour,eval_loader_sour,args):
    print('start DGM training with envir')
    model.load_state_dict(torch.load("model_weights_reconstruction_distance.pt"))

    # 解冻预测环境标签误差的子模块的参数，将其设置为可训练状态
    model.train()
    for name, param in model.named_parameters():
        param.requires_grad = False
    for param in model.environment_classifier.parameters():
        param.requires_grad = True
    #for name, param in model.named_parameters():
    #    print(f'{name}: requires_grad={param.requires_grad}')

    criterion = nn.CrossEntropyLoss()
    optimizer = Adam(filter(lambda p: p.requires_grad, model.parameters()), lr=args.learning_rate_1)

    losses = []
    accuracies_envir = []
    accuracies_dis = []

    for epoch in range(args.epochs):
        total_loss = 0.0
        #model.eval()
        #model.environment_classifier.train()
        for (inputs_source, label_source, envir_label_source, envir_onehot, _, _) in train_loader_sour:
            inputs_source = inputs_source.to(device)
            label_source = label_source.to(device)
            #envir_label_source = envir_label_source.to(device)
            envir_onehot = envir_onehot.to(device)
            
            optimizer.zero_grad()

            #计算源域的loss
            reconstructe_signals, predict_labels, distances, domain_out_sour = model(inputs_source)
            predict_labels = predict_labels.float()
            envir_onehot = envir_onehot.float()
            loss_2 = criterion(predict_labels, envir_onehot)
            loss = loss_2
           
            loss.backward()
            optimizer.step()

            total_loss += loss.item()

        avg_loss = total_loss / len(train_loader_sour)
        losses.append(avg_loss)

        # Evaluation on validation set
        if (epoch + 1) % args.val_interval == 0:
            accuracy_envir, accuracy_dis, accuracy_domain = evaluate(model, eval_loader_sour,device)
            accuracies_envir.append(accuracy_envir)
            accuracies_dis.append(accuracy_dis)
            print(f"Epoch {epoch + 1}: Loss = {avg_loss:.3f}, "
                    f"Accuracy envir = {accuracy_envir:.3f}, "
                    f"dis error = {accuracy_dis:.3f}")
            
    torch.save(model.state_dict(), "model_weights_DGM.pt")

def DGM_train(model,train_loader_sour,eval_loader_sour,args):
    print('start DGM training')
    criterion = nn.MSELoss()
    optimizer = Adam(model.parameters(), lr=args.learning_rate)

    losses = []
    accuracies_envir = []
    accuracies_dis = []

    for epoch in range(args.epochs):
        model.train()
        total_loss = 0.0

        for (inputs_source, label_source, envir_label_source, _) in train_loader_sour:

            inputs_source = inputs_source.to(device)
            label_source = label_source.to(device)
            envir_label_source = envir_label_source.to(device)
            #domain_sour = domain_sour.to(device)

            optimizer.zero_grad()

            #计算源域的loss
            reconstructe_signals, predict_labels, distances, domain_out_sour = model(inputs_source)
            predict_labels = predict_labels.float()
            loss_1 = criterion(reconstructe_signals, inputs_source)
            loss_2 = criterion(predict_labels, envir_label_source)
            loss_3 = criterion(distances, label_source)
            loss = loss_1 + loss_2 + loss_3
            loss.backward()
            optimizer.step()

            total_loss += loss.item()

        avg_loss = total_loss / len(train_loader_sour)
        losses.append(avg_loss)

        # Evaluation on validation set
        if (epoch + 1) % args.val_interval == 0:
            accuracy_envir, accuracy_dis, accuracy_domain = evaluate(model, eval_loader_sour,device)
            accuracies_envir.append(accuracy_envir)
            accuracies_dis.append(accuracy_dis)
            print(f"Epoch {epoch + 1}: Loss = {avg_loss:.3f}, "
                    f"Accuracy envir = {accuracy_envir:.3f}, "
                    f"dis error = {accuracy_dis:.3f}")

def train(model,train_loader_sour,eval_loader_sour,train_loader_tar,eval_loader_tar,args):
    print('start training')
    #model.load_state_dict(torch.load("model_weights_DGM.pt"))
    criterion = nn.MSELoss()
    criterion_1 = nn.CrossEntropyLoss()
    optimizer = Adam(model.parameters(), lr=args.learning_rate)

    losses = []
    accuracies_envir = []
    accuracies_dis = []

    for epoch in range(args.epochs):
        model.train()
        total_loss = 0.0
        loss_domain_sour = 0.0
        loss_domain_tar = 0.0

        for (inputs_source, label_source, envir_label_source, envir_oh_sour, domain_sour, domain_sour_oh), \
            (inputs_target, _, envir_label_target, envir_oh_tar, domain_tar, domain_tar_oh) in \
            zip(train_loader_sour, train_loader_tar):

            inputs_source = inputs_source.to(device)
            label_source = label_source.to(device)
            envir_oh_sour = envir_oh_sour.to(device)
            domain_sour_oh = domain_sour_oh.to(device)

            inputs_target = inputs_target.to(device)
            envir_oh_tar = envir_oh_tar.to(device)
            domain_tar_oh = domain_tar_oh.to(device)

            optimizer.zero_grad()

            #计算源域的loss
            reconstructe_signals, predict_labels, distances, domain_out_sour = model(inputs_source)
            predict_labels = predict_labels.float()
            loss_1 = criterion(reconstructe_signals, inputs_source)
            envir_oh_sour = envir_oh_sour.float()
            loss_2 = criterion_1(predict_labels, envir_oh_sour)
            loss_3 = criterion(distances, label_source)
            
            loss_source = loss_1 + loss_3
            loss_source_1 = loss_1 + loss_2 + 2*loss_3

            #计算目标域的loss
            reconstructe_signals_tar, predict_labels_tar, _, domain_out_tar = model(inputs_target)
            predict_labels_tar = predict_labels_tar.float()
            loss_t1 = criterion(reconstructe_signals_tar, inputs_target)
            envir_oh_tar = envir_oh_tar.float()
            loss_t2 = criterion_1(predict_labels_tar, envir_oh_tar)
            
            loss_target = loss_t1
            loss_target_1 = loss_t1 + loss_t2

            #计算域分类器的loss
            domain_sour_oh = domain_sour_oh.float()
            domain_tar_oh = domain_tar_oh.float()
            a = criterion_1(domain_out_sour, domain_sour_oh)
            b = criterion_1(domain_out_tar, domain_tar_oh)
            #print(a)
            #print(b)
            loss_domain = criterion_1(domain_out_sour, domain_sour_oh) + criterion_1(domain_out_tar, domain_tar_oh)

            #计算总的loss
            if (epoch + 1) % args.loss_change_interval == 0:
                loss = loss_source_1 + loss_target_1 + args.lambda_value * loss_domain
            else:
                loss = loss_source + loss_target + args.lambda_value * loss_domain
            loss.backward()
            optimizer.step()

            total_loss += loss.item()
            loss_domain_sour += a.item()
            loss_domain_tar += b.item()

        avg_loss = total_loss / (len(train_loader_sour) + len(train_loader_tar))
        loss_domain_sour_avg = loss_domain_sour / (len(train_loader_sour) + len(train_loader_tar))
        loss_domain_tar_avg = loss_domain_tar / (len(train_loader_sour) + len(train_loader_tar))
        losses.append(avg_loss)

        # Evaluation on validation set
        if (epoch + 1) % args.val_interval == 0:
            eval_loader = []
            eval_loader.extend(eval_loader_sour)
            eval_loader.extend(eval_loader_tar)
            accuracy_envir, dis_rmse, dis_MAE, accuracy_domain = evaluate(model, eval_loader_tar, device)
            accuracies_envir.append(accuracy_envir)
            accuracies_dis.append(dis_rmse)
            print(f"Epoch {epoch + 1}: Loss = {avg_loss:.3f}, "
                  f"domain loss source = {loss_domain_sour_avg:.3f}, "
                  f"domain loss target = {loss_domain_tar_avg:.3f}, "
                    f"Accuracy envir = {accuracy_envir:.3f}, "
                    f"Accuracy domain = {accuracy_domain:.3f}, "
                    f"dis error = {dis_rmse:.3f}")

def evaluate(model, eval_loader, device):
    model.eval()  # 设置模型为评估模式
    correct_envir_label = 0
    correct_domian = 0
    total_distance_error = 0
    total_dis_mae = 0
    total = 0

    with torch.no_grad():
        for inputs, label, envir_label, envir_onehot, domain_flag, domain_flag_oh in eval_loader:
            inputs = inputs.to(device)
            label = label.to(device)
            envir_label = envir_label.to(device)

            envir_label = torch.squeeze(envir_label)
            domain_flag = torch.squeeze(domain_flag)

            reconstructe_signals, predict_labels, distances, domain_out = model(inputs)
            environment_label = torch.argmax(predict_labels, dim=-1) + 1
            domain_result = torch.argmax(domain_out, dim=-1)

            # 进行标签预测
            correct_envir_label += torch.eq(envir_label, environment_label).sum().item()
            correct_domian += torch.eq(domain_flag, domain_result).sum().item()

            distance_error = torch.sqrt(torch.sum((label - distances) ** 2, dim=1))
            dis_mae = torch.mean(torch.abs(label - distances))
            total_distance_error += distance_error.sum().item()
            total_dis_mae += dis_mae.sum().item()
            total += inputs.size(0)

    accuracy_envir = correct_envir_label / total
    accuracy_domain = correct_domian / total
    dis_rmse = total_distance_error / total
    dis_MAE = total_dis_mae / total
    return accuracy_envir, dis_rmse, dis_MAE, accuracy_domain

def main(args):
    model = DGM(hidden_1=args.hidden_dimension1, hidden_2=args.hidden_dimension2).to(device)
    train_loader_sour, eval_loader_sour, train_loader_tar, eval_loader_tar, test_loader_tar = Data_new(args=args)
    #DGM_train_no_envir(model,train_loader_sour,eval_loader_sour,args)
    #DGM_train_with_envir(model,train_loader_sour,eval_loader_sour,args)
    '''
    accuracy_envir, accuracy_dis, accuracy_domain = evaluate(model, eval_loader_tar, device)
    print("未进行域对抗训练，在目标域上的准确率 "
            f"Accuracy envir = {accuracy_envir:.3f}, "
            f"dis error = {accuracy_dis:.3f}")
    '''
    train(model,train_loader_sour,eval_loader_sour,train_loader_tar,eval_loader_tar,args)
    accuracy_envir, dis_rmse, dis_mae, accuracy_domain = evaluate(model, test_loader_tar, device)
    print("进行域对抗训练，在目标域上的准确率 "
            f"Accuracy envir = {accuracy_envir:.3f}, "
            f"Accuracy domain = {accuracy_domain:.3f}, "
            f"dis RMSE = {dis_rmse:.3f}, "
            f"dis MAE = {dis_mae:.3f}")

if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('--batch_size', type=int, default=17, help='training batch size')
    parser.add_argument('--epochs', type=int, default=30, help='number of training epochs')
    parser.add_argument('--learning_rate', type=float, default=0.00001, help='learning rate')
    parser.add_argument('--learning_rate_1', type=float, default=0.0003, help='learning rate')
    parser.add_argument('--val_interval', type=int, default=1, help='intervals of validation')
    parser.add_argument('--loss_change_interval', type=int, default=2, help='intervals of training with envir')
    parser.add_argument('--lambda_value', type=int, default=1.5, help='value of lambda')
    parser.add_argument('--hidden_dimension1', type=int, default=4, help='length of environment related feature')
    parser.add_argument('--hidden_dimension2', type=int, default=8, help='length of range error related feature')
    parser.add_argument('--device', type=str, default='cuda' if torch.cuda.is_available() else 'cpu', help='device')

    args = parser.parse_args()
    args.cuda = torch.cuda.is_available()
    #返回True说明GPU可用
    device = torch.device("cuda" if args.cuda else "cpu")

    main(args)

