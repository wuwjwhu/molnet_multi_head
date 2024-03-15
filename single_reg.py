import sys

sys.path.insert(0,"./models")



import pdb
import os
import torch.nn.functional as F
import random
import datetime
from utils.util import *
from utils.dataset_loader import load_data
from models.model_single_reg import GraphDRP
import argparse
import torch
from tqdm import tqdm
from random import shuffle
import numpy as np
from copy import deepcopy
from dgllife.utils import ScaffoldSplitter, RandomSplitter
import pandas as pd


# training function at each epoch
def train(model, device, train_loader, optimizer, epoch, log_interval, args):
    print("Training on {} samples...".format(len(train_loader.dataset)))
    model.train()
    # loss_fn = nn.MSELoss()
    # DC_cross_entropy_loss = torch.nn.CrossEntropyLoss()
    # T_cross_entropy_loss = torch.nn.CrossEntropyLoss()
    avg_loss = []
    # pdb.set_trace()
    for batch_idx, data in enumerate(train_loader):
        data = data.to(device)

        optimizer.zero_grad()
        # pdb.set_trace()
        pred, _ = model(data)
        


        loss_MSE = F.mse_loss(pred.view(-1), data.y.view(-1))

        loss_MSE.backward()

        optimizer.step()
        avg_loss.append(loss_MSE.item())
        if batch_idx % log_interval == 0:
            print(
                "Train epoch: {} ({:.0f}%)\tLoss1: {:.6f}\tLoss2: {:.6f}".format(
                    epoch, 100.0 * batch_idx / len(train_loader), loss_MSE.item(), loss_MSE.item()
                )
            )
    return sum(avg_loss) / len(avg_loss)



def predicting(model, device, loader, loader_type, args):
    model.eval()
    total_preds = torch.Tensor()
    total_labels = torch.Tensor()
    print("Make prediction for {} samples...".format(len(loader.dataset)))
    with torch.no_grad():
        for batch_idx, data in tqdm(enumerate(loader)):
            data = data.to(device)
            # data_ = deepcopy(data)

            pred, _ = model(data)

            total_preds = torch.cat((total_preds, pred.cpu().view(-1, 1)), 0)
            total_labels = torch.cat((total_labels, data.y.view(-1, 1).cpu()), 0)

    return total_labels.numpy().flatten(), total_preds.numpy().flatten()


'''freeze'''


def dateStr():
    return (
        str(datetime.datetime.now())
        .replace(" ", "_")
        .replace("-", "_")
        .replace(":", "_")
        .split(".")[0]
        .replace("_", "")
    )


def main(config, yaml_path):



    model = GraphDRP(config)

    model.to(device)


    train_batch = config["batch_size"]["train"]
    val_batch = config["batch_size"]["val"]
    test_batch = config["batch_size"]["test"]
    lr = config["lr"]
    num_epoch = config["num_epoch"]
    log_interval = config["log_interval"]

    work_dir = config["work_dir"]

    date_info = ("_" + dateStr()) if config["work_dir"] != "test" else ""

    date_info = ("_" + dateStr()) if config["work_dir"] != "test" else ""
    work_dir = "./exp/seed_"+ str(config['seed']) + "/" + config['marker'] + "/" + work_dir + "_"  +  date_info

    os.makedirs("./exp/seed_"+ str(config['seed']) + "/" + config['marker'], exist_ok=True)
    os.makedirs(work_dir, exist_ok=True)

    # copyfile(yaml_path, work_dir + "/")
    model_st = config["model_name"]


    dataset = load_data(config['dataset_path'])

    # 数据集划分为训练集、验证集和测试集 # todo 改scaffold
    if config['split_type'] == 'random':
        train_dataset, val_dataset, test_dataset = RandomSplitter.train_val_test_split(
                dataset, frac_train=0.8, frac_val=0.1, frac_test=0.1, random_state=config["seed"])
    elif config['split_type'] == 'scaffold':
        train_dataset, val_dataset, test_dataset = ScaffoldSplitter.train_val_test_split(
            dataset, frac_train=0.8, frac_val=0.1, frac_test=0.1,
                scaffold_func='smiles')
    else:
        return ValueError("Expect the splitting method to be '', got {}".format(config['split_type']))
 

    # 装载dataloader
    train_loader = DataLoader(train_dataset, batch_size=config['batch_size']['train'], shuffle=True, drop_last=True)
    val_loader = DataLoader(val_dataset, batch_size=config['batch_size']['val'], shuffle=False, drop_last=False)
    test_loader = DataLoader(test_dataset, batch_size=config['batch_size']['test'], shuffle=False, drop_last=False)


    optimizer = torch.optim.Adam(
        filter(lambda p: p.requires_grad, model.parameters()), lr=lr)

    scheduler = None


    best_rmse = 9999
    best_pearson = 1
    best_epoch = -1

    model_file_name = work_dir + "/" + model_st + ".pt"
    result_file_name = work_dir + "/" + model_st + ".csv"
    loss_fig_name = work_dir + "/" + model_st + "_loss"

    train_losses = []
    val_losses = []

    rankingLossFunc = torch.nn.MarginRankingLoss(
        margin=0.0, reduction='mean')


    for epoch in tqdm(range(num_epoch+1)):

        reg_loss = train(model, device,  train_loader, optimizer, epoch + 1, log_interval, config)

        G, P = predicting(model, device, val_loader, "val", config)

        G_test, P_test = predicting(model, device, test_loader, "test", config)
        # pdb.set_trace()
        # Create a mask for valid labels
        valid_label_mask = (G != float('inf'))  # Replace -1 with np.nan if NaN is used for 'void' labels
        valid_label_mask_test = (G_test != float('inf'))  # Same for the test set
        
        # Apply the mask to labels and predictions
        G = G[valid_label_mask]
        P = P[valid_label_mask]
        G_test = G_test[valid_label_mask_test]
        P_test = P_test[valid_label_mask_test]
        
        ret = [rmse(G, P), mse(G, P), pearson(G, P), spearman(G, P)]

        ret_test = [
            rmse(G_test, P_test),
            mse(G_test, P_test),
            pearson(G_test, P_test),
            spearman(G_test, P_test),
            rankingLossFunc(torch.tensor(G_test), torch.tensor(
                P_test), torch.ones_like(torch.tensor(P_test))).item()
        ]
        # print(ret_test)

        train_losses.append(reg_loss)
        val_losses.append(ret[0])

        # draw_sort_pred_gt(P, G, title=work_dir + "/val_" +str(epoch))

        draw_sort_pred_gt(
            P_test, G_test, title=work_dir + "/test_" + str(epoch))

        if ret[0] < best_rmse:
            torch.save(model.state_dict(), model_file_name)

            with open(result_file_name, "a") as f:
                f.write("\n " + str(epoch))
                f.write("\n rmse:"+str(ret_test[0]))
                f.write("\n mse:"+str(ret_test[1]))
                f.write("\n pearson:"+str(ret_test[2]))
                f.write("\n spearman:"+str(ret_test[3]))
                f.write("\n rankingloss:"+str(ret_test[4])+"\n")

            best_epoch = epoch + 1
            best_rmse = ret[0]
            best_pearson = ret[2]
            print(
                " rmse improved at epoch ",
                best_epoch,
                "; best_rmse:",
                best_rmse,
                model_st,
            )

        else:
            print(
                " no improvement since epoch ",
                best_epoch,
                "; best_rmse, best pearson:",
                best_rmse,
                best_pearson,
                model_st,
            )

        draw_loss(train_losses, val_losses, loss_fig_name)

    # test the best model on test set
    best_model = GraphDRP(config)
    # best_model.load_state_dict(torch.load('/home/nas/wwj/molnet_multi_head/exp/SINGLE/cls_bace__20240312165917/GraphDRP.pt', map_location=torch.device(device)), strict=True)
    best_model.load_state_dict(torch.load(model_file_name, map_location=torch.device(device)), strict=True)
    best_model.to(device)
    testset = load_data(config['testset_path'])
    testset_loader = DataLoader(testset, batch_size=config['batch_size']['test'], shuffle=False, drop_last=False)
    G_test, P_test = predicting(best_model, device, testset_loader, "test", config)
    
    valid_label_mask_test = (G_test != float('inf'))  # Same for the test set
    G_test = G_test[valid_label_mask_test]
    P_test = P_test[valid_label_mask_test]
    
    ret_test = [
        rmse(G_test, P_test),
        mse(G_test, P_test),
        pearson(G_test, P_test),
        spearman(G_test, P_test),
        rankingLossFunc(torch.tensor(G_test), torch.tensor(
            P_test), torch.ones_like(torch.tensor(P_test))).item()
    ]
    draw_sort_pred_gt(
        P_test, G_test, title=work_dir + "/testset")
    with open(result_file_name, "a") as f:
        f.write("\n on test set")
        f.write("\n rmse:"+str(ret_test[0]))
        f.write("\n mse:"+str(ret_test[1]))
        f.write("\n pearson:"+str(ret_test[2]))
        f.write("\n spearman:"+str(ret_test[3]))
        f.write("\n rankingloss:"+str(ret_test[4])+"\n")
        
        
        
    # make predictions on merged training set and save to to_fill_training_set
    # pdb.set_trace()
    predictset = load_data(config['predictset_path'])
    predict_loader = DataLoader(predictset, batch_size=config['batch_size']['test'], shuffle=False, drop_last=False)
    
    dataset_df = pd.read_csv(config['dataset_path'])
    N = len(dataset_df.columns) - 1
    target_df = pd.read_csv(config['predictset_path'])
    _, logits_test = predicting(best_model, device, predict_loader, "test", config)
    if len(logits_test) % N == 0:
        reshaped_predictions = logits_test.reshape(-1, N)
    else:
        raise ValueError("The number of predictions does not match")
    predictions_df = pd.DataFrame(reshaped_predictions, columns=dataset_df.columns[1:])
    combined_df = pd.concat([target_df['smiles'], predictions_df], axis=1)
    combined_df.to_csv(config['to_fill_training_set_path'], index=False)    


def seed_torch(seed=42):
    seed = int(seed)
    random.seed(seed)
    os.environ["CUBLAS_WORKSPACE_CONFIG"] = ":4096:8"
    os.environ["PYTHONHASHSEED"] = str(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)
    torch.backends.cudnn.deterministic = True
    torch.backends.cudnn.benchmark = False
    torch.backends.cudnn.enabled = False


def getConfig():
    parser = argparse.ArgumentParser(description="")
    parser.add_argument(
        "--config",
        type=str,
        required=False,
        default="./config/Transformer_edge_concat_GDSCv2.yaml",
        help="",
    )
    args = parser.parse_args()
    import yaml

    with open(args.config, "r") as stream:
        config = yaml.safe_load(stream)
    return config, args.config


def load_weight(pretrain_model, model, remainRaw=False):
    pre_dict = {}

    pretrain_weight = torch.load(
        pretrain_model, map_location=torch.device('cpu'))

    for key, value in pretrain_weight.items():

        if "drug_module" in key or "cell_module" in key:
            pre_dict[key] = value
        else:
            if remainRaw:
                key_names = [key,
                             key.replace("fusion_module", "fusion_module1"),
                             key.replace("fusion_module", "fusion_module2")
                             ]
                pre_dict[key_names[2]] = value
            else:
                key_names = [key.replace("fusion_module", "fusion_module1"),
                             key.replace("fusion_module", "fusion_module2")
                             ]
            pre_dict[key_names[0]] = value
            pre_dict[key_names[1]] = value

    model.load_state_dict(pre_dict, strict=True)

    return model



if __name__ == "__main__":
    config, yaml_path = getConfig()
    seed_torch(config["seed"])

    cuda_name = config["cuda_name"]

    print("CPU/GPU: ", torch.cuda.is_available())

    device = torch.device(cuda_name if torch.cuda.is_available() else "cpu")

    main(config, yaml_path)
