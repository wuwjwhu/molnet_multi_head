import sys

sys.path.insert(0,"./models")

import pdb
import os
import torch.nn.functional as F
import random
import datetime
from utils.util import *
from utils.dataset_loader import load_data
from models.model_single_cls import GraphMultiHead
import argparse
import torch
from tqdm import tqdm
from random import shuffle
import numpy as np
from copy import deepcopy
from sklearn.metrics import accuracy_score, precision_score, recall_score, f1_score, roc_auc_score
from dgllife.utils import ScaffoldSplitter, RandomSplitter
import pandas as pd

def train(model, device, train_loader, optimizer, epoch, log_interval, args):
    print("Training on {} samples...".format(len(train_loader.dataset)))
    model.train()
    avg_loss = []

    for batch_idx, data in enumerate(train_loader):
        data = data.to(device)
        optimizer.zero_grad()
        pred, _ = model(data)

        # Adjusting the targets and predictions
        targets = data.y.float()  # Assuming data.y is your label matrix

        # Mask for 'void' labels, assuming they are marked as float('inf')
        mask = targets != float('inf')

        # Compute loss only on non-'void' labels
        loss_BCE = F.binary_cross_entropy_with_logits(pred[mask], targets[mask], reduction='mean')

        loss_BCE.backward()
        optimizer.step()

        avg_loss.append(loss_BCE.item())

        if batch_idx % log_interval == 0:
            print("Train Epoch: {} [{}/{} ({:.0f}%)]\tLoss: {:.6f}".format(
                epoch, batch_idx * len(data), len(train_loader.dataset),
                100. * batch_idx / len(train_loader), loss_BCE.item()))

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

    model = GraphMultiHead(config)
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
    work_dir = "./exp/scaffold_seed_"+ str(config['seed']) + "/" + config['marker'] + "/" + work_dir + "_"  +  date_info

    os.makedirs("./exp/scaffold_seed_"+ str(config['seed']) + "/" + config['marker'], exist_ok=True)
    os.makedirs(work_dir, exist_ok=True)


    model_st = config["model_name"]

    dataset = load_data(config['dataset_path'])

    # 数据集划分为训练集、验证集和测试集
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


    best_roc_auc = -1
    best_acc = 0
    best_epoch = -1

    model_file_name = work_dir + "/" + model_st + ".pt"
    result_file_name = work_dir + "/" + model_st + ".csv"
    loss_fig_name = work_dir + "/" + model_st + "_loss"

    train_losses = []
    val_losses = []
    # val_pearsons = []

    rankingLossFunc = torch.nn.MarginRankingLoss(
        margin=0.0, reduction='mean')


    for epoch in tqdm(range(num_epoch+1)):
        
        cls_loss = train(
            model, device,  train_loader, optimizer, epoch + 1, log_interval, config
        )
        # Assuming 'predicting' function returns the raw logits
        G, logits = predicting(model, device, val_loader, "val", config)
        G_test, logits_test = predicting(model, device, test_loader, "test", config)
        
        # Convert logits to probabilities
        P = torch.sigmoid(torch.tensor(logits)).numpy()
        P_test = torch.sigmoid(torch.tensor(logits_test)).numpy()

        # Convert probabilities to binary predictions
        binary_preds = (P > 0.5).astype(int)
        binary_preds_test = (P_test > 0.5).astype(int)

        # Create a mask for valid labels
        valid_label_mask = (G != float('inf'))  # Replace float('inf') with np.nan if NaN is used for 'void' labels
        valid_label_mask_test = (G_test != float('inf'))  # Same for the test set

        # Apply the mask to labels and predictions
        valid_labels = G[valid_label_mask]
        valid_preds = binary_preds[valid_label_mask]
        valid_labels_test = G_test[valid_label_mask_test]
        valid_preds_test = binary_preds_test[valid_label_mask_test]

        # Calculate metrics for valid data only
        accuracy = accuracy_score(valid_labels, valid_preds)
        precision = precision_score(valid_labels, valid_preds, average='macro')
        recall = recall_score(valid_labels, valid_preds, average='macro')
        f1 = f1_score(valid_labels, valid_preds, average='macro')
        roc_auc = roc_auc_score(valid_labels, P[valid_label_mask], average='macro')

        accuracy_test = accuracy_score(valid_labels_test, valid_preds_test)
        precision_test = precision_score(valid_labels_test, valid_preds_test, average='macro')
        recall_test = recall_score(valid_labels_test, valid_preds_test, average='macro')
        f1_test = f1_score(valid_labels_test, valid_preds_test, average='macro')
        roc_auc_test = roc_auc_score(valid_labels_test, P_test[valid_label_mask_test], average='macro')

        train_losses.append(cls_loss)
        val_losses.append(roc_auc)

        # draw_sort_pred_gt(P, G, title=work_dir + "/val_" +str(epoch))

        draw_sort_pred_gt(
            P_test, G_test, title=work_dir + "/test_" + str(epoch))

        if roc_auc > best_roc_auc:
            torch.save(model.state_dict(), model_file_name)

            with open(result_file_name, "a") as f:
                f.write("\n " + str(epoch))
                f.write("\n accuracy:"+str(accuracy_test))
                f.write("\n precision:"+str(precision_test))
                f.write("\n recall:"+str(recall_test))
                f.write("\n f1:"+str(f1_test))
                f.write("\n roc_auc:"+str(roc_auc_test)+"\n")

            best_epoch = epoch + 1
            best_roc_auc = roc_auc
            best_acc = accuracy
            print(
                " roc_auc improved at epoch ",
                best_epoch,
                "; best_roc_auc:",
                best_roc_auc,
                model_st,
            )

        else:
            print(
                " no improvement since epoch ",
                best_epoch,
                "; best_roc_auc, best accuracy:",
                best_roc_auc,
                best_acc,
                model_st,
            )

        draw_loss(train_losses, val_losses, loss_fig_name)
        # draw_pearson(val_pearsons, pearson_fig_name)
        
    # The following is not for transfer task
    
    # load the best model
    best_model = GraphMultiHead(config)
    # best_model.load_state_dict(torch.load('/home/nas/wwj/molnet_multi_head/exp/SINGLE/cls_bace__20240312165917/GraphMultiHead.pt', map_location=torch.device(device)), strict=True)
    best_model.load_state_dict(torch.load(model_file_name, map_location=torch.device(device)), strict=True)
    best_model.to(device)
    
    
    # test the best model on test set
    testset = load_data(config['testset_path'])
    testset_loader = DataLoader(testset, batch_size=config['batch_size']['test'], shuffle=False, drop_last=False)
    G_test, P_test = predicting(best_model, device, testset_loader, "test", config)
    
    valid_label_mask_test = (G_test != float('inf'))  # Same for the test set
    valid_labels_test = G_test[valid_label_mask_test]
    valid_preds_test = P_test[valid_label_mask_test]
    P_test = torch.sigmoid(torch.tensor(valid_preds_test)).numpy()
    binary_preds_test = (P_test > 0.5).astype(int)
    
    accuracy_test = accuracy_score(valid_labels_test, binary_preds_test)
    precision_test = precision_score(valid_labels_test, binary_preds_test, average='macro')
    recall_test = recall_score(valid_labels_test, binary_preds_test, average='macro')
    f1_test = f1_score(valid_labels_test, binary_preds_test, average='macro')
    roc_auc_test = roc_auc_score(valid_labels_test, P_test, average='macro')
    
    draw_sort_pred_gt(P_test, valid_labels_test, title=work_dir + "/testset")
    with open(result_file_name, "a") as f:
        f.write("\n on test set")
        f.write("\n accuracy:"+str(accuracy_test))
        f.write("\n precision:"+str(precision_test))
        f.write("\n recall:"+str(recall_test))
        f.write("\n f1:"+str(f1_test))
        f.write("\n roc_auc:"+str(roc_auc_test)+"\n")

        
    # test the best model on clean test set
    testset = load_data(config['cleanset_path'])
    testset_loader = DataLoader(testset, batch_size=config['batch_size']['test'], shuffle=False, drop_last=False)
    G_test, P_test = predicting(best_model, device, testset_loader, "test", config)
    
    valid_label_mask_test = (G_test != float('inf'))  # Same for the test set
    valid_labels_test = G_test[valid_label_mask_test]
    valid_preds_test = P_test[valid_label_mask_test]
    P_test = torch.sigmoid(torch.tensor(valid_preds_test)).numpy()
    binary_preds_test = (P_test > 0.5).astype(int)
    
    accuracy_test = accuracy_score(valid_labels_test, binary_preds_test)
    precision_test = precision_score(valid_labels_test, binary_preds_test, average='macro')
    recall_test = recall_score(valid_labels_test, binary_preds_test, average='macro')
    f1_test = f1_score(valid_labels_test, binary_preds_test, average='macro')
    roc_auc_test = roc_auc_score(valid_labels_test, P_test, average='macro')
    
    draw_sort_pred_gt(P_test, valid_labels_test, title=work_dir + "/cleanset")
    with open(result_file_name, "a") as f:
        f.write("\n on clean test set")
        f.write("\n accuracy:"+str(accuracy_test))
        f.write("\n precision:"+str(precision_test))
        f.write("\n recall:"+str(recall_test))
        f.write("\n f1:"+str(f1_test))
        f.write("\n roc_auc:"+str(roc_auc_test)+"\n")
        
    # make predictions on merged training set and save to to_fill_training_set
    predictset = load_data(config['predictset_path'])
    predict_loader = DataLoader(predictset, batch_size=config['batch_size']['test'], shuffle=False, drop_last=False)
    dataset_df = pd.read_csv(config['dataset_path'])
    N = len(dataset_df.columns) - 1
    target_df = pd.read_csv(config['predictset_path'])
    _, logits_test = predicting(best_model, device, predict_loader, "test", config)
    logits_test = (logits_test > 0.5).astype(int)
    if len(logits_test) % N == 0:
        reshaped_predictions = logits_test.reshape(-1, N)
    else:
        raise ValueError("The number of predictions does not match")
    predictions_df = pd.DataFrame(reshaped_predictions, columns=dataset_df.columns[1:])
    combined_df = pd.concat([target_df['smiles'], predictions_df], axis=1)
    ensure_dir(config['to_fill_training_set_path'])
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

def ensure_dir(file_path):
    directory = os.path.dirname(file_path)
    if not os.path.exists(directory):
        os.makedirs(directory)


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
