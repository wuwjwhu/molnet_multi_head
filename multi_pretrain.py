import sys

sys.path.insert(0,"./models")



import pdb
import os
import torch.nn.functional as F
import random
import datetime
from utils.util import *
from utils.dataset_with_indicator_loader import load_data
# from models.model_graphdrp import GraphDRP
from models.model_multi import GraphDRP
import argparse
import torch
from tqdm import tqdm
from random import shuffle
import pandas as pd
import numpy as np
from copy import deepcopy
from sklearn.metrics import accuracy_score, precision_score, recall_score, f1_score, roc_auc_score
from dgllife.utils import ScaffoldSplitter, RandomSplitter
from sklearn.metrics import mean_squared_error, r2_score


def classification_loss(output, target, indicator, weight):
    # Ensure that target, output, and indicator have the same dimensions
    target = target.view(1, -1)
    output = output.view(1, -1)
    indicator = indicator.view(1, -1)
    
    # Create the weights tensor based on the values in indicator
    weights = torch.where(indicator == 1, torch.ones_like(indicator, dtype=torch.float), torch.full_like(indicator, weight, dtype=torch.float))
    
    # Calculate the weighted binary cross entropy loss
    return F.binary_cross_entropy_with_logits(output, target, weight=weights, reduction='mean')

def regression_loss(output, target, indicator, weight): 
    reg_loss = 0.0
    for i, (output_, target_, indicator_) in enumerate(zip(output, target, indicator)):
        weight_ = (1 if indicator_ == 1 else weight)
        reg_loss += F.mse_loss(output_, target_, reduction='mean') * weight_
    return reg_loss

def loss_weight_calculating(config):
    dataset_performance = config.get('dataset_performance', [])  # Use ROC AUC for classification, RMSE for regression
    predicted_label_weight = []
    for i, performance in enumerate(dataset_performance):
        if i < 6:  # Assuming first 6 tasks are classification
            # Convert ROC AUC to a form where lower scores increase the weight
            # One minus ROC AUC could serve to invert its effect, with some scaling factor if needed
            weight = performance  
        else:  # Assuming the rest are regression tasks
            # Invert RMSE, ensuring higher errors increase the task weight
            weight = 1 / (performance + 1e-5)
        predicted_label_weight.append(weight)
    
    # Optional: Normalize weights to ensure they sum to a specific value (e.g., number of tasks)
    total_weight = sum(predicted_label_weight)
    predicted_label_weight = [w / total_weight for w in predicted_label_weight]

    return predicted_label_weight

def train(model, device, train_loader, optimizer, epoch, log_interval, config):
    print("Training on {} samples...".format(len(train_loader.dataset)))
    model.train()
    avg_loss = []

    for batch_idx, data in enumerate(train_loader):
        data = data.to(device)
        optimizer.zero_grad()
        outputs = model(data)
        # pdb.set_trace()
        # Assuming data.y is a list of label tensors, one for each task
        labels = data.y  
        indicators = data.indicator
        predicted_label_weight = loss_weight_calculating(config)

        total_loss = 0
        for i, (output, label, indicator) in enumerate(zip(outputs, labels, indicators)):
            if i < 6: # if task_type == 'classification':
                # if i < 2:
                #     continue
                # if i == 2:
                #     pdb.set_trace()
                weight =  predicted_label_weight[i]
                total_loss += classification_loss(output, label, indicator, weight)
            else: # elif task_type == 'regression':
                # pdb.set_trace()
                total_loss += regression_loss(output, label, indicator, weight)

        total_loss.backward()
        optimizer.step()
        avg_loss.append(total_loss.item())

        if batch_idx % log_interval == 0:
            print("Train Epoch: {} [{}/{} ({:.0f}%)]\tLoss: {:.6f}".format(
                epoch, batch_idx * len(data), len(train_loader.dataset),
                100. * batch_idx / len(train_loader), total_loss.item() ))

    # Avoid division by zero if avg_loss is empty
    return sum(avg_loss) / len(avg_loss) if avg_loss else float('nan')



def predict(model, device, loader, config):
    model.eval()  # Set the model to evaluation mode
    print("Making predictions for {} samples...".format(len(loader.dataset)))

    # Initialize containers for predictions and labels for each task
    total_preds = [[] for _ in range(len(config['task_output_dims']))]
    total_labels = [[] for _ in range(len(config['task_output_dims']))]

    with torch.no_grad():  # Inference mode, no gradients needed
        for data in loader:
            data = data.to(device)

            preds = model(data)  # Get model predictions

            # Loop over each task
            for i, (pred, label) in enumerate(zip(preds, data.y)):
                # Collect predictions and labels for each task
                # Ensure tensor is moved to CPU and detached
                total_preds[i].append(pred.cpu().detach())
                total_labels[i].append(label.cpu().detach())

    # Post-process to concatenate all batch results for each task and convert them to numpy arrays
    for i in range(len(total_preds)):
        total_preds[i] = torch.cat(total_preds[i], dim=0).numpy()
        total_labels[i] = torch.cat(total_labels[i], dim=0).numpy()

    return total_labels, total_preds

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
    work_dir = "./exp/scaffold_seed_"+ str(config['seed']) + "/" + config['marker'] + "/" + work_dir + "_"  +  date_info

    os.makedirs("./exp/scaffold_seed_"+ str(config['seed']) + "/" + config['marker'], exist_ok=True)
    os.makedirs(work_dir, exist_ok=True)

    model_st = config["model_name"]

    dataset = load_data(config)

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


    model_file_prefix = work_dir + "/" + model_st
    result_file_prefix = work_dir + "/" + model_st
    loss_fig_name = work_dir + "/" + model_st + "_loss"

    train_losses = []
    val_losses = []
    # val_pearsons = []

    rankingLossFunc = torch.nn.MarginRankingLoss(
        margin=0.0, reduction='mean')
    
    task_name = config['task_name']
    num_classification_tasks = config['num_classification_tasks']
    
    best_performance = {
    'classification': [0] * num_classification_tasks,  # Initialize with zeros or appropriate starting values for ROC AUC
    'regression': [float('inf')] * (len(config['task_output_dims']) - num_classification_tasks)  # Initialize with infinity for RMSE
    }
    best_model_states = {
        'classification': [None] * num_classification_tasks,
        'regression': [None] * (len(config['task_output_dims']) - num_classification_tasks)
    }

    for epoch in tqdm(range(num_epoch + 1)):
        loss = train(model, device, train_loader, optimizer, epoch + 1, log_interval, config)

        # Lists of predictions and labels for each task
        G_list, logits_list = predict(model, device, val_loader, config)
        G_test_list, logits_test_list = predict(model, device, test_loader, config)

        # Iterate over each task to calculate metrics separately, handling classification and regression differently
        for i, (logits, labels, logits_test, labels_test) in enumerate(zip(logits_list, G_list, logits_test_list, G_test_list)):
            # Classification metrics
            # pdb.set_trace()
            if i < num_classification_tasks:
                # if i == 5:
                #     pdb.set_trace()
                #     labels = np.array(labels).flatten()  # Assuming 'labels' is your array of labels
                #     # Check the unique classes and their counts
                #     unique_classes, counts = np.unique(labels, return_counts=True)
                #     print("Classes:", unique_classes)
                #     print("Counts:", counts)
                #     # continue
                probabilities = torch.sigmoid(torch.tensor(logits)).numpy().flatten()
                roc_auc = roc_auc_score(np.array(labels).flatten(), probabilities, average='macro')

                if roc_auc > best_performance['classification'][i]:
                    best_performance['classification'][i] = roc_auc
                    best_model_states['classification'][i] = model.state_dict()  # Save model state
                    torch.save(model.state_dict(), f"{model_file_prefix}_epoch_{epoch}_best_model_classification_task_{task_name[i]}.pt")  # Save to file
                    test_probabilities = torch.sigmoid(torch.tensor(logits_test)).numpy()
                    test_binary_preds = (test_probabilities > 0.5).astype(int)
                    test_labels_np = np.array(labels_test)
                    test_accuracy = accuracy_score(test_labels_np, test_binary_preds)
                    test_precision = precision_score(test_labels_np, test_binary_preds, average='macro', zero_division=0)
                    test_recall = recall_score(test_labels_np, test_binary_preds, average='macro', zero_division=0)
                    test_f1 = f1_score(test_labels_np, test_binary_preds, average='macro', zero_division=0)
                    test_roc_auc = roc_auc_score(test_labels_np.flatten(), test_probabilities.flatten(), average='macro')
                    with open(result_file_prefix + "_dataset_"+ task_name[i] + ".csv", "a") as f:
                        f.write("\n " + str(epoch))
                        f.write("\n accuracy:"+str(test_accuracy))
                        f.write("\n precision:"+str(test_precision))
                        f.write("\n recall:"+str(test_recall))
                        f.write("\n f1:"+str(test_f1))
                        f.write("\n roc_auc:"+str(test_roc_auc)+"\n")
                print(f"Validation - Classification Task {task_name[i]}: ROC AUC: {roc_auc:.4f}")

            # Regression metrics
            else:
                predictions = torch.tensor(logits).numpy()
                rmse = sqrt(mean_squared_error(np.array(labels), predictions))

                if rmse < best_performance['regression'][i - num_classification_tasks]:
                    best_performance['regression'][i - num_classification_tasks] = rmse
                    best_model_states['regression'][i - num_classification_tasks] = model.state_dict()  # Save model state
                    torch.save(model.state_dict(), f"{model_file_prefix}_epoch_{epoch}_best_model_regression_task_{task_name[i]}.pt")  # Save to file
                    test_predictions = torch.tensor(logits_test).numpy()
                    test_labels_np = np.array(labels_test)
                    test_mse = mean_squared_error(test_labels_np, test_predictions)
                    test_rmse = sqrt(test_mse)
                    test_r2 = r2_score(test_labels_np, test_predictions)
                    with open(result_file_prefix + "_dataset_"+ task_name[i] + ".csv", "a") as f:
                        f.write("\n " + str(epoch))
                        f.write("\n mse:"+str(test_mse))
                        f.write("\n rmse:"+str(test_rmse))
                        f.write("\n r2:"+str(test_r2)+"\n")
                print(f"Validation - Regression Task {task_name[i]}: RMSE: {rmse:.4f}")


        # Log or print the metrics
        # ...

        train_losses.append(loss)

        draw_loss(train_losses, val_losses, loss_fig_name)
        # draw_pearson(val_pearsons, pearson_fig_name)
        
        # save drug_module only for transfer tasks
        if epoch % 50 == 0:
            torch.save(model.drug_module.state_dict(),f'{model_file_prefix}_epoch_{epoch}_drug_module.pt')

    # test the best model on test set
    # for i in range(config['num_tasks']):
    #     best_model = GraphDRP(config)
    #     if i < num_classification_tasks:
    #         best_model.load_state_dict(torch.load(""))
    #         model.load(best_model_states['classification'][i])

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
