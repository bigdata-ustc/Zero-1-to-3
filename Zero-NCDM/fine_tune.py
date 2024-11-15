import torch
import torch.nn as nn
import torch.optim as optim
import numpy as np
import json
import sys
from sklearn.metrics import roc_auc_score
from data_loader import FineTuneDataLoader
from model import Net

# Read configurations from config.txt
with open('config.txt') as i_f:
    i_f.readline()
    student_n, exer_n, knowledge_n = list(map(eval, i_f.readline().split(',')))

# Domain configurations
domain = ['geometry', 'function', 'probability', 'physics', 'arithmetic', 'english']
domain_stu = [15283, 15404, 4076, 13369, 14073, 5906] # Student number in each domain

# Select domain and epoch for fine-tuning
domain_id = 0  # Change this based on your desired domain
select_epoch = 3  # Select which epoch to fine-tune

# Set target domain and sample size
traget_domain = domain[domain_id]
sample_num = int(domain_stu[domain_id] * 0.01)
epoch_n = 30 # Epoch number for fine-tuning
device = torch.device('cpu')

def fine_tune(epoch):
    print('Fine-tuning model...')
    data_loader = FineTuneDataLoader('fine-tune', traget_domain, sample_num, 0)
    net = Net(student_n, exer_n, knowledge_n, domain, device)
    
    # Load pretrained model
    load_snapshot(net, f'model/{traget_domain}/model_epoch{epoch}')
    net = net.to(device)
    
    optimizer = optim.Adam(net.parameters(), lr=0.002)
    loss_function = nn.MSELoss()

    for en in range(epoch_n):
        data_loader.reset()
        running_loss = 0.0
        batch_count = 0
        while not data_loader.is_end():
            batch_count += 1
            input_stu_ids, input_knowledge_embs, labels, exer_emb, know_emb, domain_ids = data_loader.next_batch()
            input_stu_ids, input_knowledge_embs, labels, exer_emb, know_emb, domain_ids = (
                input_stu_ids.to(device), input_knowledge_embs.to(device),
                labels.to(device), exer_emb.to(device), know_emb.to(device),
                domain_ids.to(device)
            )
            
            optimizer.zero_grad()
            output = net.fine_tune(input_stu_ids, input_knowledge_embs, exer_emb, know_emb, domain_id)
            loss = loss_function(output.float(), labels.float())
            loss.backward()
            optimizer.step()
            net.apply_clipper()

            running_loss += loss.item()
            if batch_count % 200 == 199:
                print(f'[{en+1}, {batch_count+1}] loss: {running_loss / 200:.10f}')
                running_loss = 0.0

        # Validate and save model every epoch
        validate(net, en, traget_domain)
        save_snapshot(net, f'model/{traget_domain}/fine_tune_{sample_num}_model_F_epoch{en+1}')

def validate(net, epoch, traget_domain):
    data_loader = FineTuneDataLoader('fine-tune-val', traget_domain, sample_num, 0)
    print('Validating model...')
    data_loader.reset()

    net = net.to(device)
    net.eval()

    correct_count, exer_count = 0, 0
    pred_all, label_all = [], []

    while not data_loader.is_end():
        input_stu_ids, input_knowledge_embs, labels, exer_emb, know_emb, domain_ids = data_loader.next_batch()
        input_stu_ids, input_knowledge_embs, labels, exer_emb, know_emb, domain_ids = (
            input_stu_ids.to(device), input_knowledge_embs.to(device),
            labels.to(device), exer_emb.to(device), know_emb.to(device),
            domain_ids.to(device)
        )

        output = net.fine_tune(input_stu_ids, input_knowledge_embs, exer_emb, know_emb, domain_id)
        output = output.view(-1)

        # Compute accuracy
        correct_count += sum((labels[i] == 1 and output[i] > 0.5) or (labels[i] == 0 and output[i] < 0.5) for i in range(len(labels)))
        exer_count += len(labels)
        pred_all.extend(output.cpu().tolist())
        label_all.extend(labels.cpu().tolist())

    # Compute accuracy, RMSE, and AUC
    pred_all = np.array(pred_all)
    label_all = np.array(label_all)
    accuracy = correct_count / exer_count
    rmse = np.sqrt(np.mean((label_all - pred_all) ** 2))
    auc = roc_auc_score(label_all, pred_all)

    print(f'epoch={epoch+1}, accuracy={accuracy:.6f}, rmse={rmse:.6f}, auc={auc:.6f}')
    
    # Write results to file
    with open(f'result/{traget_domain}/model_val_{sample_num}.txt', 'a', encoding='utf8') as f:
        f.write(f'epoch={epoch+1}, accuracy={accuracy:.6f}, rmse={rmse:.6f}, auc={auc:.6f}\n')

    return rmse, auc

def load_snapshot(model, filename):
    with open(filename, 'rb') as f:
        model.load_state_dict(torch.load(f, map_location=lambda s, loc: s))

def save_snapshot(model, filename):
    with open(filename, 'wb') as f:
        torch.save(model.state_dict(), f)

if __name__ == '__main__':
    fine_tune(select_epoch)
