import torch
import torch.nn as nn
import torch.optim as optim
import numpy as np
import json
import sys
from sklearn.metrics import roc_auc_score
from data_loader import FineTuneDataLoader
from model import Net

# can be changed according to config.txt
with open('config.txt') as i_f:
    i_f.readline()
    student_n, exer_n, knowledge_n = list(map(eval, i_f.readline().split(',')))

domain = ['geometry', 'function', 'probability', 'physics', 'arithmetic', 'english']
domain_stu = [15283, 15404, 4076, 13369, 14073, 5906]

domain_id = 0
select_epoch = 3

domain_id = 1
select_epoch = 1

domain_id = 2
select_epoch = 1

domain_id = 3
select_epoch = 1

domain_id = 4
select_epoch = 1

domain_id = 5
select_epoch = 1

peer_num = 500

traget_domain = domain[domain_id]
sample_num = int(domain_stu[domain_id] * 0.01)

epoch_n = 30
device = torch.device(('cuda:2') if torch.cuda.is_available() else 'cpu')
print(domain_id)

def fine_tune(epoch):
    print('Fine-tuning model...')
    data_loader = FineTuneDataLoader('fine-tune_step_2', traget_domain, sample_num, peer_num)
    net = Net(student_n, exer_n, knowledge_n, domain, device)
    load_snapshot(net, 'model/' + traget_domain + '/model_epoch' + str(epoch))
    net = net.to(device)
    optimizer = optim.Adam(net.parameters(), lr=0.002)

    loss_function = nn.MSELoss()
    for epoch in range(epoch_n):
        data_loader.reset()
        running_loss = 0.0
        batch_count = 0
        while not data_loader.is_end():
            batch_count += 1
            input_stu_ids, input_knowledge_embs, labels, exer_emb, know_emb, domain_ids = data_loader.next_batch()
            input_stu_ids, input_knowledge_embs, labels, exer_emb, know_emb, domain_ids = (
                input_stu_ids.to(device), input_knowledge_embs.to(device), labels.to(device),
                exer_emb.to(device), know_emb.to(device), domain_ids.to(device)
            )
            optimizer.zero_grad()
            output = net.fine_tune(input_stu_ids, input_knowledge_embs, exer_emb, know_emb, domain_id)

            loss = loss_function(output.float(), labels.float())
            loss.backward()
            optimizer.step()
            net.apply_clipper()

            running_loss += loss.item()
            if batch_count % 200 == 199:
                print('[%d, %5d] loss: %.10f' % (epoch + 1, batch_count + 1, running_loss / 200))
                running_loss = 0.0

        # Validate and save current model every epoch
        validate(net, epoch, traget_domain)
        save_snapshot(net, 'model/' + traget_domain + '/fine_tune_step_2_' + str(sample_num) + '_peer_' + str(peer_num) + '_model_epoch' + str(epoch + 1))

def validate(net, epoch, traget_domain):
    data_loader = FineTuneDataLoader('fine-tune-val_step_2', traget_domain, sample_num, peer_num)
    print('Validating model...')
    data_loader.reset()

    net = net.to(device)
    net.eval()

    correct_count, exer_count = 0, 0
    batch_count, batch_avg_loss = 0, 0.0
    pred_all, label_all = [], []
    while not data_loader.is_end():
        batch_count += 1
        input_stu_ids, input_knowledge_embs, labels, exer_emb, know_emb, domain_ids = data_loader.next_batch()
        input_stu_ids, input_knowledge_embs, labels, exer_emb, know_emb, domain_ids = (
            input_stu_ids.to(device), input_knowledge_embs.to(device), labels.to(device),
            exer_emb.to(device), know_emb.to(device), domain_ids.to(device)
        )

        output = net.fine_tune(input_stu_ids, input_knowledge_embs, exer_emb, know_emb, domain_id)
        output = output.view(-1)

        # Compute accuracy
        for i in range(len(labels)):
            if (labels[i] == 1 and output[i] > 0.5) or (labels[i] == 0 and output[i] < 0.5):
                correct_count += 1
        exer_count += len(labels)
        pred_all += output.to(torch.device('cpu')).tolist()
        label_all += labels.to(torch.device('cpu')).tolist()

    pred_all = np.array(pred_all)
    label_all = np.array(label_all)

    # Compute accuracy
    accuracy = correct_count / exer_count
    # Compute RMSE
    rmse = np.sqrt(np.mean((label_all - pred_all) ** 2))
    # Compute AUC
    auc = roc_auc_score(label_all, pred_all)

    print('epoch= %d, accuracy= %f, rmse= %f, auc= %f' % (epoch + 1, accuracy, rmse, auc))
    with open('result/' + traget_domain + '/model_val_step_2_' + str(sample_num) + '_peer_' + str(peer_num) + '.txt', 'a', encoding='utf8') as f:
        f.write('epoch= %d, accuracy= %f, rmse= %f, auc= %f\n' % (epoch + 1, accuracy, rmse, auc))

    return rmse, auc

def load_snapshot(model, filename):
    with open(filename, 'rb') as f:
        model.load_state_dict(torch.load(f, map_location=lambda s, loc: s))

def save_snapshot(model, filename):
    with open(filename, 'wb') as f:
        torch.save(model.state_dict(), f)

if __name__ == '__main__':
    fine_tune(select_epoch)
