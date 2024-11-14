import torch
import torch.nn as nn
import torch.optim as optim
import numpy as np
import json
import sys
from sklearn.metrics import roc_auc_score
from data_loader import TrainDataLoader, ValTestDataLoader
from model import Net


# can be changed according to config.txt
with open('config.txt') as i_f:
    i_f.readline()
    student_n, exer_n, knowledge_n = list(map(eval, i_f.readline().split(',')))
device = torch.device(('cuda:2') if torch.cuda.is_available() else 'cpu')
domain = ['geometry', 'function', 'probability', 'physics', 'arithmetic', 'english']
domain_id = 5
traget_domain = domain[domain_id]
epoch_n = 20

print (domain_id)
def train():
    data_loader = TrainDataLoader(traget_domain, domain)
    net = Net(student_n, exer_n, knowledge_n, domain, device)

    net = net.to(device)
    optimizer = optim.Adam(net.parameters(), lr=0.001)
    print('training model...')

    loss_function = nn.MSELoss()
    for epoch in range(epoch_n):
        data_loader.reset()
        running_loss = 0.0
        batch_count = 0
        while not data_loader.is_end():
            batch_count += 1
            input_stu_ids, input_stu_ids_other, input_stu_ids_sha, input_exer_ids, input_exer_ids_other, Q_embs, Q_embs_other, Q_embs_sha, ys_spe, ys_spe_other, ys_sha, exer_emb, exer_emb_other, exer_emb_sha, know_emb, know_emb_other, know_emb_sha, input_domain = data_loader.next_batch()
            input_stu_ids, input_stu_ids_other, input_stu_ids_sha, input_exer_ids, input_exer_ids_other, Q_embs, Q_embs_other, Q_embs_sha, ys_spe, ys_spe_other, ys_sha, exer_emb, exer_emb_other, exer_emb_sha, know_emb, know_emb_other, know_emb_sha, input_domain = input_stu_ids.to(device), input_stu_ids_other.to(device), input_stu_ids_sha.to(device), input_exer_ids.to(device), input_exer_ids_other.to(device), Q_embs.to(device), Q_embs_other.to(device), Q_embs_sha.to(device), ys_spe.to(device), ys_spe_other.to(device), ys_sha.to(device), exer_emb.to(device), exer_emb_other.to(device), exer_emb_sha.to(device), know_emb.to(device), know_emb_other.to(device), know_emb_sha.to(device), input_domain.to(device)
            optimizer.zero_grad()
            output_spe, output_spe_other, output_glo_sha, output_sha = net.forward(input_stu_ids, input_stu_ids_other, input_stu_ids_sha, Q_embs, Q_embs_other, Q_embs_sha, exer_emb, exer_emb_other, exer_emb_sha, know_emb, know_emb_other, know_emb_sha, input_domain)
            # output_0 = torch.ones(output_1.size()).to(device) - output_1
            # output = torch.cat((output_0, output_1), 1)
            # output_spe_0 = torch.ones(output_spe.size()).to(device) - output_spe
            # output_spe_1 = torch.cat((output_spe_0, output_spe), 1)

            # grad_penalty = 0
            # loss = loss_function(torch.log(output_spe_1 + 0.0001), ys_spe)

            # grad_penalty = 0
            loss = 1 + loss_function(output_spe.float(), ys_spe.float()) - loss_function(output_spe_other.float(), ys_spe_other.float()) + loss_function(output_glo_sha.float(), ys_sha.float()) - loss_function(output_sha.float(), ys_spe.float())
            # loss2 = loss_function(output_glo_sha.float(), ys_sha.float())
            # loss2 -= loss_function(output_sha.float(), ys_spe.float())
            loss.backward()
            optimizer.step()
            net.apply_clipper()

            running_loss += loss.item()
            if batch_count % 200 == 199:
                print('[%d, %5d] loss: %.10f' % (epoch + 1, batch_count + 1, running_loss / 200))
                running_loss = 0.0

        # validate and save current model every epoch
        # rmse, auc = validate(net, epoch)
        save_snapshot(net, 'model/' + traget_domain + '/model_epoch' + str(epoch + 1))


def validate(model, epoch):
    data_loader = ValTestDataLoader('validation')
    net = Net(student_n, exer_n, knowledge_n)
    print('validating model...')
    data_loader.reset()
    # load model parameters
    net.load_state_dict(model.state_dict())
    net = net.to(device)
    net.eval()

    correct_count, exer_count = 0, 0
    batch_count, batch_avg_loss = 0, 0.0
    pred_all, label_all = [], []
    while not data_loader.is_end():
        batch_count += 1
        input_stu_ids, input_exer_ids, input_knowledge_embs, labels = data_loader.next_batch()
        input_stu_ids, input_exer_ids, input_knowledge_embs, labels = input_stu_ids.to(device), input_exer_ids.to(
            device), input_knowledge_embs.to(device), labels.to(device)
        output = net.forward(input_stu_ids, input_exer_ids, input_knowledge_embs)
        output = output.view(-1)
        # compute accuracy
        for i in range(len(labels)):
            if (labels[i] == 1 and output[i] > 0.5) or (labels[i] == 0 and output[i] < 0.5):
                correct_count += 1
        exer_count += len(labels)
        pred_all += output.to(torch.device('cpu')).tolist()
        label_all += labels.to(torch.device('cpu')).tolist()

    pred_all = np.array(pred_all)
    label_all = np.array(label_all)
    # compute accuracy
    accuracy = correct_count / exer_count
    # compute RMSE
    rmse = np.sqrt(np.mean((label_all - pred_all) ** 2))
    # compute AUC
    auc = roc_auc_score(label_all, pred_all)
    print('epoch= %d, accuracy= %f, rmse= %f, auc= %f' % (epoch+1, accuracy, rmse, auc))
    with open('result/model_val.txt', 'a', encoding='utf8') as f:
        f.write('epoch= %d, accuracy= %f, rmse= %f, auc= %f\n' % (epoch+1, accuracy, rmse, auc))

    return rmse, auc

def save_snapshot(model, filename):
    f = open(filename, 'wb')
    torch.save(model.state_dict(), f)
    f.close()

if __name__ == '__main__':
    train()
