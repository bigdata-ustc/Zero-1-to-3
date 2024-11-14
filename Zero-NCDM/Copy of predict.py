import torch
import numpy as np
import json
import sys
from sklearn.metrics import roc_auc_score, f1_score, precision_score, recall_score
from data_loader import ValTestDataLoader
from model import Net

# can be changed according to config.txt
with open('config.txt') as i_f:
    i_f.readline()
    student_n, exer_n, knowledge_n = list(map(eval, i_f.readline().split(',')))

domain = ['geometry', 'function', 'probability', 'physics', 'arithmetic', 'english']
domain_id = 5
traget_domain = domain[domain_id]
print(domain_id)
print ()
def validate(epoch):
    device = torch.device('cpu')
    data_loader = ValTestDataLoader('validation', traget_domain)
    net = Net(student_n, exer_n, knowledge_n, domain, device)
    print('validate model...')
    data_loader.reset()
    load_snapshot(net, 'model/' + traget_domain +'/model_epoch' + str(epoch))
    net = net.to(device)
    net.eval()

    correct_count, exer_count = 0, 0
    pred_all, label_all, binary_pre = [], [], []
    while not data_loader.is_end():
        # input_stu_ids, input_exer_ids, input_knowledge_embs, labels = data_loader.next_batch()
        # input_stu_ids, input_exer_ids, input_knowledge_embs, labels = input_stu_ids.to(device), input_exer_ids.to(
        #     device), input_knowledge_embs.to(device), labels.to(device)
        # out_put = net(input_stu_ids, input_exer_ids, input_knowledge_embs)

        input_stu_ids, Q_embs, ys_spe, exer_emb, know_emb = data_loader.next_batch()
        input_stu_ids, Q_embs, ys_spe, exer_emb, know_emb = input_stu_ids.to(device), Q_embs.to(device), ys_spe.to(device), exer_emb.to(device), know_emb.to(device)
        out_put = net.val(input_stu_ids, Q_embs, exer_emb, know_emb)


        out_put = out_put.view(-1)
        # compute accuracy
        for i in range(len(ys_spe)):
            if (ys_spe[i] == 1 and out_put[i] > 0.5) or (ys_spe[i] == 0 and out_put[i] < 0.5):
                correct_count += 1
            if out_put[i] >= 0.5:
                binary_pre.append(int(1))
            else:
                binary_pre.append(int(0))
        exer_count += len(ys_spe)
        pred_all += out_put.tolist()
        label_all += ys_spe.tolist()


    pred_all = np.array(pred_all)
    label_all = np.array(label_all)
    binary_pre = np.array(binary_pre)
    # compute accuracy
    accuracy = correct_count / exer_count
    # compute RMSE
    rmse = np.sqrt(np.mean((label_all - pred_all) ** 2))
    mae = np.mean(np.sqrt((label_all - pred_all) ** 2))
    # compute AUC
    auc = roc_auc_score(label_all, pred_all)
    f1 = f1_score(label_all, binary_pre)
    precision = precision_score(label_all, binary_pre)
    recall = recall_score(label_all, binary_pre)
    # print('epoch= %d, accuracy= %f, rmse= %f, auc= %f' % (epoch, accuracy, rmse, auc))
    print('domain: %s, accuracy= %f, rmse= %f, auc= %f, f1_score=%f, precision=%f, recall=%f, mae=%f' % (traget_domain, accuracy, rmse, auc, f1, precision, recall, mae))

    with open('result/' + traget_domain +'/model_val.txt', 'a', encoding='utf8') as f:
        f.write('epoch= %d, accuracy= %f, rmse= %f, auc= %f, f1_score=%f, precision=%f, recall=%f, mae=%f\n' % (epoch, accuracy, rmse, auc, f1, precision, recall,mae))

def load_snapshot(model, filename):
    f = open(filename, 'rb')
    model.load_state_dict(torch.load(f, map_location=lambda s, loc: s))
    f.close()

if __name__ == '__main__':
    for i in range(20):
        print (i+1)
        validate(i+1)
