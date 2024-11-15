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

with open('./data/stu_list.json', encoding='utf8') as i_f:
    all_stu_list = json.load(i_f)

domain = ['geometry', 'function', 'probability', 'physics', 'arithmetic', 'english']
domain_stu = [15283, 15404, 4076, 13369, 14073, 5906]

domain_id = 0
select_epoch = 3
select_epoch_tune = 4

domain_id = 1
select_epoch = 1
select_epoch_tune = 7

domain_id = 2
select_epoch = 1
select_epoch_tune = 15

domain_id = 3
select_epoch = 1
select_epoch_tune = 9

domain_id = 4
select_epoch = 1
select_epoch_tune = 1

domain_id = 5
select_epoch = 1
select_epoch_tune = 23

traget_domain = domain[domain_id]
sample_num = int(domain_stu[domain_id] * 0.01)
peer_num = 50  # can adjust peer_num as needed
# peer_num = 10
# peer_num = 30
# peer_num = 100
# peer_num = 300
# peer_num = 500

device = torch.device('cpu')

def augment(epoch, epoch_tune):
    print('augment...')
    net = Net(student_n, exer_n, knowledge_n, domain, device)
    load_snapshot(net, 'model/' + traget_domain + '/model_epoch' + str(epoch))
    net = net.to(device)
    net.eval()

    net_tune = Net(student_n, exer_n, knowledge_n, domain, device)
    load_snapshot(net_tune, 'model/' + traget_domain + '/fine_tune_' + str(sample_num) + '_model_F_epoch' + str(epoch_tune))
    net_tune = net_tune.to(device)
    net_tune.eval()

    anchors, anchor_logs_train, anchor_logs_val, anchor_logs_test = get_anchor()
    student_emb_spe, student_emb_sha = net.get_student_emb()
    get_student_emb_tune = net_tune.get_student_emb_tune(domain_id)  # torch.Size([21068, 256])

    simu_log_train = []
    simu_log_val = []
    for i in anchors:
        similar_domain = find_domain(i, get_student_emb_tune[i].detach().numpy(), student_emb_spe.detach().numpy())
        peer = find_peer(i, similar_domain, student_emb_spe.detach().numpy())
        log_train, log_val = generate_data(i, peer, anchor_logs_train, anchor_logs_val)
        simu_log_train += log_train
        simu_log_val += log_val

    with open('data/' + traget_domain + '/simulate_train_' + str(sample_num) + '_peer_' + str(peer_num) + '.json', 'w', encoding='utf-8') as f:
        json.dump(simu_log_train, f, indent=4)
    
    with open('data/' + traget_domain + '/simulate_val_' + str(sample_num) + '_peer_' + str(peer_num) + '.json', 'w', encoding='utf-8') as f:
        json.dump(simu_log_val, f, indent=4)


def generate_data(anchor_id, peer, anchor_logs_train, anchor_logs_val):
    log_train = []
    log_val = []
    
    for l in anchor_logs_train:
        if all_stu_list[anchor_id] == l['user_id']:
            for p in peer:
                log_train.append({
                    "user_id": all_stu_list[p],
                    "exer_id": l['exer_id'],
                    "score": l['score'],
                    "domain": l['domain'],
                    "exer_name": l['exer_name'],
                    "knowledge_code": l['knowledge_code']
                })
    
    for l in anchor_logs_val:
        if all_stu_list[anchor_id] == l['user_id']:
            for p in peer:
                log_val.append({
                    "user_id": all_stu_list[p],
                    "exer_id": l['exer_id'],
                    "score": l['score'],
                    "domain": l['domain'],
                    "exer_name": l['exer_name'],
                    "knowledge_code": l['knowledge_code']
                })

    return log_train, log_val


def find_peer(anchor_id, similar_domain, student_emb_spe):
    peer = []
    peer_score = {}
    anchor_emb = student_emb_spe[student_n * similar_domain + anchor_id]
    
    for stu in range(student_n):
        vec = student_emb_spe[student_n * similar_domain + stu]
        sim_score = anchor_emb.dot(vec) / (np.linalg.norm(anchor_emb) * np.linalg.norm(vec))
        peer_score[stu] = sim_score

    peer_score = sorted(peer_score.items(), key=lambda x: x[1], reverse=True)
    
    for p in peer_score[:peer_num]:
        peer.append(p[0])
    
    return peer


def find_domain(anchor_id, anchor_emb, student_emb_spe):
    sim = -100
    sim_id = -100
    
    for do in range(len(domain)):
        if do != domain_id:
            vec = student_emb_spe[student_n * do + anchor_id]
            sim_score = anchor_emb.dot(vec) / (np.linalg.norm(anchor_emb) * np.linalg.norm(vec))
            
            if sim < sim_score:
                sim = sim_score
                sim_id = do

    return sim_id


def get_anchor():
    with open('data/' + traget_domain + '/fine_tune_' + str(sample_num) + '.json') as f:
        anchor_logs_train = json.load(f)
    
    with open('data/' + traget_domain + '/fine_tune_val_' + str(sample_num) + '.json') as f:
        anchor_logs_val = json.load(f)
    
    with open('data/' + traget_domain + '/new_data_test_' + str(sample_num) + '.json') as f:
        anchor_logs_test = json.load(f)
    
    anchors = []
    for l in anchor_logs_train:
        anchors.append(all_stu_list.index(l['user_id']))
    
    anchors = list(set(anchors))
    return anchors, anchor_logs_train, anchor_logs_val, anchor_logs_test


def load_snapshot(model, filename):
    with open(filename, 'rb') as f:
        model.load_state_dict(torch.load(f, map_location=lambda s, loc: s))


if __name__ == '__main__':
    augment(select_epoch, select_epoch_tune)
