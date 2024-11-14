import json
import torch


class TrainDataLoader(object):
    '''
    data loader for training
    '''
    def __init__(self, traget_domain, domain):
        self.traget_domain = traget_domain
        self.domain_num = len(domain)
        self.batch_size = 32
        self.ptr = 0
        self.data = []
        with open('config.txt') as i_f:
            i_f.readline()
            self.student_n, self.exer_n, self.knowledge_n = list(map(eval, i_f.readline().split(',')))
        data_file = './data/' + self.traget_domain + '/train.json'
        config_file = 'config.txt'
        with open('./data/stu_list.json', encoding='utf8') as i_f:
            self.all_stu_list = json.load(i_f)
        with open('./data/iflytek_exercise_embedding.json', encoding='utf8') as i_f:
            self.exer_embedding = json.load(i_f)
        with open('./data/know_embedding.json', encoding='utf8') as i_f:
            self.know_embedding = json.load(i_f)
        with open(data_file, encoding='utf8') as i_f:
            self.data = json.load(i_f)
        with open(config_file) as i_f:
            i_f.readline()
            _, _, knowledge_n = i_f.readline().split(',')
        self.knowledge_dim = int(knowledge_n)

    def next_batch(self):
        if self.is_end():
            return None, None, None, None
        input_stu_ids, input_stu_ids_other, input_stu_ids_sha, input_exer_ids, input_exer_ids_other, input_Q, input_Q_other, input_Q_sha, ys_spe, ys_spe_other, ys_sha, domain, exer_emb, exer_emb_other, exer_emb_sha, know_emb, know_emb_other, know_emb_sha = [], [], [], [], [], [], [], [], [], [], [], [], [], [], [], [], [], []
        for count in range(self.batch_size):
            log = self.data[self.ptr + count]
            Q_emb = [0.] * self.knowledge_dim
            for knowledge_code in [log['knowledge_code']]:
                Q_emb[knowledge_code - 1] = 1.0
            y = log['score']
            input_stu_ids.append(self.student_n*log['domain']+(self.all_stu_list.index(log['user_id'])))
            input_stu_ids_sha.append(self.student_n*log['domain']+(self.all_stu_list.index(log['user_id'])))
            input_exer_ids.append(log['exer_id'] - 1)
            input_Q.append(Q_emb)
            input_Q_sha.append(Q_emb)
            exer_emb.append(self.exer_embedding[log['exer_name']][0])
            know_emb.append(self.know_embedding[str(log['knowledge_code'])])
            exer_emb_sha.append(self.exer_embedding[log['exer_name']][0])
            know_emb_sha.append(self.know_embedding[str(log['knowledge_code'])])
            ys_spe.append(y)
            ys_sha.append(y)
            domain.append(log['domain'])
            for do in range(self.domain_num):
                if do != log['domain']:
                    input_stu_ids_other.append(self.student_n * do + (self.all_stu_list.index(log['user_id'])))
                    input_exer_ids_other.append(log['exer_id'] - 1)
                    input_Q_other.append(Q_emb)
                    know_emb_other.append(self.know_embedding[str(log['knowledge_code'])])
                    exer_emb_other.append(self.exer_embedding[log['exer_name']][0])
                    ys_spe_other.append(y)

                    input_stu_ids_sha.append(self.student_n * do + (self.all_stu_list.index(log['user_id'])))
                    input_Q_sha.append(Q_emb)
                    know_emb_sha.append(self.know_embedding[str(log['knowledge_code'])])
                    exer_emb_sha.append(self.exer_embedding[log['exer_name']][0])
                    ys_sha.append(y)
        self.ptr += self.batch_size
        return torch.LongTensor(input_stu_ids), torch.LongTensor(input_stu_ids_other), torch.LongTensor(input_stu_ids_sha), torch.LongTensor(input_exer_ids), torch.LongTensor(input_exer_ids_other), torch.Tensor(input_Q), torch.Tensor(input_Q_other), torch.Tensor(input_Q_sha), torch.LongTensor(ys_spe), torch.LongTensor(ys_spe_other), torch.LongTensor(ys_sha), torch.Tensor(exer_emb), torch.Tensor(exer_emb_other), torch.Tensor(exer_emb_sha), torch.Tensor(know_emb), torch.Tensor(know_emb_other), torch.Tensor(know_emb_sha), torch.LongTensor(domain)

    def is_end(self):
        if self.ptr + self.batch_size > len(self.data):
            return True
        else:
            return False

    def reset(self):
        self.ptr = 0


class ValTestDataLoader(object):
    def __init__(self, d_type='validation', traget_domain=''):
        self.ptr = 0
        self.data = []
        self.d_type = d_type
        self.traget_domain = traget_domain

        self.domain_num = 6
        self.batch_size = 1

        if d_type == 'validation':
            data_file = './data/' + self.traget_domain + '/val.json'
        else:
            data_file = 'data/test_set.json'

        with open('./data/stu_list.json', encoding='utf8') as i_f:
            self.all_stu_list = json.load(i_f)
        with open('./data/iflytek_exercise_embedding.json', encoding='utf8') as i_f:
            self.exer_embedding = json.load(i_f)
        with open('./data/know_embedding.json', encoding='utf8') as i_f:
            self.know_embedding = json.load(i_f)

        config_file = 'config.txt'
        with open(data_file, encoding='utf8') as i_f:
            self.data = json.load(i_f)
        with open(config_file) as i_f:
            i_f.readline()
            self.student_n, self.exer_n, self.knowledge_n = list(map(eval, i_f.readline().split(',')))
            self.knowledge_dim = int(self.knowledge_n)

    def next_batch(self):
        if self.is_end():
            return None, None, None, None
        input_stu_ids, input_stu_ids_other, input_stu_ids_sha, input_exer_ids, input_exer_ids_other, input_Q, input_Q_other, input_Q_sha, ys_spe, ys_spe_other, ys_sha, domain, exer_emb, exer_emb_other, exer_emb_sha, know_emb, know_emb_other, know_emb_sha = [], [], [], [], [], [], [], [], [], [], [], [], [], [], [], [], [], []
        for count in range(self.batch_size):
            log = self.data[self.ptr + count]
            Q_emb = [0.] * self.knowledge_dim
            for knowledge_code in [log['knowledge_code']]:
                Q_emb[knowledge_code - 1] = 1.0
            y = log['score']
            input_stu_ids.append(self.student_n * log['domain'] + (self.all_stu_list.index(log['user_id'])))
            input_stu_ids_sha.append(self.student_n * log['domain'] + (self.all_stu_list.index(log['user_id'])))
            input_exer_ids.append(log['exer_id'] - 1)
            input_Q.append(Q_emb)
            input_Q_sha.append(Q_emb)
            exer_emb.append(self.exer_embedding[log['exer_name']][0])
            know_emb.append(self.know_embedding[str(log['knowledge_code'])])
            exer_emb_sha.append(self.exer_embedding[log['exer_name']][0])
            know_emb_sha.append(self.know_embedding[str(log['knowledge_code'])])
            ys_spe.append(y)
            ys_sha.append(y)
            domain.append(log['domain'])
            for do in range(self.domain_num):
                if do != log['domain']:
                    input_stu_ids_other.append(self.student_n * do + (self.all_stu_list.index(log['user_id'])))
                    input_exer_ids_other.append(log['exer_id'] - 1)
                    input_Q_other.append(Q_emb)
                    know_emb_other.append(self.know_embedding[str(log['knowledge_code'])])
                    exer_emb_other.append(self.exer_embedding[log['exer_name']][0])
                    ys_spe_other.append(y)

                    input_stu_ids_sha.append(self.student_n * do + (self.all_stu_list.index(log['user_id'])))
                    input_Q_sha.append(Q_emb)
                    know_emb_sha.append(self.know_embedding[str(log['knowledge_code'])])
                    exer_emb_sha.append(self.exer_embedding[log['exer_name']][0])
                    ys_sha.append(y)
        self.ptr += self.batch_size
        return torch.LongTensor(input_stu_ids), torch.Tensor(input_Q), torch.LongTensor(ys_spe), torch.LongTensor(exer_emb), torch.LongTensor(know_emb)

    def is_end(self):
        if self.ptr + self.batch_size > len(self.data):
            return True
        else:
            return False

    def reset(self):
        self.ptr = 0

class FineTuneDataLoader(object):
    '''
    data loader for training
    '''
    def __init__(self, type, traget_domain, sample_num, peer_num):
        self.batch_size = 32
        self.ptr = 0
        self.data = []
        self.traget_domain = traget_domain
        self.sample_num = sample_num

        with open('config.txt') as i_f:
            i_f.readline()
            self.student_n, self.exer_n, self.knowledge_n = list(map(eval, i_f.readline().split(',')))
        if type == 'fine-tune':
            data_file = './data/' + self.traget_domain + '/fine_tune_' + str(sample_num) + '.json'
        elif type == 'fine-tune-val':
            self.batch_size = 1
            data_file = './data/' + self.traget_domain + '/fine_tune_val_' + str(sample_num) + '.json'
        elif type == 'fine-tune_step_2':
            data_file = './data/' + self.traget_domain + '/simulate_train_' + str(sample_num) + '_peer_' + str(
                peer_num) + '.json'
        elif type == 'fine-tune-val_step_2':
            self.batch_size = 1
            data_file = './data/' + self.traget_domain + '/simulate_val_' + str(sample_num) + '_peer_' + str(
                peer_num) + '.json'
        elif type == 'fine-tune-test_step_2':
            self.batch_size = 1
            data_file = './data/' + self.traget_domain + '/new_data_test_' + str(sample_num) + '.json'
        config_file = 'config.txt'
        with open('./data/stu_list.json', encoding='utf8') as i_f:
            self.all_stu_list = json.load(i_f)
        with open('./data/iflytek_exercise_embedding.json', encoding='utf8') as i_f:
            self.exer_embedding = json.load(i_f)
        with open('./data/know_embedding.json', encoding='utf8') as i_f:
            self.know_embedding = json.load(i_f)
        with open(data_file, encoding='utf8') as i_f:
            self.data = json.load(i_f)
        with open(config_file) as i_f:
            i_f.readline()
            _, _, knowledge_n = i_f.readline().split(',')
        self.knowledge_dim = int(knowledge_n)

    def next_batch(self):
        if self.is_end():
            return None, None, None, None
        input_stu_ids, input_exer_ids, input_Q, ys, exer_emb, know_emb, domain = [], [], [], [], [], [], []
        for count in range(self.batch_size):
            log = self.data[self.ptr + count]
            Q_emb = [0.] * self.knowledge_dim
            for knowledge_code in [log['knowledge_code']]:
                Q_emb[knowledge_code - 1] = 1.0
            y = log['score']

            input_exer_ids.append(log['exer_id'] - 1)
            input_Q.append(Q_emb)
            ys.append(y)
            domain.append(log['domain'])
            input_stu_ids.append(self.all_stu_list.index(log['user_id']))
            exer_emb.append(self.exer_embedding[log['exer_name']][0])
            know_emb.append(self.know_embedding[str(log['knowledge_code'])])
        self.ptr += self.batch_size
        return torch.LongTensor(input_stu_ids), torch.Tensor(input_Q), torch.LongTensor(ys), torch.Tensor(exer_emb), torch.Tensor(know_emb), torch.LongTensor(domain)

    def is_end(self):
        if self.ptr + self.batch_size > len(self.data):
            return True
        else:
            return False

    def reset(self):
        self.ptr = 0

