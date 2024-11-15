import json
import torch


class TrainDataLoader:
    '''
    Data loader for training
    '''
    def __init__(self, traget_domain, domain):
        self.traget_domain = traget_domain
        self.domain_num = len(domain)
        self.batch_size = 32
        self.ptr = 0
        self.data = []
        
        # Load config and data
        self._load_config()
        self._load_data()

    def _load_config(self):
        with open('config.txt') as i_f:
            i_f.readline()
            self.student_n, self.exer_n, self.knowledge_n = list(map(eval, i_f.readline().split(',')))

    def _load_data(self):
        data_file = f'./data/{self.traget_domain}/train.json'
        config_file = 'config.txt'
        
        # Load data files
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
        
        # Initialize batch data containers
        batch_data = self._initialize_batch_data()

        # Populate batch data
        for count in range(self.batch_size):
            log = self.data[self.ptr + count]
            Q_emb = [0.] * self.knowledge_dim
            Q_emb[log['knowledge_code'] - 1] = 1.0  # Set knowledge embedding

            y = log['score']  # User's score

            # Add user, exercise, and knowledge embeddings to batch
            self._add_to_batch_data(batch_data, log, Q_emb, y)

        self.ptr += self.batch_size
        return self._convert_to_tensors(batch_data)

    def _initialize_batch_data(self):
        return {
            'input_stu_ids': [], 'input_stu_ids_other': [], 'input_stu_ids_sha': [],
            'input_exer_ids': [], 'input_exer_ids_other': [], 'input_Q': [],
            'input_Q_other': [], 'input_Q_sha': [], 'ys_spe': [], 'ys_spe_other': [],
            'ys_sha': [], 'domain': [], 'exer_emb': [], 'exer_emb_other': [],
            'exer_emb_sha': [], 'know_emb': [], 'know_emb_other': [], 'know_emb_sha': []
        }

    def _add_to_batch_data(self, batch_data, log, Q_emb, y):
        domain = log['domain']

        # Add current domain data
        batch_data['input_stu_ids'].append(self.student_n * domain + self.all_stu_list.index(log['user_id']))
        batch_data['input_stu_ids_sha'].append(self.student_n * domain + self.all_stu_list.index(log['user_id']))
        batch_data['input_exer_ids'].append(log['exer_id'] - 1)
        batch_data['input_Q'].append(Q_emb)
        batch_data['input_Q_sha'].append(Q_emb)
        batch_data['exer_emb'].append(self.exer_embedding[log['exer_name']][0])
        batch_data['know_emb'].append(self.know_embedding[str(log['knowledge_code'])])
        batch_data['exer_emb_sha'].append(self.exer_embedding[log['exer_name']][0])
        batch_data['know_emb_sha'].append(self.know_embedding[str(log['knowledge_code'])])
        batch_data['ys_spe'].append(y)
        batch_data['ys_sha'].append(y)
        batch_data['domain'].append(domain)

        # Add other domain data
        for do in range(self.domain_num):
            if do != domain:
                batch_data['input_stu_ids_other'].append(self.student_n * do + self.all_stu_list.index(log['user_id']))
                batch_data['input_exer_ids_other'].append(log['exer_id'] - 1)
                batch_data['input_Q_other'].append(Q_emb)
                batch_data['know_emb_other'].append(self.know_embedding[str(log['knowledge_code'])])
                batch_data['exer_emb_other'].append(self.exer_embedding[log['exer_name']][0])
                batch_data['ys_spe_other'].append(y)

                batch_data['input_stu_ids_sha'].append(self.student_n * do + self.all_stu_list.index(log['user_id']))
                batch_data['input_Q_sha'].append(Q_emb)
                batch_data['know_emb_sha'].append(self.know_embedding[str(log['knowledge_code'])])
                batch_data['exer_emb_sha'].append(self.exer_embedding[log['exer_name']][0])
                batch_data['ys_sha'].append(y)

    def _convert_to_tensors(self, batch_data):
        return (torch.LongTensor(batch_data['input_stu_ids']),
                torch.LongTensor(batch_data['input_stu_ids_other']),
                torch.LongTensor(batch_data['input_stu_ids_sha']),
                torch.LongTensor(batch_data['input_exer_ids']),
                torch.LongTensor(batch_data['input_exer_ids_other']),
                torch.Tensor(batch_data['input_Q']),
                torch.Tensor(batch_data['input_Q_other']),
                torch.Tensor(batch_data['input_Q_sha']),
                torch.LongTensor(batch_data['ys_spe']),
                torch.LongTensor(batch_data['ys_spe_other']),
                torch.LongTensor(batch_data['ys_sha']),
                torch.Tensor(batch_data['exer_emb']),
                torch.Tensor(batch_data['exer_emb_other']),
                torch.Tensor(batch_data['exer_emb_sha']),
                torch.Tensor(batch_data['know_emb']),
                torch.Tensor(batch_data['know_emb_other']),
                torch.Tensor(batch_data['know_emb_sha']),
                torch.LongTensor(batch_data['domain']))

    def is_end(self):
        return self.ptr + self.batch_size > len(self.data)

    def reset(self):
        self.ptr = 0


class ValTestDataLoader:
    '''
    Data loader for validation and testing
    '''
    def __init__(self, d_type='validation', traget_domain=''):
        self.ptr = 0
        self.data = []
        self.d_type = d_type
        self.traget_domain = traget_domain

        self.domain_num = 6
        self.batch_size = 1

        data_file = f'./data/{self.traget_domain}/{d_type}.json' if d_type == 'validation' else 'data/test_set.json'

        self._load_config()
        self._load_data(data_file)

    def _load_config(self):
        with open('config.txt') as i_f:
            i_f.readline()
            self.student_n, self.exer_n, self.knowledge_n = list(map(eval, i_f.readline().split(',')))
            self.knowledge_dim = int(self.knowledge_n)

    def _load_data(self, data_file):
        with open('./data/stu_list.json', encoding='utf8') as i_f:
            self.all_stu_list = json.load(i_f)
        with open('./data/iflytek_exercise_embedding.json', encoding='utf8') as i_f:
            self.exer_embedding = json.load(i_f)
        with open('./data/know_embedding.json', encoding='utf8') as i_f:
            self.know_embedding = json.load(i_f)
        with open(data_file, encoding='utf8') as i_f:
            self.data = json.load(i_f)

    def next_batch(self):
        if self.is_end():
            return None, None, None, None
        
        batch_data = self._initialize_batch_data()

        # Populate batch data
        for count in range(self.batch_size):
            log = self.data[self.ptr + count]
            Q_emb = [0.] * self.knowledge_dim
            Q_emb[log['knowledge_code'] - 1] = 1.0  # Set knowledge embedding

            y = log['score']

            # Add user, exercise, and knowledge embeddings to batch
            self._add_to_batch_data(batch_data, log, Q_emb, y)

        self.ptr += self.batch_size
        return self._convert_to_tensors(batch_data)

    def _initialize_batch_data(self):
        return {
            'input_stu_ids': [], 'input_exer_ids': [], 'input_Q': [],
            'ys_spe': [], 'exer_emb': [], 'know_emb': [], 'domain': []
        }

    def _add_to_batch_data(self, batch_data, log, Q_emb, y):
        domain = log['domain']

        # Add current domain data
        batch_data['input_stu_ids'].append(self.student_n * domain + self.all_stu_list.index(log['user_id']))
        batch_data['input_exer_ids'].append(log['exer_id'] - 1)
        batch_data['input_Q'].append(Q_emb)
        batch_data['exer_emb'].append(self.exer_embedding[log['exer_name']][0])
        batch_data['know_emb'].append(self.know_embedding[str(log['knowledge_code'])])
        batch_data['ys_spe'].append(y)
        batch_data['domain'].append(domain)

    def _convert_to_tensors(self, batch_data):
        return (torch.LongTensor(batch_data['input_stu_ids']),
                torch.LongTensor(batch_data['input_exer_ids']),
                torch.Tensor(batch_data['input_Q']),
                torch.LongTensor(batch_data['ys_spe']),
                torch.Tensor(batch_data['exer_emb']),
                torch.Tensor(batch_data['know_emb']),
                torch.LongTensor(batch_data['domain']))

    def is_end(self):
        return self.ptr + self.batch_size > len(self.data)

    def reset(self):
        self.ptr = 0


class FineTuneDataLoader:
    '''
    Data loader for fine-tuning
    '''
    def __init__(self, traget_domain=''):
        self.ptr = 0
        self.batch_size = 32
        self.data = []
        self.traget_domain = traget_domain

        self._load_config()
        self._load_data()

    def _load_config(self):
        with open('config.txt') as i_f:
            i_f.readline()
            self.student_n, self.exer_n, self.knowledge_n = list(map(eval, i_f.readline().split(',')))

    def _load_data(self):
        data_file = f'./data/{self.traget_domain}/fine_tune.json'

        # Load data files
        with open('./data/stu_list.json', encoding='utf8') as i_f:
            self.all_stu_list = json.load(i_f)
        with open('./data/iflytek_exercise_embedding.json', encoding='utf8') as i_f:
            self.exer_embedding = json.load(i_f)
        with open('./data/know_embedding.json', encoding='utf8') as i_f:
            self.know_embedding = json.load(i_f)
        with open(data_file, encoding='utf8') as i_f:
            self.data = json.load(i_f)

    def next_batch(self):
        if self.is_end():
            return None, None, None, None
        
        batch_data = self._initialize_batch_data()

        # Populate batch data
        for count in range(self.batch_size):
            log = self.data[self.ptr + count]
            Q_emb = [0.] * self.knowledge_dim
            Q_emb[log['knowledge_code'] - 1] = 1.0  # Set knowledge embedding

            y = log['score']

            # Add user, exercise, and knowledge embeddings to batch
            self._add_to_batch_data(batch_data, log, Q_emb, y)

        self.ptr += self.batch_size
        return self._convert_to_tensors(batch_data)

    def _initialize_batch_data(self):
        return {
            'input_stu_ids': [], 'input_exer_ids': [], 'input_Q': [],
            'ys_spe': [], 'exer_emb': [], 'know_emb': [], 'domain': []
        }

    def _add_to_batch_data(self, batch_data, log, Q_emb, y):
        domain = log['domain']

        # Add current domain data
        batch_data['input_stu_ids'].append(self.student_n * domain + self.all_stu_list.index(log['user_id']))
        batch_data['input_exer_ids'].append(log['exer_id'] - 1)
        batch_data['input_Q'].append(Q_emb)
        batch_data['exer_emb'].append(self.exer_embedding[log['exer_name']][0])
        batch_data['know_emb'].append(self.know_embedding[str(log['knowledge_code'])])
        batch_data['ys_spe'].append(y)
        batch_data['domain'].append(domain)

    def _convert_to_tensors(self, batch_data):
        return (torch.LongTensor(batch_data['input_stu_ids']),
                torch.LongTensor(batch_data['input_exer_ids']),
                torch.Tensor(batch_data['input_Q']),
                torch.LongTensor(batch_data['ys_spe']),
                torch.Tensor(batch_data['exer_emb']),
                torch.Tensor(batch_data['know_emb']),
                torch.LongTensor(batch_data['domain']))

    def is_end(self):
        return self.ptr + self.batch_size > len(self.data)

    def reset(self):
        self.ptr = 0
