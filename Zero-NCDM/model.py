import torch
import torch.nn as nn


class Net(nn.Module):
    '''
    NeuralCDM
    '''
    def __init__(self, student_n, exer_n, knowledge_n, domain, device):
        self.knowledge_dim = 256
        self.exer_n = exer_n
        self.emb_num = student_n
        self.stu_dim = self.knowledge_dim
        self.prednet_input_len = self.knowledge_dim
        self.prednet_len1, self.prednet_len2 = 512, 256  # changeable
        self.device = device
        self.know_num = knowledge_n
        self.domain_num = len(domain)

        super(Net, self).__init__()

        # network structure
        self.student_emb_spe = nn.Embedding(self.emb_num*self.domain_num, self.stu_dim)
        self.student_emb_sha = nn.Embedding(self.emb_num*self.domain_num, self.stu_dim)
        # self.k_difficulty = nn.Embedding(self.exer_n, self.knowledge_dim)
        # self.e_discrimination = nn.Embedding(self.exer_n, 1)
        self.diff = nn.Linear(768, self.knowledge_dim)
        self.know = nn.Linear(768, self.knowledge_dim)
        self.disc = nn.Linear(768, 1)

        self.full_stu = nn.Linear(2*self.knowledge_dim, self.know_num)
        self.full_exer = nn.Linear(2*self.knowledge_dim, self.know_num)

        self.prednet_full1 = nn.Linear(self.know_num, self.prednet_len1)
        # self.drop_1 = nn.Dropout(p=0.5)
        self.prednet_full2 = nn.Linear(self.prednet_len1, 1)
        # self.drop_2 = nn.Dropout(p=0.5)
        # self.prednet_full3 = nn.Linear(self.prednet_len2, 1)

        # initialization
        for name, param in self.named_parameters():
            if 'weight' in name:
                nn.init.xavier_normal_(param)

    def forward(self, stu_id_spe, stu_ids_spe_other, stu_ids_glo_sha, Q_emb_spe, Q_emb_spe_other, Q_emb_sha, exer_emb_spe, exer_emb_spe_other, exer_emb_sha, know_emb_spe, know_emb_spe_other, know_emb_sha, input_domain):
        '''
        :param stu_id: LongTensor
        :param exer_id: LongTensor
        :param kn_emb: FloatTensor, the knowledge relevancy vectors
        :return: FloatTensor, the probabilities of answering correctly
        '''
        # before prednet
        stu_emb_spe = torch.sigmoid(self.student_emb_spe(stu_id_spe))
        stu_emb_glo_sha = self.student_emb_sha(stu_ids_glo_sha)
        stu_emb_sha = self.student_emb_sha(stu_id_spe)
        stu_emb_spe_other = self.student_emb_spe(stu_ids_spe_other)

        exe_emb_spe = torch.sigmoid(self.diff(exer_emb_spe.to(self.device)))
        k_emb_spe =torch.sigmoid( self.know(know_emb_spe.to(self.device)))

        exe_emb_spe_other = self.diff(exer_emb_spe_other.to(self.device))
        k_emb_spe_other = self.know(know_emb_spe_other.to(self.device))

        exe_emb_glo_sha = self.diff(exer_emb_sha.to(self.device))
        k_emb_glo_sha = self.know(know_emb_sha.to(self.device))

        e_discrimination_spe = torch.sigmoid(torch.sigmoid(self.disc(exer_emb_spe.to(self.device)))) * 10
        e_discrimination_spe_other = torch.sigmoid(self.disc(exer_emb_spe_other.to(self.device))) * 10
        e_discrimination_glo_sha = torch.sigmoid(self.disc(exer_emb_sha.to(self.device))) * 10

        stu_emb_pro_spe = torch.ones([stu_id_spe.shape[0], self.know_num]).to(self.device)
        pro_value_spe = torch.sigmoid(self.full_stu(torch.cat([stu_emb_spe, k_emb_spe], dim=1)))
        proficiency_spe = stu_emb_pro_spe * pro_value_spe

        exe_emb_diff_spe = torch.ones([exer_emb_spe.shape[0], self.know_num]).to(self.device)
        diff_value_spe = torch.sigmoid(self.full_exer(torch.cat([exe_emb_spe, k_emb_spe], dim=1)))
        k_difficulty_spe = exe_emb_diff_spe * diff_value_spe
        # print (pro_value_spe)
        # print (diff_value_spe)
        # print()
        # prednet
        input_x = e_discrimination_spe*(pro_value_spe - diff_value_spe) * Q_emb_spe
        input_x = torch.sigmoid(self.prednet_full1(input_x))
        output_spe = torch.sigmoid(self.prednet_full2(input_x)).squeeze(1)


        # global share
        stu_emb_pro_glo_sha = torch.ones([stu_ids_glo_sha.shape[0], self.know_num]).to(self.device)
        pro_value_glo_sha = torch.sigmoid(self.full_stu(torch.cat([stu_emb_glo_sha, k_emb_glo_sha], dim=1)))
        proficiency_glo_sha = stu_emb_pro_glo_sha * pro_value_glo_sha

        exe_emb_diff_glo_sha = torch.ones([exe_emb_glo_sha.shape[0], self.know_num]).to(self.device)
        diff_value_glo_sha = torch.sigmoid(self.full_exer(torch.cat([exe_emb_glo_sha, k_emb_glo_sha], dim=1)))
        k_difficulty_glo_sha = exe_emb_diff_glo_sha * diff_value_glo_sha

        # prednet
        input_x1 = e_discrimination_glo_sha * (pro_value_glo_sha - diff_value_glo_sha) * Q_emb_sha
        input_x1 = torch.sigmoid(self.prednet_full1(input_x1))
        output_glo_sha = torch.sigmoid(self.prednet_full2(input_x1)).squeeze(1)

        # share
        stu_emb_pro_sha = torch.ones([stu_id_spe.shape[0], self.know_num]).to(self.device)
        pro_value_sha = torch.sigmoid(self.full_stu(torch.cat([stu_emb_sha, k_emb_spe], dim=1)))
        proficiency_sha = stu_emb_pro_sha * pro_value_sha

        # print (proficiency_spe)
        # print()

        # prednet
        input_x2 = e_discrimination_spe * (pro_value_sha - diff_value_spe) * Q_emb_spe
        input_x2 = torch.sigmoid(self.prednet_full1(input_x2))
        output_sha = torch.sigmoid(self.prednet_full2(input_x2)).squeeze(1)

        # spe_other
        stu_emb_pro_spe_other = torch.ones([stu_ids_spe_other.shape[0], self.know_num]).to(self.device)
        pro_value_spe_other = torch.sigmoid(self.full_stu(torch.cat([stu_emb_spe_other, k_emb_spe_other], dim=1)))
        proficiency_spe_other = stu_emb_pro_spe_other * pro_value_spe_other

        exe_emb_diff_spe_other = torch.ones([exer_emb_spe_other.shape[0], self.know_num]).to(self.device)
        diff_value_spe_other = torch.sigmoid(self.full_exer(torch.cat([exe_emb_spe_other, k_emb_spe_other], dim=1)))
        k_difficulty_spe_other = exe_emb_diff_spe_other * diff_value_spe_other

        # prednet
        input_x3 = e_discrimination_spe_other * (pro_value_spe_other - diff_value_spe_other) * Q_emb_spe_other
        input_x3 = torch.sigmoid(self.prednet_full1(input_x3))
        output_spe_other = torch.sigmoid(self.prednet_full2(input_x3)).squeeze(1)
        return output_spe, output_spe_other, output_glo_sha, output_sha


    def val(self, stu_id_spe, Q_emb_spe, exer_emb_spe, know_emb_spe):
        '''
        :param stu_id: LongTensor
        :param exer_id: LongTensor
        :param kn_emb: FloatTensor, the knowledge relevancy vectors
        :return: FloatTensor, the probabilities of answering correctly
        '''
        # before prednet
        stu_emb_spe = torch.sigmoid(self.student_emb_spe(stu_id_spe))
        stu_emb_sha = self.student_emb_sha(stu_id_spe)

        # print (exer_emb_spe.shape)
        exe_emb_spe = torch.sigmoid(self.diff(exer_emb_spe.float().to(self.device)))
        k_emb_spe =torch.sigmoid( self.know(know_emb_spe.float().to(self.device)))

        e_discrimination_spe = torch.sigmoid(torch.sigmoid(self.disc(exer_emb_spe.float().to(self.device)))) * 10

        stu_emb_pro_spe = torch.ones([stu_id_spe.shape[0], self.know_num]).to(self.device)
        pro_value_spe = torch.sigmoid(self.full_stu(torch.cat([stu_emb_spe, k_emb_spe], dim=1)))
        proficiency_spe = stu_emb_pro_spe * pro_value_spe

        exe_emb_diff_spe = torch.ones([exer_emb_spe.shape[0], self.know_num]).to(self.device)
        diff_value_spe = torch.sigmoid(self.full_exer(torch.cat([exe_emb_spe, k_emb_spe], dim=1)))
        k_difficulty_spe = exe_emb_diff_spe * diff_value_spe

        input_x = e_discrimination_spe*(pro_value_spe - diff_value_spe) * Q_emb_spe
        input_x = torch.sigmoid(self.prednet_full1(input_x))
        output_spe = torch.sigmoid(self.prednet_full2(input_x)).squeeze(1)


        # share
        stu_emb_pro_sha = torch.ones([stu_id_spe.shape[0], self.know_num]).to(self.device)
        pro_value_sha = torch.sigmoid(self.full_stu(torch.cat([stu_emb_sha, k_emb_spe], dim=1)))
        proficiency_sha = stu_emb_pro_sha * pro_value_sha

        # print (proficiency_spe)
        # print()

        # prednet
        input_x2 = e_discrimination_spe * (pro_value_sha - diff_value_spe) * Q_emb_spe
        input_x2 = torch.sigmoid(self.prednet_full1(input_x2))
        output_sha = torch.sigmoid(self.prednet_full2(input_x2)).squeeze(1)

        # print (output_spe.shape)
        # print (output_sha.shape)
        # print ((output_spe + output_sha).shape)
        # print ((output_spe + output_sha).shape)

        return (output_spe + output_sha) / 2

    def fine_tune(self, stu_id, Q_emb, exer_emb, know_emb, domain_id):
        '''
        :param stu_id: LongTensor
        :param exer_id: LongTensor
        :param kn_emb: FloatTensor, the knowledge relevancy vectors
        :return: FloatTensor, the probabilities of answering correctly
        '''
        # before prednet
        # self.student_n * do + (self.all_stu_list.index(log['user_id']))
        stu_emb_temp = torch.zeros(self.student_emb_spe(stu_id).shape).to(self.device)
        for i in range(self.domain_num):
            if i != domain_id:
                # print (torch.sigmoid(self.student_emb_spe(self.emb_num * i + stu_id)))
                stu_emb_temp += torch.sigmoid(self.student_emb_spe(self.emb_num * i + stu_id))
        stu_emb = stu_emb_temp/(self.domain_num-1)
        # print(stu_emb)
        # print ()

        # print (exer_emb_spe.shape)
        exe_emb = torch.sigmoid(self.diff(exer_emb.float().to(self.device)))
        k_emb =torch.sigmoid( self.know(know_emb.float().to(self.device)))

        e_discrimination = torch.sigmoid(torch.sigmoid(self.disc(exer_emb.float().to(self.device)))) * 10

        pro_value = torch.sigmoid(self.full_stu(torch.cat([stu_emb, k_emb], dim=1)))

        diff_value = torch.sigmoid(self.full_exer(torch.cat([exe_emb, k_emb], dim=1)))

        input_x = e_discrimination*(pro_value - diff_value) * Q_emb
        input_x = torch.sigmoid(self.prednet_full1(input_x))
        output = torch.sigmoid(self.prednet_full2(input_x)).squeeze(1)
        # print (output)

        return output

    def get_student_emb(self):
        stu_id = torch.LongTensor(list(range(self.emb_num * self.domain_num))).to(self.device)
        student_emb_spe = torch.sigmoid(self.student_emb_spe(stu_id))
        student_emb_sha = torch.sigmoid(self.student_emb_sha(stu_id))
        return student_emb_spe.data, student_emb_sha.data

    def get_student_emb_spe(self):
        student_emb_spe_0 = torch.sigmoid(
            self.student_emb_spe(torch.LongTensor(list(range(self.emb_num * 1))).to(self.device)))[0:500]
        student_emb_spe_1 = torch.sigmoid(
            self.student_emb_spe(torch.LongTensor(list(range(self.emb_num * 1, self.emb_num * 2))).to(self.device)))[0:500]
        student_emb_spe_2 = torch.sigmoid(
            self.student_emb_spe(torch.LongTensor(list(range(self.emb_num * 2, self.emb_num * 3))).to(self.device)))[0:500]
        student_emb_spe_3 = torch.sigmoid(
            self.student_emb_spe(torch.LongTensor(list(range(self.emb_num * 3, self.emb_num * 4))).to(self.device)))[0:500]
        student_emb_spe_4 = torch.sigmoid(
            self.student_emb_spe(torch.LongTensor(list(range(self.emb_num * 4, self.emb_num * 5))).to(self.device)))[0:500]
        student_emb_spe_5 = torch.sigmoid(
            self.student_emb_spe(torch.LongTensor(list(range(self.emb_num * 5, self.emb_num * 6))).to(self.device)))[0:500]
        return student_emb_spe_0.data, student_emb_spe_1.data, student_emb_spe_2.data, student_emb_spe_3.data, student_emb_spe_4.data, student_emb_spe_5.data

    def get_student_emb_sha(self):
        student_emb_sha_0 = torch.sigmoid(self.student_emb_sha(torch.LongTensor(list(range(self.emb_num*1))).to(self.device)))[0:500]
        student_emb_sha_1 = torch.sigmoid(self.student_emb_sha(torch.LongTensor(list(range(self.emb_num * 1, self.emb_num*2))).to(self.device)))[0:500]
        student_emb_sha_2 = torch.sigmoid(self.student_emb_sha(torch.LongTensor(list(range(self.emb_num * 2, self.emb_num*3))).to(self.device)))[0:500]
        student_emb_sha_3 = torch.sigmoid(self.student_emb_sha(torch.LongTensor(list(range(self.emb_num * 3, self.emb_num*4))).to(self.device)))[0:500]
        student_emb_sha_4 = torch.sigmoid(self.student_emb_sha(torch.LongTensor(list(range(self.emb_num * 4, self.emb_num*5))).to(self.device)))[0:500]
        student_emb_sha_5 = torch.sigmoid(self.student_emb_sha(torch.LongTensor(list(range(self.emb_num * 5, self.emb_num*6))).to(self.device)))[0:500]
        return student_emb_sha_0.data, student_emb_sha_1.data, student_emb_sha_2.data, student_emb_sha_3.data, student_emb_sha_4.data, student_emb_sha_5.data

    def get_student_emb_tune(self, domain_id):
        stu_id = torch.LongTensor(list(range(self.emb_num))).to(self.device)
        stu_emb_temp = torch.zeros(self.student_emb_spe(stu_id).shape)
        for i in range(self.domain_num):
            if i != domain_id:
                stu_emb_temp += torch.sigmoid(self.student_emb_spe(self.emb_num * i + stu_id))
        return stu_emb_temp/(self.domain_num-1)

    def apply_clipper(self):
        clipper = NoneNegClipper()
        self.prednet_full1.apply(clipper)
        self.prednet_full2.apply(clipper)
        # self.prednet_full3.apply(clipper)

    def get_knowledge_status(self, stu_id):
        stat_emb = torch.sigmoid(self.student_emb(stu_id))
        return stat_emb.data

    def get_exer_params(self, exer_id):
        k_difficulty = torch.sigmoid(self.k_difficulty(exer_id))
        e_discrimination = torch.sigmoid(self.e_discrimination(exer_id)) * 10
        return k_difficulty.data, e_discrimination.data


class NoneNegClipper(object):
    def __init__(self):
        super(NoneNegClipper, self).__init__()

    def __call__(self, module):
        if hasattr(module, 'weight'):
            w = module.weight.data
            a = torch.relu(torch.neg(w))
            w.add_(a)
