import torch
import torch.nn as nn
import torch.optim as optim
import numpy as np
import json
import sys
from sklearn.metrics import roc_auc_score
from data_loader import TrainDataLoader, ValTestDataLoader
from model import Net


# Configuration loading from config.txt
with open('config.txt') as i_f:
    i_f.readline()
    student_n, exer_n, knowledge_n = list(map(eval, i_f.readline().split(',')))

# Device setup
device = torch.device(('cuda:2') if torch.cuda.is_available() else 'cpu')

# Domain setup
domain = ['geometry', 'function', 'probability', 'physics', 'arithmetic', 'english']
domain_id = 5
traget_domain = domain[domain_id]
epoch_n = 20

print(domain_id)

def train():
    data_loader = TrainDataLoader(traget_domain, domain)
    net = Net(student_n, exer_n, knowledge_n, domain, device)

    # Model and optimizer setup
    net = net.to(device)
    optimizer = optim.Adam(net.parameters(), lr=0.001)
    print('Training model...')

    loss_function = nn.MSELoss()

    # Training loop
    for epoch in range(epoch_n):
        data_loader.reset()
        running_loss = 0.0
        batch_count = 0
        
        while not data_loader.is_end():
            batch_count += 1

            # Load batch data and move to device
            (
                input_stu_ids, input_stu_ids_other, input_stu_ids_sha,
                input_exer_ids, input_exer_ids_other, Q_embs, Q_embs_other,
                Q_embs_sha, ys_spe, ys_spe_other, ys_sha, exer_emb, exer_emb_other,
                exer_emb_sha, know_emb, know_emb_other, know_emb_sha, input_domain
            ) = data_loader.next_batch()

            input_stu_ids, input_stu_ids_other, input_stu_ids_sha = (
                input_stu_ids.to(device), input_stu_ids_other.to(device),
                input_stu_ids_sha.to(device)
            )
            input_exer_ids, input_exer_ids_other = (
                input_exer_ids.to(device), input_exer_ids_other.to(device)
            )
            Q_embs, Q_embs_other, Q_embs_sha = (
                Q_embs.to(device), Q_embs_other.to(device), Q_embs_sha.to(device)
            )
            ys_spe, ys_spe_other, ys_sha = (
                ys_spe.to(device), ys_spe_other.to(device), ys_sha.to(device)
            )
            exer_emb, exer_emb_other, exer_emb_sha = (
                exer_emb.to(device), exer_emb_other.to(device), exer_emb_sha.to(device)
            )
            know_emb, know_emb_other, know_emb_sha = (
                know_emb.to(device), know_emb_other.to(device), know_emb_sha.to(device)
            )
            input_domain = input_domain.to(device)

            # Zero gradients
            optimizer.zero_grad()

            # Forward pass
            output_spe, output_spe_other, output_glo_sha, output_sha = net.forward(
                input_stu_ids, input_stu_ids_other, input_stu_ids_sha,
                Q_embs, Q_embs_other, Q_embs_sha, exer_emb, exer_emb_other,
                exer_emb_sha, know_emb, know_emb_other, know_emb_sha, input_domain
            )

            # Compute loss
            loss = 1 + loss_function(output_spe.float(), ys_spe.float()) - \
                   loss_function(output_spe_other.float(), ys_spe_other.float()) + \
                   loss_function(output_glo_sha.float(), ys_sha.float()) - \
                   loss_function(output_sha.float(), ys_spe.float())

            # Backpropagation and optimizer step
            loss.backward()
            optimizer.step()
            net.apply_clipper()

            # Update running loss
            running_loss += loss.item()
            
            # Print every 200 batches
            if batch_count % 200 == 199:
                print(f'[{epoch+1}, {batch_count+1}] loss: {running_loss / 200:.10f}')
                running_loss = 0.0

        # Save model snapshot after every epoch
        save_snapshot(net, f'model/{traget_domain}/model_epoch{epoch + 1}')


def validate(model, epoch):
    data_loader = ValTestDataLoader('validation')
    net = Net(student_n, exer_n, knowledge_n)
    print('Validating model...')
    data_loader.reset()

    # Load model parameters
    net.load_state_dict(model.state_dict())
    net = net.to(device)
    net.eval()

    # Initialize evaluation metrics
    correct_count, exer_count = 0, 0
    batch_count, batch_avg_loss = 0, 0.0
    pred_all, label_all = [], []

    # Validation loop
    while not data_loader.is_end():
        batch_count += 1

        # Load batch data and move to device
        input_stu_ids, input_exer_ids, input_knowledge_embs, labels = data_loader.next_batch()
        input_stu_ids, input_exer_ids, input_knowledge_embs, labels = (
            input_stu_ids.to(device), input_exer_ids.to(device),
            input_knowledge_embs.to(device), labels.to(device)
        )

        # Forward pass
        output = net.forward(input_stu_ids, input_exer_ids, input_knowledge_embs)
        output = output.view(-1)

        # Compute accuracy
        for i in range(len(labels)):
            if (labels[i] == 1 and output[i] > 0.5) or (labels[i] == 0 and output[i] < 0.5):
                correct_count += 1
        exer_count += len(labels)

        # Store predictions and labels
        pred_all += output.to(torch.device('cpu')).tolist()
        label_all += labels.to(torch.device('cpu')).tolist()

    # Convert to numpy arrays
    pred_all = np.array(pred_all)
    label_all = np.array(label_all)

    # Compute metrics
    accuracy = correct_count / exer_count
    rmse = np.sqrt(np.mean((label_all - pred_all) ** 2))
    auc = roc_auc_score(label_all, pred_all)

    # Print and save results
    print(f'epoch= {epoch+1}, accuracy= {accuracy:.6f}, rmse= {rmse:.6f}, auc= {auc:.6f}')
    with open('result/model_val.txt', 'a', encoding='utf8') as f:
        f.write(f'epoch= {epoch+1}, accuracy= {accuracy:.6f}, rmse= {rmse:.6f}, auc= {auc:.6f}\n')

    return rmse, auc


def save_snapshot(model, filename):
    with open(filename, 'wb') as f:
        torch.save(model.state_dict(), f)


if __name__ == '__main__':
    train()
