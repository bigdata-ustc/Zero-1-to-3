# Zero-1-to-3: Domain-level Zero-shot Cognitive Diagnosis via One Batch of Early-bird Students towards Three Diagnostic Objectives

This repository contains the implementation for the paper titled **Zero-1-to-3: Domain-level Zero-shot Cognitive Diagnosis via One Batch of Early-bird Students towards Three Diagnostic Objectives**, published at `AAAI 2024` [[Paper](https://arxiv.org/abs/2312.13434)]. 

Authors: [Weibo Gao](https://scholar.google.com/citations?user=k19RS74AAAAJ&hl=zh-CN), [Qi Liu](http://staff.ustc.edu.cn/~qiliuql), [Hao Wang](http://staff.ustc.edu.cn/~wanghao3), et al.

Email: weibogao@mail.ustc.edu.cn

## Environment Settings
We use PyTorch as the backend.
- Torch version: '1.7.1'

## Datasets
```
[1] Dbe-kt22: A knowledge tracing dataset based on online student evaluation. arXiv'22
[2] XES3G5M: a knowledge tracing benchmark dataset with auxiliary information. NeurIPS'23
```

## Running

1. Select a model for running, e.g., Zero-NCDM
   ```
    CD Zero-NCDM
   ```
2. Pre-training and testing the model, in multiple source domains
   ```
   python train.py
   ```
3. Fine-tuning the model using early-bird students' logs in the target domain:
   ```
   python fine_tune.py
   ```
4. Generating the simulated logs
   ```
   python generate_coll_data.py
   ```
5. Fine-tuning using the simulated data
   ```
   python fine_tune_step_2.py
   ```

## Related Works
- **RCD: Relation Map Driven Cognitive Diagnosis for Intelligent Education Systems (SIGIR'2021)** [[Paper](https://dl.acm.org/doi/abs/10.1145/3404835.3462932)] [[Code](https://github.com/bigdata-ustc/RCD/)] [[Presentation Video](https://dl.acm.org/action/downloadSupplement?doi=10.1145%2F3404835.3462932&file=RCD.mp4)]

- **Leveraging Transferable Knowledge Concept Graph Embedding for Cold-Start Cognitive Diagnosis (SIGIR'2023)** [[Paper](https://dl.acm.org/doi/10.1145/3539618.3591774)] [[Code](https://github.com/bigdata-ustc/TechCD)] [[Presentation Video](https://dl.acm.org/action/downloadSupplement?doi=10.1145%2F3539618.3591774&file=SIGIR23-fp1870.mp4)]

- **FedJudge: Federated Legal Large Language Model** [[Paper](https://arxiv.org/abs/2309.08173)] [[Code](https://github.com/yuelinan/FedJudge)]

### Last Update Date: March 14, 2024
