# Zero-1-to-3: Domain-level Zero-shot Cognitive Diagnosis via One Batch of Early-bird Students towards Three Diagnostic Objectives

This repository contains the implementation for the paper titled **Zero-1-to-3: Domain-level Zero-shot Cognitive Diagnosis via One Batch of Early-bird Students towards Three Diagnostic Objectives**, published at `AAAI 2024` [[Paper](https://arxiv.org/abs/2312.13434)]. 

Authors: [Weibo Gao](https://scholar.google.com/citations?user=k19RS74AAAAJ&hl=zh-CN), [Qi Liu](http://staff.ustc.edu.cn/~qiliuql), [Hao Wang](http://staff.ustc.edu.cn/~wanghao3), et al.

Email: weibogao@mail.ustc.edu.cn

## News
- 2025-03-10：We have updated multiple manually annotated sample datasets on different learning topics for reference. If you use other datasets, please partition them into multiple domains based on knowledge topics, difficulty levels, or randomly, and process them into the provided data-demo format. Due to the first author's current time constraints with upcoming deadlines, a more detailed README is temporarily unavailable. If you encounter any difficulties while running the code, feel free to reach out via WeChat. The first author (gaoweibo1997) will provide verbal guidance on running the code or processing the data. Thank you for your interest and understanding! （In Chinese：我们更新了多个由人工标注的学习主题数据样例，供参考。若使用其他数据，请根据数据的知识主题、难度等因素，或随机划分成多个域，并处理成提供的 data-demo 格式。由于第一作者近期正在处理紧迫的DDL，暂时无法提供更详细的 README。如在运行代码时遇到困难，欢迎通过微信交流，第一作者（gaoweibo1997）将通过微信口述指导代码运行或数据处理。感谢您的关注与理解！）

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
