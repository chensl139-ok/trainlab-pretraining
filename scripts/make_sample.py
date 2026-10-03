"""Synthetic plumbing corpus. Not suitable for evaluating language quality."""
import json
from pathlib import Path
path=Path(__file__).resolve().parent.parent/'examples'/'sample-corpus.jsonl'
with path.open('w') as f:
    for i in range(240):
        topic=['数据清洗','模型评估','梯度下降','实验复现','学习计划','科学记录'][i%6]
        text=f'实验记录第 {i} 篇，主题是{topic}。'+''.join(f'第 {j} 项观察：学习模型训练需要提出假设、记录输入、比较结果，样本编号为 {i*37+j}。Training a model requires data, an objective, and careful evaluation. We record experiment {i} and observation {j}. ' for j in range(12))
        f.write(json.dumps({'text':text},ensure_ascii=False)+'\n')
print(path)
