"""torchrun --standalone --nproc_per_node=8 scripts/ddp_check.py"""
import os
import torch
import torch.distributed as dist
rank=int(os.environ['LOCAL_RANK'])
torch.cuda.set_device(rank)
dist.init_process_group('nccl')
x=torch.tensor(float(dist.get_rank()+1),device=f'cuda:{rank}')
dist.all_reduce(x)
expected=dist.get_world_size()*(dist.get_world_size()+1)/2
assert x.item()==expected,(x.item(),expected)
if dist.get_rank()==0:
    print(f'NCCL all_reduce passed on {dist.get_world_size()} GPUs',flush=True)
dist.destroy_process_group()
