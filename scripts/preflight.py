#!/usr/bin/env python3
"""Run INSIDE GPU container before starting a large training job."""
import json
import subprocess
import sys
import torch
r={'torch':torch.__version__,'cuda_runtime':torch.version.cuda,'gpu_count':torch.cuda.device_count(),'cards':[]}
if not torch.cuda.is_available():
    raise SystemExit('CUDA 不可用。检查 NVIDIA 驱动和 Container Toolkit。')
for i in range(torch.cuda.device_count()):
    torch.cuda.set_device(i)
    props=torch.cuda.get_device_properties(i)
    # Runs actual kernels, rather than only enumerating hardware.
    x=torch.randn(256,256,device=f'cuda:{i}',dtype=torch.bfloat16,requires_grad=True)
    y=(x@x).float().square().mean()
    y.backward()
    torch.cuda.synchronize()
    if not torch.isfinite(y):
        raise SystemExit(f'GPU {i} kernel test failed')
    r['cards'].append({'index':i,'name':props.name,'memory_gib':round(props.total_memory/1024**3,1),'compute_capability':list(torch.cuda.get_device_capability(i)),'bf16_matmul_backward':'passed'})
print(json.dumps(r,ensure_ascii=False,indent=2))
