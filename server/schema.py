from typing import Literal
from pydantic import BaseModel, ConfigDict, Field, model_validator

class TrainConfig(BaseModel):
    model_config = ConfigDict(extra='forbid')
    name: str = Field(default='GPT 预训练实验', min_length=1, max_length=80)
    dataset_id: str = Field(pattern=r'^[a-f0-9]{32}$')
    gpu_ids: list[int] = Field(default_factory=lambda: [0], min_length=1, max_length=8)
    layers: int = Field(default=6, ge=2, le=32)
    hidden_size: int = Field(default=384, ge=128, le=2048)
    heads: int = Field(default=6, ge=2, le=32)
    seq_length: int = Field(default=256, ge=64, le=2048)
    vocab_size: int = Field(default=4096, ge=512, le=65536)
    micro_batch: int = Field(default=2, ge=1, le=16)
    grad_accum: int = Field(default=8, ge=1, le=256)
    max_steps: int = Field(default=100, ge=10, le=1000000)
    learning_rate: float = Field(default=0.0003, gt=0, le=0.01)
    warmup_ratio: float = Field(default=0.03, ge=0, le=0.5)
    weight_decay: float = Field(default=0.1, ge=0, le=1)
    eval_steps: int = Field(default=25, ge=1, le=10000)
    save_steps: int = Field(default=25, ge=1, le=10000)
    seed: int = Field(default=42, ge=0, le=2147483647)
    precision: Literal['bf16', 'fp32'] = 'bf16'
    gradient_checkpointing: bool = True
    max_runtime_seconds: int = Field(default=21600, ge=60, le=604800)

    @model_validator(mode='after')
    def consistent(self):
        if self.hidden_size % self.heads:
            raise ValueError('hidden_size 必须能被 heads 整除')
        if len(set(self.gpu_ids)) != len(self.gpu_ids) or any(x < 0 or x > 63 for x in self.gpu_ids):
            raise ValueError('GPU 编号必须唯一，且介于 0 和 63')
        if self.eval_steps > self.max_steps or self.save_steps > self.max_steps:
            raise ValueError('评估和保存间隔不能大于总步数')
        return self

    def estimate(self):
        h, l = self.hidden_size, self.layers
        # GPT-2 tied embedding, learned positional embeddings, blocks and final LN.
        params = self.vocab_size*h + self.seq_length*h + l*(12*h*h+13*h)+2*h
        effective = self.micro_batch*self.grad_accum*len(self.gpu_ids)
        return {'estimated_parameters': params, 'effective_batch': effective,
                'tokens_per_update': effective*self.seq_length,
                'scheduled_tokens': effective*self.seq_length*self.max_steps,
                'note': '参数量按请求词表上限估计；分词器实际词表可能较小。DDP 每卡保存完整模型；不是显存容量保证。'}


class CatalogImport(BaseModel):
    model_config = ConfigDict(extra='forbid')
    documents: Literal[50, 200, 500] = 50
