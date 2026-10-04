"""Bounded full-parameter SFT/DPO on immutable platform model artifacts."""
import hashlib
import json
from pathlib import Path

FORMATS = {'pretrain': {'text'}, 'sft': {'prompt', 'response'}, 'dpo': {'prompt', 'chosen', 'rejected'}}
MODEL_FIELDS = ('architecture', 'layers', 'hidden_size', 'heads', 'kv_heads', 'intermediate_size', 'seq_length', 'vocab_size', 'parameter_count')


def inherited_config(config):
    from server.schema import TrainConfig
    return {key:config.get(key,TrainConfig.model_fields[key].default) for key in MODEL_FIELDS}


def record_format(item):
    if not isinstance(item, dict):
        raise ValueError('每行必须是 JSON 对象')
    kind = next((k for k, fields in FORMATS.items() if set(item) == fields), None)
    if not kind or any(not isinstance(v, str) or not v.strip() for v in item.values()):
        raise ValueError('数据需为非空字符串：预训练 text；SFT prompt/response；DPO prompt/chosen/rejected')
    if kind == 'dpo' and item['chosen'].strip() == item['rejected'].strip():
        raise ValueError('DPO 的 chosen 和 rejected 不能相同')
    return kind


def prompt_text(prompt, tokenizer=None):
    if tokenizer is not None and tokenizer.chat_template:
        return tokenizer.apply_chat_template([{'role':'user','content':prompt.strip()}],tokenize=False,add_generation_prompt=True,enable_thinking=False)
    return '### User:\n' + prompt.strip() + '\n\n### Assistant:\n'


def model_fingerprint(path):
    path = Path(path)
    files = sorted(p for p in path.rglob('*') if p.is_file() and (p.suffix in ('.json', '.safetensors', '.jinja')))
    if path.is_symlink() or any(p.is_symlink() for p in path.rglob('*')):
        raise ValueError('基础模型目录不能包含符号链接')
    h = hashlib.sha256()
    for file in files:
        h.update(str(file.relative_to(path)).encode() + b'\0')
        with file.open('rb') as f:
            for chunk in iter(lambda: f.read(1024 * 1024), b''):
                h.update(chunk)
    return h.hexdigest()


def encode_pair(tokenizer, prompt, answer, length):
    prefix = tokenizer.encode(prompt_text(prompt, tokenizer), add_special_tokens=False)
    completion = tokenizer.encode(answer.strip(), add_special_tokens=False) + [tokenizer.eos_token_id]
    ids = prefix + completion
    if not prefix or len(ids) > length:
        raise ValueError(f'格式化后的样本有 {len(ids)} tokens，超过基础模型上下文 {length}；请缩短完整样本，平台不会截断回答')
    return {'input_ids': ids, 'labels': [-100] * len(prefix) + completion}


def prepare_post(config, output):
    from transformers import PreTrainedTokenizerFast
    from server.evaluate import digest
    path = Path(config['dataset_path'])
    if digest(path) != config['dataset_sha256']:
        raise ValueError('数据文件摘要与提交时不一致')
    base = Path(config['base_model_path'])
    if model_fingerprint(base) != config['base_model_sha256']:
        raise ValueError('基础模型文件在提交后发生改变，拒绝后训练')
    tokenizer = PreTrainedTokenizerFast.from_pretrained(base, local_files_only=True)
    if tokenizer.eos_token_id is None or tokenizer.pad_token_id is None:
        raise ValueError('基础模型缺少 EOS/PAD token')
    tokenizer.save_pretrained(output / 'tokenizer')
    prepared = output / 'prepared'
    prepared.mkdir(parents=True, exist_ok=True)
    counts = {'train': 0, 'validation': 0}
    tokens = {'train': 0, 'validation': 0}
    seen = set()
    files = {s: (prepared / (s + '.jsonl')).open('w') for s in counts}
    try:
        with path.open(encoding='utf-8') as src:
            for number, line in enumerate(src, 1):
                if not line.strip():
                    continue
                item = json.loads(line)
                if record_format(item) != config['stage']:
                    raise ValueError('数据格式与训练阶段不匹配')
                canonical = json.dumps({k: v.strip() for k, v in item.items()}, sort_keys=True, ensure_ascii=False)
                key = hashlib.sha256(canonical.encode()).digest()
                if key in seen:
                    continue
                seen.add(key)
                # All answers to the same normalized prompt stay in the same split.
                bucket = int(hashlib.sha256((str(config['seed']) + item['prompt'].strip()).encode()).hexdigest()[:8], 16) % 100
                split = 'validation' if bucket < 10 else 'train'
                try:
                    if config['stage'] == 'sft':
                        row = encode_pair(tokenizer, item['prompt'], item['response'], config['seq_length'])
                        count = sum(x != -100 for x in row['labels'])
                    else:
                        row = {k: encode_pair(tokenizer, item['prompt'], item[k], config['seq_length']) for k in ('chosen', 'rejected')}
                        if row['chosen']['input_ids'] == row['rejected']['input_ids']:
                            raise ValueError('两种回答分词后相同')
                        count = sum(x != -100 for pair in row.values() for x in pair['labels'])
                except ValueError as exc:
                    raise ValueError(f'第 {number} 行：{exc}') from exc
                files[split].write(json.dumps(row) + '\n')
                counts[split] += 1
                tokens[split] += count
    finally:
        for file in files.values():
            file.close()
    if min(counts.values()) < 2 or counts['train'] < max(1, len(config['gpu_ids'])) * config['micro_batch']:
        raise ValueError('按提示词分组后训练/验证各至少 2 个样本，训练样本需覆盖每卡 batch；请增加不同提示词')
    meta = {'stage': config['stage'], 'examples': counts, 'response_tokens': tokens, 'vocab_size': len(tokenizer),
            'dataset_sha256': config['dataset_sha256'], 'base_model_sha256': config['base_model_sha256'],
            'split': 'seeded SHA256 by normalized prompt, approx 90/10; exact dedup',
            'format': 'trainlab prompt-response v1', 'loss_mask': 'completion and EOS only; no truncation or packing'}
    (output / 'data_manifest.json').write_text(json.dumps(meta, ensure_ascii=False, indent=2))
    return meta


class PostDataset:
    def __init__(self, path):
        self.path = Path(path)
        self.offsets = []
        with self.path.open('rb') as f:
            while True:
                offset = f.tell()
                if not f.readline():
                    break
                self.offsets.append(offset)

    def __len__(self):
        return len(self.offsets)

    def __getitem__(self, index):
        with self.path.open('rb') as f:
            f.seek(self.offsets[index])
            return json.loads(f.readline())


def collator(pad_id, paired=False):
    import torch
    def pad(rows):
        n = max(len(r['input_ids']) for r in rows)
        return {k: torch.tensor([r[k] + [fill] * (n - len(r[k])) for r in rows])
                for k, fill in [('input_ids', pad_id), ('labels', -100)]} | {
                    'attention_mask': torch.tensor([[1] * len(r['input_ids']) + [0] * (n - len(r['input_ids'])) for r in rows])}
    def batch(rows):
        if not paired:
            return pad(rows)
        # A single forward pass for chosen + rejected; pair ordering stays stable.
        return pad([r['chosen'] for r in rows] + [r['rejected'] for r in rows])
    return batch


def response_logps(logits, labels):
    import torch
    shifted = labels[:, 1:]
    mask = shifted != -100
    selected = torch.log_softmax(logits[:, :-1].float(), dim=-1).gather(-1, shifted.masked_fill(~mask, 0).unsqueeze(-1)).squeeze(-1)
    return (selected * mask).sum(-1)


def preference_loss(policy, reference, beta):
    import torch.nn.functional as F
    chosen, rejected = policy.chunk(2)
    ref_chosen, ref_rejected = reference.chunk(2)
    return -F.logsigmoid(beta * ((chosen - rejected) - (ref_chosen - ref_rejected)))


def dpo_trainer_class():
    import torch
    from transformers import Trainer
    class DPOTrainer(Trainer):
        def __init__(self, *args, reference, beta, **kwargs):
            super().__init__(*args, **kwargs)
            self.reference = reference.to(self.args.device).eval().requires_grad_(False)
            self.beta = beta
            # The custom loss is a pair mean; Trainer must scale gradient accumulation.
            self.model_accepts_loss_kwargs = False

        def compute_loss(self, model, inputs, return_outputs=False, num_items_in_batch=None):
            inputs = dict(inputs)
            labels = inputs.pop('labels')
            outputs = model(**inputs, use_cache=False)
            policy = response_logps(outputs.logits, labels)
            with torch.no_grad():
                ref = response_logps(self.reference(**inputs, use_cache=False).logits, labels)
            loss = preference_loss(policy, ref, self.beta).mean()
            return (loss, outputs) if return_outputs else loss

        def prediction_step(self, model, inputs, prediction_loss_only, ignore_keys=None):
            with torch.no_grad(), self.compute_loss_context_manager():
                loss = self.compute_loss(model, self._prepare_inputs(inputs))
            return loss.detach(), None, None
    return DPOTrainer
