"""Small, document-preserving samples from a fixed public dataset catalog.

No arbitrary URLs, remote dataset scripts, private credentials, or split mixing.
The Viewer serves mutable snapshots: the manifest records hashes and row origins,
not a claim of revision-pinned upstream reproducibility.
"""
import hashlib
import json
import time
import urllib.error
import urllib.parse
import urllib.request
from pathlib import Path

CATALOG = [
    dict(key='sample',name='合成流程示例',language='中文',dataset='trainlab/sample-corpus',config='local',license='项目自带合成示例',url='/sample-corpus.jsonl',description='离线可用，用于检查上传、分词、训练和恢复流程，不用于模型能力评估。',network=False),
    dict(key='tinystories',name='TinyStories',language='英文',dataset='roneneldan/TinyStories',config='default',license='CDLA-Sharing-1.0',url='https://huggingface.co/datasets/roneneldan/TinyStories',description='词汇较简单的合成短故事；适合小语言模型从零预训练入门。',network=True),
    dict(key='wikipedia-zh',name='Wikipedia 中文',language='中文',dataset='wikimedia/wikipedia',config='20231101.zh',license='CC BY-SA 3.0 / GFDL（数据卡标注）',url='https://huggingface.co/datasets/wikimedia/wikipedia',description='2023-11-01 中文百科快照，整篇文章作为文档，适合中文分词与预训练实验。',network=True),
    dict(key='wikipedia-en',name='Wikipedia 英文',language='英文',dataset='wikimedia/wikipedia',config='20231101.en',license='CC BY-SA 3.0 / GFDL（数据卡标注）',url='https://huggingface.co/datasets/wikimedia/wikipedia',description='2023-11-01 英文百科快照；篇幅较长，可用于观察上下文长度与训练成本。',network=True),
]
MAX_RESPONSE=8*1024*1024
MAX_OUTPUT=20*1024*1024

class CatalogError(ValueError):pass
class NoRedirect(urllib.request.HTTPRedirectHandler):
    def redirect_request(self,req,fp,code,msg,headers,newurl):
        raise CatalogError('数据源发生重定向，已停止导入；请从官方来源下载后上传')

def fetch_rows(source,offset,length,deadline):
    remaining=deadline-time.monotonic()
    if remaining<=0:raise CatalogError('导入超时，请选择较小样本后重试')
    query=urllib.parse.urlencode(dict(dataset=source['dataset'],config=source['config'],split='train',offset=offset,length=length))
    req=urllib.request.Request('https://datasets-server.huggingface.co/rows?'+query,headers={'User-Agent':'TrainLab/4 dataset-import','Accept':'application/json'})
    try:
        with urllib.request.build_opener(NoRedirect).open(req,timeout=min(10,remaining)) as response:
            body=bytearray()
            while True:
                if time.monotonic()>deadline:raise CatalogError('导入超时，请选择较小样本后重试')
                chunk=response.read(65536)
                if not chunk:break
                body.extend(chunk)
                if len(body)>MAX_RESPONSE:raise CatalogError('数据源单页过大，请下载后按文档整理并上传')
        result=json.loads(body)
        if not isinstance(result,dict) or not isinstance(result.get('rows'),list):raise ValueError()
        return result
    except CatalogError:raise
    except (OSError,ValueError) as exc:
        raise CatalogError('无法读取 Hugging Face 数据源，请检查服务器外网连接后重试，或下载 JSONL 后上传') from exc

def collect(source,count,root):
    deadline=time.monotonic()+60
    texts=[];origins=[];seen=set();size=0;skipped=0;responses=[];offset=0
    local=source['key']=='sample'
    for page in range(12):
        if time.monotonic()>deadline:raise CatalogError('导入超时，请选择较小样本后重试')
        if local:
            rows=[{'row_idx':i,'row':json.loads(line)} for i,line in enumerate((Path(root)/'examples/sample-corpus.jsonl').read_text().splitlines()) if line.strip()]
            result={'rows':rows,'num_rows_total':len(rows)}
        else:result=fetch_rows(source,offset,min(50,count-len(texts)),deadline)
        rows=result['rows']
        if not all(isinstance(x,dict) and isinstance(x.get('row'),dict) for x in rows):raise CatalogError('数据源格式变化，请检查官方数据卡')
        if not rows:break
        responses.append(hashlib.sha256(json.dumps(result,ensure_ascii=False,sort_keys=True).encode()).hexdigest())
        for entry in rows:
            row=entry.get('row',{});text=row.get('text')
            if entry.get('truncated_cells') or not isinstance(text,str) or not text.strip():skipped+=1;continue
            text=text.strip();digest=hashlib.sha256(text.encode()).hexdigest()
            line=(json.dumps({'text':text},ensure_ascii=False)+'\n').encode()
            if len(line)>1024*1024 or digest in seen:skipped+=1;continue
            if size+len(line)>MAX_OUTPUT:raise CatalogError('样本超过 20 MiB，请选择更少文档')
            seen.add(digest);texts.append(line);size+=len(line)
            origin={'row_index':entry.get('row_idx'),'text_sha256':digest}
            for field in ('id','title','url'):
                if isinstance(row.get(field),str):origin[field]=row[field][:2048]
            origins.append(origin)
            if len(texts)==count:break
        offset+=len(rows)
        if local or len(texts)==count or offset>=result.get('num_rows_total',offset):break
    if len(texts)<20:raise CatalogError('可用完整文档不足 20 篇，无法用于当前训练流程')
    manifest={'schema_version':1,'source':source,'split':'local' if local else 'train','requested_documents':count,'documents':len(texts),'skipped':skipped,'selection':'从起始位置顺序取完整文档；非随机、非完整数据集','upstream_revision':'local' if local else 'Dataset Viewer 当前快照；不保证固定上游版本','response_sha256':responses,'origins':origins,'evaluation':'平台按完整文档重新划分训练/验证集；不是官方测试集成绩'}
    return b''.join(texts),manifest
