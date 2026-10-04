"""Small, document-preserving samples from a fixed public dataset catalog.

No arbitrary URLs, remote dataset scripts, private credentials, or split mixing.
ModelScope streams bounded JSONL prefixes. Local samples and provenance are hashed;
mutable upstream branches are never described as revision-pinned snapshots.
"""
import hashlib
import json
import time
import urllib.error
import urllib.parse
import urllib.request
from pathlib import Path

CATALOG = [
    dict(key='ms-mini-pretrain',name='中文预训练样本',language='中文',provider='ModelScope',dataset='BazingaLyn/mini_pretrain_dataset',file='pretrain_hq_v7.jsonl',revision='master',license='Apache-2.0（数据卡标注）',url='https://modelscope.cn/datasets/BazingaLyn/mini_pretrain_dataset',description='整理为 text 字段的中文预训练语料；按完整行读取小样本，不下载整个 4 GB 文件。',network=True),
    dict(key='ms-minimind',name='MiniMind 轻量预训练',language='中文 / 英文',provider='ModelScope',dataset='gongjy/minimind_dataset',file='pretrain_t2t_mini.jsonl',revision='master',license='CC-BY-NC-4.0（非商业，数据卡标注）',url='https://modelscope.cn/datasets/gongjy/minimind_dataset',description='中英混合、已整理为 text 的训练语料；仅供符合数据许可的学习与实验。',network=True),
    dict(key='sample',name='合成流程示例',language='中文',provider='本地',dataset='trainlab/sample-corpus',config='local',license='项目自带合成示例',url='/sample-corpus.jsonl',description='完全离线，用于检查上传、分词、训练和恢复流程，不用于模型能力评估。',network=False),
]

MAX_RESPONSE=8*1024*1024
MAX_OUTPUT=20*1024*1024

class CatalogError(ValueError):pass
class NoRedirect(urllib.request.HTTPRedirectHandler):
    def redirect_request(self,req,fp,code,msg,headers,newurl):
        raise CatalogError('数据源发生重定向，已停止导入；请从官方来源下载后上传')

class ModelScopeRedirect(urllib.request.HTTPRedirectHandler):
    def redirect_request(self,req,fp,code,msg,headers,newurl):
        url=urllib.parse.urlparse(newurl)
        if url.scheme!='https' or url.hostname not in {'modelscope.cn','www.modelscope.cn','cdn-lfs-cn-1.modelscope.cn'} or url.username or url.password or url.port not in (None,443):
            raise CatalogError('ModelScope 下载跳转到了未允许的域名，请从数据卡下载后上传')
        return super().redirect_request(req,fp,code,msg,headers,newurl)

def fetch_modelscope(source,count,deadline):
    query=urllib.parse.urlencode({'Revision':source['revision'],'FilePath':source['file']})
    url='https://modelscope.cn/api/v1/datasets/'+source['dataset']+'/repo?'+query
    req=urllib.request.Request(url,headers={'User-Agent':'TrainLab/5 dataset-import','Accept':'application/octet-stream','Range':f'bytes=0-{MAX_OUTPUT-1}','Accept-Encoding':'identity'})
    rows=[];scanned=0;received=0;prefix=hashlib.sha256();etag=None
    try:
        with urllib.request.build_opener(ModelScopeRedirect).open(req,timeout=min(10,max(.1,deadline-time.monotonic()))) as response:
            if response.status==206 and not response.headers.get('Content-Range','').startswith('bytes 0-'):
                raise CatalogError('数据源返回的字节范围不一致')
            if response.headers.get('Content-Encoding','identity')!='identity':raise CatalogError('数据源返回了不支持的压缩流')
            etag=response.headers.get('ETag')
            # Stop on whole JSONL documents, bounded even if the host ignores Range.
            while scanned<5000 and len(rows)<min(600,count+100):
                if time.monotonic()>deadline:raise CatalogError('导入超时，请选择较小样本后重试')
                raw=response.readline(min(1024*1024+1,MAX_OUTPUT-received+1))
                if not raw:break
                received+=len(raw);prefix.update(raw);scanned+=1
                if received>MAX_OUTPUT:break
                if len(raw)>1024*1024:raise CatalogError('数据源单篇文档超过 1 MiB，请整理后上传')
                if not raw.endswith(b'\n'):break # Never accept a partial Range response as a document.
                try:
                    row=json.loads(raw)
                    if not isinstance(row,dict):raise ValueError()
                except (ValueError,UnicodeError):raise CatalogError('ModelScope 文件不是有效的 JSONL 文本数据')
                rows.append({'row_idx':scanned-1,'row':row})
        return {'rows':rows,'num_rows_total':len(rows),'downloaded_bytes':received,'prefix_sha256':prefix.hexdigest(),'etag':etag}
    except CatalogError:raise
    except (OSError,ValueError) as exc:
        raise CatalogError('无法读取 ModelScope 数据源，请检查 modelscope.cn 与 cdn-lfs-cn-1.modelscope.cn 的 HTTPS 连通性，或下载 JSONL 后上传') from exc

def collect(source,count,root):
    deadline=time.monotonic()+60
    texts=[];origins=[];seen=set();size=0;skipped=0;responses=[];offset=0
    local=source['key']=='sample'
    for page in range(1):
        if time.monotonic()>deadline:raise CatalogError('导入超时，请选择较小样本后重试')
        if local:
            rows=[{'row_idx':i,'row':json.loads(line)} for i,line in enumerate((Path(root)/'examples/sample-corpus.jsonl').read_text().splitlines()) if line.strip()]
            result={'rows':rows,'num_rows_total':len(rows)}
        else:result=fetch_modelscope(source,count,deadline)
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
    manifest={'schema_version':1,'source':source,'split':'local' if local else 'pretraining_file','requested_documents':count,'documents':len(texts),'skipped':skipped,'selection':'从起始位置顺序取完整文档；非随机、非完整数据集','upstream_revision':'local' if local else source['revision']+'（可变分支，使用本地样本摘要复现）','upstream_file':source.get('file'),'upstream_etag':result.get('etag'),'downloaded_prefix_sha256':result.get('prefix_sha256'),'response_sha256':responses,'origins':origins,'evaluation':'平台按完整文档重新划分训练/验证集；不是官方测试集成绩'}
    return b''.join(texts),manifest
