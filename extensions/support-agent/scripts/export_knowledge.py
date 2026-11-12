"""Export the existing synthetic business corpus as traceable UTF-8 Markdown.

The JSON remains the frozen evaluation input. Markdown bodies retain each
original paragraph verbatim; headings and provenance add no business rules.
"""
import argparse
import hashlib
import json
from pathlib import Path

SERVICE = Path(__file__).resolve().parents[1]


def digest(data):
    return hashlib.sha256(data).hexdigest()


def save(path, data):
    if path.exists():
        if path.read_bytes() != data:
            raise ValueError(f'Refusing to overwrite different content: {path.name}; choose a new output directory.')
        return
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_bytes(data)


def export(source, destination):
    original = source.read_bytes()
    records = json.loads(original.decode('utf-8-sig'))
    assert isinstance(records, list) and records
    assert len({row['id'] for row in records}) == len(records)
    assert len({row['org_id'] for row in records}) == 1, 'Export each organization separately; never combine scopes.'
    title = '星桥云办公客服业务知识手册（模拟资料）'
    combined = f'# {title}\n\n本手册逐条转换自 data/knowledge.json，仅增加标题和来源标记，不补充或改写业务规则。\n\n资料描述的是模拟产品，不代表本机知识库导入工具支持文中所有文件类型或上传限额。\n\n'
    articles = []
    for row in records:
        identifier, content = row['id'], row['content']
        assert identifier.startswith('KB-') and identifier[3:].isdigit()
        assert all(isinstance(row[field], str) for field in ('title', 'source', 'content', 'org_id'))
        heading = f'## {identifier} {row["title"]}\n\n来源：{row["source"]}\n\n'
        start = len(combined) + len(heading)
        combined += heading + content + '\n\n'
        article = f'# {row["title"]}\n\n原始编号：{identifier}\n\n来源：{row["source"]}\n\n{content}\n'
        filename = 'articles/' + identifier + '.md'
        save(destination / filename, article.encode('utf-8'))
        assert combined[start:start + len(content)] == content
        articles.append({'id':identifier, 'title':row['title'], 'source':row['source'], 'filename':filename,
                         'content_sha256':digest(content.encode('utf-8')), 'file_sha256':digest(article.encode('utf-8')),
                         'handbook_content_start':start, 'handbook_content_end':start + len(content)})
    save(destination / 'knowledge.md', combined.encode('utf-8'))
    manifest = {'source':'data/knowledge.json', 'source_sha256':digest(original), 'org_id':records[0]['org_id'],
                'title':title, 'synthetic':True, 'business_paragraphs_verbatim':True,
                'document_count':len(records), 'handbook_file':'knowledge.md',
                'handbook_sha256':digest(combined.encode('utf-8')), 'articles':articles,
                'note':'The combined handbook and articles contain the same source facts. Upload one representation for a chosen corpus; converting format is not new independent knowledge.'}
    save(destination / 'manifest.json', (json.dumps(manifest, ensure_ascii=False, indent=2) + '\n').encode('utf-8'))
    assert source.read_bytes() == original
    return manifest


if __name__ == '__main__':
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--source', type=Path, default=SERVICE / 'data/knowledge.json')
    parser.add_argument('--output', type=Path, default=SERVICE / 'data/documents')
    args = parser.parse_args()
    manifest = export(args.source, args.output)
    print(json.dumps({'output':str(args.output), 'articles':manifest['document_count'],
                      'handbook_sha256':manifest['handbook_sha256'], 'original_json_unchanged':True}, ensure_ascii=False))
