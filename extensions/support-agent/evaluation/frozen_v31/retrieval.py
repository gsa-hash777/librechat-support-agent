"""Small-corpus lexical retrieval with explicit evidence limits, not an answer judge.

No model/download/remote request is used. Scores rank text; they are never a
probability that the text answers the question. Conservative query guards only
cover named fact types, so the answering model must still check each claim.
"""
from collections import Counter
import math
import re
import unicodedata

from .content import suspicious_reasons, tool_view
from .search import tokenize

RETRIEVAL_VERSION = 'knowledge-v3.1'
MIN_COVERAGE = 0.12
STOP_WORDS = ('请问', '请帮', '帮我', '一下', '怎么', '如何', '什么', '哪些', '是否', '能否',
              '可以', '需要', '应该', '告诉', '知道', '查询', '查找', '检索', '手册', '文档',
              '知识', '资料', '问题', '这个', '那个', '当前', '星桥', '为什么', '怎么办',
              '现在', '我想', '说明', '确认', '发生', '提供', '相关', '信息')
STOP_TOKENS = {term for word in STOP_WORDS for term in tokenize(word)}
ENGLISH_STOP = {'the', 'and', 'how', 'what', 'why', 'with', 'for', 'please', 'can', 'you'}

# These are generic information needs, not answer keys, document IDs or test cases.
# Missing topic words prevent nearby billing/login docs from proving a precise claim.
FACT_TYPES = (
    ('pricing', r'报价|多少钱|单价|年费|价格|收费标准', r'报价|多少钱|单价|年费|价格|收费标准|[¥￥]|\d\s*元'),
    ('tax', r'含税|不含税|税率', r'含税|不含税|税率'),
    ('certificate', r'认证.{0,8}(?:编号|到期|有效期)|证书编号', r'认证证书|证书编号|certificate\s*(?:id|number)'),
    ('contact', r'热线|联系电话|联系邮箱|客服电话', r'热线|联系电话|联系邮箱|客服电话|[\w.+-]+@[\w.-]+\.[a-z]+'),
    ('roadmap', r'路线图|下个版本|未来版本|预计上线|计划上线', r'路线图|下个版本|未来版本|预计上线|计划上线'),
    ('private_deployment', r'私有化|本地部署|离线部署', r'私有化|本地部署|离线部署'),
    ('availability', r'可用率|可用性百分比|赔偿标准|服务赔偿', r'可用率|可用性百分比|赔偿|\d+(?:\.\d+)?\s*%'),
)


def normalize(value):
    return unicodedata.normalize('NFKC', value).casefold()


def lexical_text(value):
    # Error identifiers occur as HTTP429 / HTTP 429 and UTF8 / UTF 8. Separate
    # letter-number boundaries consistently in both queries and source text;
    # retain the numeric part so e.g. 401 never becomes evidence for 403.
    return re.sub(r'(?<=[a-z])(?=\d)|(?<=\d)(?=[a-z])', ' ', normalize(value))


def terms(value):
    return set(tokenize(lexical_text(value))) - STOP_TOKENS - ENGLISH_STOP


def _clarification(query):
    plain = re.sub(r'[\s，。！？?!、：:；;]', '', normalize(query))
    phrases = '这个 那个 这些 那些 这样 那样 请 帮忙 帮我 麻烦 我 你 一下 的 了 个 怎么办 怎么 如何 什么 问题 故障 报错 错误 失败 异常 不能用 不可用 不好用 坏了 坏 处理 解决 修复 查查 查询 看看 看 出现 遇到 系统 软件 功能 有 是 很 不 能 用 呀 吗 吧 啊 呢 哦 啦'.split()
    # Longest phrases first avoids stripping “了” before “坏了”; polite filler
    # does not turn an otherwise unspecified problem into a concrete subject.
    generic = re.sub('|'.join(sorted(phrases, key=len, reverse=True)), '', plain)
    if not generic:
        return '请补充具体产品或功能、错误提示，以及发生时的操作；目前信息不足以选择对应文档。'
    return None


def _anchors(query):
    # Require literal evidence for structured identifiers, not ordinary English
    # verbs such as explain/retry. Uppercase acronyms and common protocol names
    # are useful conservative hints; this is not general entity recognition.
    protocols = {'http', 'https', 'api', 'json', 'csv', 'utf', 'iso', 'saml', 'sso',
                 'pdf', 'docx', 'sql', 'oauth', 'jwt'}
    raw = unicodedata.normalize('NFKC', query)
    result = []
    for word in re.findall(r'[A-Za-z][A-Za-z0-9_-]+', raw):
        if word.casefold() in ENGLISH_STOP:
            continue
        if word.isupper() or word.casefold() in protocols or re.search(r'\d|_', word):
            result.extend(lexical_text(word).split())
    # Include numbers bound to an ASCII identifier, plus standalone HTTP status
    # codes. Do not turn requested answer lengths (e.g. “3 个步骤”) into anchors.
    result.extend(re.findall(r'[a-z]\s*(\d+)', normalize(query)))
    result.extend(re.findall(r'(?<![a-z0-9])([145]\d\d)(?![a-z0-9])', normalize(query)))
    return list(dict.fromkeys(result))


def _rank(documents, query, limit):
    qt = terms(query)
    if not qt:
        return []
    index = []
    for document in documents:
        body = document.get('content', '')
        title = document.get('title', '')
        heading = document.get('heading', '') or ''
        weighted = Counter(tokenize(lexical_text(title + ' ' + title + ' ' + heading + ' ' + body)))
        index.append((document, weighted))
    if not index:
        return []
    avg = max(1, sum(sum(tokens.values()) for _, tokens in index) / len(index))
    df = Counter(token for _, tokens in index for token in tokens)
    scored = []
    for doc, tokens in index:
        covered = qt & tokens.keys()
        coverage = len(covered) / len(qt)
        if not covered or coverage < MIN_COVERAGE:
            continue
        size = sum(tokens.values())
        score = sum(math.log(1 + (len(index) - df[t] + .5) / (df[t] + .5)) *
                    tokens[t] * 2.5 / (tokens[t] + 1.5 * (.25 + .75 * size / avg)) for t in covered)
        if score > 0:
            scored.append({**doc, 'score': round(score, 6), 'query_coverage': round(coverage, 6)})
    scored.sort(key=lambda row: (-row['score'], str(row.get('chunk_id', row['id']))))
    # First give different documents a chance; then use remaining slots for chunks.
    primary, secondary, seen = [], [], set()
    for row in scored:
        key = row.get('doc_id', row['id'])
        (secondary if key in seen else primary).append(row)
        seen.add(key)
    return (primary + secondary)[:limit]


def retrieve(documents, query, limit=5, *, ranker=None):
    limit = max(1, min(10, int(limit)))
    question = _clarification(query)
    common = {'query': query, 'retrieval_version': RETRIEVAL_VERSION,
              'score_meaning': '排序分数不是答案正确率；相关文档不代表包含所问事实。',
              'clarification': question, 'reason_codes': [], 'results': []}
    if question:
        return {**common, 'evidence_status': 'clarify', 'reason_codes': ['missing_problem_context'],
                'guidance': '先追问具体业务问题；不要根据模糊请求猜测解决方案或创建工单。'}
    safe, excluded = [], 0
    for item in documents:
        if item.get('document_unsafe') or any(suspicious_reasons(str(item.get(field, ''))) for field in ('title', 'content', 'source', 'heading')):
            excluded += 1
            continue
        safe.append(item)
    # Trusted server-side ranker injection; the original evidence policy applies
    # identically to lexical, hybrid and reranked candidates.
    ranked = (ranker or _rank)(safe, query, limit)
    text = lexical_text('\n'.join(str(row.get(field, '')) for row in ranked for field in ('title', 'content')))
    missing = []
    if not ranked:
        missing.append('no_relevant_evidence')
    else:
        for anchor in _anchors(query):
            if not re.search(r'(?<![a-z0-9_])' + re.escape(anchor) + r'(?![a-z0-9_])', text):
                missing.append('missing_literal_anchor')
                break
        for name, trigger, proof in FACT_TYPES:
            if re.search(trigger, normalize(query)) and not re.search(proof, text):
                missing.append('missing_' + name + '_evidence')
        event = re.search(r'(恢复|上传|索引|重置|退款|开票|响应|解决|处理|重试)[^，。？！?]{0,12}?(?:后|之后).{0,40}(?:多久|几分钟|几小时|时间|时限|分钟|小时)', query)
        if event:
            sentences = [sentence for row in ranked for sentence in re.split(r'[。！？\n]', row.get('content', ''))]
            durations = [sentence for sentence in sentences if re.search(re.escape(event[1]) + r'.{0,40}(?:\d+|[一二三四五六七八九十]+)\s*(?:分钟|小时|天|秒)', sentence)]
            if not durations:
                missing.append('missing_event_specific_duration')
            elif re.search(r'保证|承诺|最迟|最多', query) and not any(re.search(r'保证|承诺|最迟|最多', sentence) for sentence in durations):
                missing.append('missing_guaranteed_duration')
    if missing:
        return {**common, 'evidence_status': 'insufficient', 'reason_codes': list(dict.fromkeys(missing)),
                'excluded_unsafe_chunks': excluded,
                'guidance': '当前检索没有足够证据支持所问事实。明确说明未知或缺少资料；可请用户提供正式资料/更具体问题，不得编造价格、证书、时限、渠道或功能。'}
    # No unchecked filenames/headings can bypass the existing text projection.
    projected = [{key: value for key, value in row.items() if key not in {'heading', 'filename', 'document_unsafe', 'org_id'}} for row in ranked]
    return tool_view({**common, 'evidence_status': 'candidates', 'results': projected,
                      'excluded_unsafe_chunks': excluded,
                      'guidance': '这些是相关候选段落，不是答案已被证明。逐项核对原文：只回答段落明确支持的事实并引用doc_id、revision、chunk_id和行号；没有明确依据的部分说明未知。不得把建议当作已执行，不推断未提供的官方渠道、时限或功能。'})
