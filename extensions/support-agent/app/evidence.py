"""Bounded information-needs checks, not semantic entailment or confidence.

Rules are about question types, never dataset identifiers or reference answers.
Evidence must contain a concrete value or explicit limitation in a relevant
sentence. General procedural questions remain candidates for the answering
model to verify; this module does not certify arbitrary generated claims.
"""
from dataclasses import dataclass, replace
import math
import re
import unicodedata

from .search import tokenize

POLICY_VERSION = 'evidence-v4.1'
SCOPE = 'heuristic_information_needs_not_semantic_entailment'
NUMBER = r'(?:\d+(?:\.\d+)?|[一二两三四五六七八九十百]+)'
TIME = NUMBER + r'\s*(?:个)?(?:毫秒|秒|分钟|小时|自然日|工作日|天|日|周|月|年)'
UNKNOWN = re.compile(r'未(?:提供|披露|公开|确定|说明|给出|记录)|尚无|暂无|待(?:确认|核实|补充)|未知|不得编造|不详')
NEGATIVE = re.compile(r'(?:不能|无法|不予|并不|不再|不)(?:.{0,5})(?:保证|承诺|支持|提供)|不等于.{0,12}(?:保证|承诺)|不是.{0,12}(?:保证|承诺|时限)|不构成.{0,8}(?:保证|承诺)')
TOPICS = (
    (r'永久删除', r'永久删除'), (r'审计(?:日志|记录)?', r'审计'),
    (r'(?:密码|重置).{0,10}(?:链接|邮件)|(?:链接|邮件).{0,10}(?:重置|密码)', r'重置|密码'),
    (r'邀请', r'邀请'), (r'锁定|输错密码', r'锁定|输错密码'),
    (r'证书|saml|sso', r'证书|saml|sso'), (r'csv', r'csv'),
    (r'恢复', r'恢复'), (r'索引', r'索引'), (r'上传', r'上传'),
    (r'发票|开票', r'发票|开票'), (r'工单', r'工单|响应|优先级'),
    (r'存储|存放', r'存储|存放|数据'), (r'备份', r'备份'),
)


def norm(value):
    return unicodedata.normalize('NFKC', value).casefold()


def clarification(query):
    """Ask for the smallest useful business context, never account secrets."""
    value = norm(query)
    domain_words = '账号 账户 登录 密码 文件 文档 附件 上传 下载 导出 表格 报表 通知 邮件 消息 账单 发票 订单 付款 网络 连接 知识库 索引 成员 邀请 批量 登记 项目 归档 采购 审批 交接 编辑 协作 台账 报销 会议室 同步 设备 照片 验收'.split()
    has_object = any(word in value for word in domain_words)
    # Classify the user's request before treating words such as "异常" as an
    # observed incident. A policy question can legitimately mention failures.
    missing_description = r'(?:没|未|没有|不记得|忘了).{0,12}(?:错误内容|错误提示|现象|操作)'
    factual = r'多久|多少|几(?:位|天|分钟|小时|个月|行|次)|有效期|保留期|算法|税率|价格|编号|上限|时区|格式|编码|保证|承诺|支持.{1,16}吗'
    procedural = r'(?:如何|怎样|怎么).{1,24}(?:登记|记录|填写|提交|核对|区分|选择|判断|申请|查看|确认|计算)|(?:可以|是否|能否|允许|应当|应该).{1,24}(?:吗|么|[？?])'
    actionable = r'\b[145]\d\d\b|invalid\s+audience|乱码|收不到|未收到|忘记密码|锁定|登录失败|登录不上|上传失败|超时|签名验证失败|席位不足'
    deictic = r'(?:这样|那样|这个|那个|刚才|之前|还是).{0,16}(?:操作|提示|那样|意思|弄好|对不对|是否正确|继续)|(?:为什么|怎么会)(?:这样|那样)|(?:重新|再)来(?:一|1)次'
    if not has_object and re.search(deictic,value) and not re.search(r'(?:http\s*)?[145]\d\d|价格|税率|报价|有效期|保留期',value):
        if re.search(r'操作.{0,6}(?:对不对|是否正确)',value):
            return {'missing_fields':['object','operation'], 'questions':['你想确认哪个功能的哪一步操作？请描述具体操作内容与预期目的。']}
        return {'missing_fields':['object','symptom'], 'questions':['请补充涉及哪个功能，以及刚才提示的具体内容或实际现象；仅凭指代无法确定对应资料。']}
    rule_question = r'(?:核对|保留|提供|提交|记录|填写|选择|检查|登记|确认|由).{0,8}(?:哪些|什么|哪个|谁)|(?:哪些|什么|哪个).{1,12}(?:材料|信息|字段|条件|步骤)|(?:规范|规则|制度|手册).{0,8}(?:要求|规定)'
    if not re.search(missing_description,value) and (re.search(factual+'|'+procedural+'|'+actionable,value) or (has_object and re.search(rule_question,value))):
        return None
    vague_issue = r'出问题|不成功|出故障|有问题|不正常|不对劲|不对|不好使|不好用|不能用|不行|过不去|没反应|用不了|报错|异常|出错|坏了|(?:就|还是|仍然|又是)那样'
    if re.search(missing_description,value) or (re.search(vague_issue,value) and not re.search(r'\b[145]\d\d\b|\b[a-z]+[-_]?\d{2,}\b|[“"].{2,}[”"]|乱码|超时|失败.{0,6}(?:原因|因为)',value)):
        if has_object:
            return {'missing_fields':['operation','symptom'], 'questions':['请补充在哪一步操作遇到问题，以及具体的错误提示或现象；不要提供密码、验证码或完整密钥。']}
        return {'missing_fields':['object','symptom'], 'questions':['是哪个功能或业务对象出了问题？具体的错误提示或现象是什么？']}
    # Specific observed symptoms and factual questions are actionable. Merely
    # naming an object ("账号有问题") is not an observed symptom.
    if re.search(r'\b[145]\d\d\b|invalid\s+audience|乱码|收不到|未收到|忘记密码|锁定|登录失败|登录不上|上传失败|超时|签名验证失败|席位不足', value):
        return None
    if re.search(factual, value):
        return None
    # Naming a business area and asking for help does not specify an operation
    # or the desired outcome. Do not infer one from the highest-ranked article.
    if re.search(r'帮(?:我|忙|一下|着)|帮.{0,6}(?:看|处理)|看(?:一看|一下|看)|请教|求助',value):
        specific_request = re.search(r'如何|怎样|怎么|为什么|原因|步骤|流程|规则|条件|区别|需要.{1,12}(?:材料|信息)|[“"].{2,}[”"]',value)
        if not specific_request:
            if has_object:
                return {'missing_fields':['operation','symptom'], 'questions':['你希望在这个功能中完成什么操作？目前遇到了什么具体提示或现象？请不要提供密码、验证码或完整密钥。']}
            return {'missing_fields':['object','symptom'], 'questions':['你想处理哪个功能或业务对象？请补充具体诉求及遇到的提示或现象。']}
    filler = ('请问 请帮忙 请帮 帮忙 帮我 麻烦 看一下 看看 一下 想问 想知道 问一下 问问 '
              '为什么 怎么办 怎么处理 怎么解决 如何处理 怎么如何 什么 这个 那个 这些 那些 它 这边 那边 '
              '刚才的 刚刚 刚才 之前的 之前 现在 最近 今天 还是 仍然 依然 一直 总是 又 有点 有些 好像 '
              '遇到了 遇到 发生了 出现了 出现 有问题 有毛病 有故障 不正常 不对劲 不好使 不好用 不能用 不可用 用不了 '
              '没反应 没好 不行 失败 异常 出错 报错 错误 坏了 坏 故障 问题 帮助 处理 解决 修复 排查 '
              '我的 我们 我 你 的 了 吗 呀 吧 啊 呢 哦 是 有 很 能 不 用 请 个').split()
    generic_objects = '系统 软件 平台 功能 服务 页面 客户端 程序 应用 东西 操作 事情 内容 信息'.split()
    domain_objects = domain_words
    plain = re.sub(r'[\s，。！？?!、：:；;~～…“”"\'（）()]', '', value)
    # Match longest once; repeatedly deleting short characters can fabricate a
    # generic question from a concrete product/topic name.
    residual = re.sub('|'.join(map(re.escape, sorted(filler+generic_objects+domain_objects, key=len, reverse=True))), '', plain)
    if residual:
        return None
    if any(word in value for word in domain_objects):
        if re.search(r'账号|账户|登录|密码', value):
            question = '是在登录、重置密码还是其他账号操作时遇到问题？具体出现了什么提示或现象？请不要提供密码、验证码或重置链接。'
        elif re.search(r'文件|文档|附件|上传|下载|导出|表格|报表|索引', value):
            question = '是在上传、查看、索引还是导出哪一步遇到问题？具体的错误提示或现象是什么？'
        elif re.search(r'账单|发票|订单|付款', value):
            question = '是在查看账单、申请发票还是其他计费操作时遇到问题？具体提示是什么？请勿提供支付凭据。'
        else:
            question = '在哪一步操作遇到问题？具体出现了什么提示或现象？'
        return {'missing_fields':['operation','symptom'], 'questions':[question]}
    return {'missing_fields':['object','symptom'],
            'questions':['请说明是哪个产品、功能或业务对象遇到问题，以及具体的错误提示或现象；目前无法确定应检索哪类资料。']}


@dataclass(frozen=True)
class Need:
    code: str
    label: str
    subject: str
    proof: str
    topic: str = ''
    allow_negative: bool = False
    source_query: str = ''
    event: str = ''


# Each proof requires a value/relationship, not just a topic heading. Unknown
# statements and instructions to confirm a value are not numeric evidence.
RULES = (
    ('pricing','价格或费用数额',r'报价|多少钱|单价|年费|价格|收费标准',r'报价|单价|年费|价格|收费|费用',NUMBER+r'\s*元|[¥￥]\s*'+NUMBER),
    ('tax','税率或含税规则',r'含税|不含税|税率',r'税率|含税|不含税',r'税率.{0,12}'+NUMBER+r'\s*%|(?:价格|金额|报价|费用).{0,12}(?:已含税|不含税)|'+NUMBER+r'\s*元.{0,8}含税|含税价.{0,12}'+NUMBER),
    ('certificate','证书编号或有效信息',r'认证.{0,8}(?:编号|到期|有效期)|证书(?:编号|有效期|到期)',r'证书|认证',r'(?:证书|认证)(?:编号|号码|有效期|到期).{0,15}[a-z0-9][a-z0-9/-]{3,}'),
    ('key_bits','算法或密钥的位数',r'(?:算法|密钥|加密|证书).{0,12}(?:多少位|几位|位数|长度)|(?:多少位|几位).{0,8}(?:算法|密钥|加密)',r'算法|密钥|加密|证书|rsa|aes',r'(?:rsa|aes)[ -]?\d{3,4}|\d{3,4}\s*(?:位|bit)'),
    ('algorithm','具体算法名称',r'算法|加密方式|签名方式',r'算法|加密|签名|证书',r'\b(?:rsa|aes)(?:[- ]?\d+)?\b|hmac[- ]?sha[- ]?\d+|ecdsa|ed25519|sha[- ]?(?:256|512)'),
    ('retention','所问对象的保留时长',r'(?:保留|留存|保存).{0,12}(?:多久|多长|几|多少|时长|期限)|(?:多久|多长).{0,8}(?:保留|留存|保存)|保留期|留存期',r'保留|留存|保存',r'(?:保留|留存|保存).{0,24}'+TIME),
    ('storage_location','数据存储的明确地域或城市',r'(?:数据|存储|服务器|备份).{0,15}(?:城市|国家|地区|地域|在哪里|放在哪)|(?:哪个|哪座).{0,4}(?:城市|国家)',r'存储|存放|保存|服务器|数据|备份',r'(?:存储在|存放在|保存于|位于|所在地为|地域为|区域为).{0,8}(?:市|省|地区|区域|中国|美国|欧洲|新加坡)'),
    ('pagination','分页机制或分页大小',r'分页|每页|一页',r'分页|每页|一页|游标|cursor|page_size',r'(?:每页|分页大小|page_size).{0,12}\d+|(?:使用|采用|基于|通过).{0,12}(?:游标|cursor).{0,10}分页|不支持.{0,6}分页'),
    ('count_limit','所问对象的数量或大小上限',r'多少(?:行|条|个|列|人|次|份|张|台|件|mb)|几(?:行|条|列|人|份|张|台|件)|上限|最多|最大.{0,6}(?:文件|大小|数量|规模)|大小限制',r'上限|最多|最大|不得超过|不超过|限制|限额',r'(?:上限|最多|最大|不得超过|不超过|限制|限额).{0,24}'+NUMBER+r'\s*(?:行|条|个|列|人|次|份|张|台|件|mb|gb|kb)'),
    ('contact','可核验的联系渠道',r'热线|联系电话|联系邮箱|客服电话',r'热线|电话|邮箱|联系',r'[\w.+-]+@[\w.-]+\.[a-z]+|(?:电话|热线).{0,8}\+?[0-9][0-9 -]{5,}'),
    ('roadmap','明确的版本或上线计划',r'路线图|下个版本|未来版本|预计上线|计划上线',r'路线图|版本|上线',r'(?:计划|预计|将于).{0,20}(?:\d{4}年|\d{4}-\d|v\d|版本\d)'),
    ('private_deployment','部署能力的明确说明',r'私有化|本地部署|离线部署',r'私有化|本地部署|离线部署',r'(?:支持|不支持|可以|无法|不能).{0,12}(?:私有化|本地部署|离线部署)'),
    ('availability','可用率或赔偿标准',r'可用率|可用性百分比|赔偿标准|服务赔偿',r'可用率|可用性|赔偿',r'(?:可用率|可用性|赔偿).{0,20}'+NUMBER+r'\s*%'),
    ('accuracy','所问能力的准确率',r'准确率|正确率|识别率',r'准确率|正确率|识别率',r'(?:准确率|正确率|识别率).{0,20}'+NUMBER+r'\s*%'),
    ('resolution','图像或照片的像素要求',r'多少像素|几像素|分辨率',r'像素|分辨率|长边|短边',NUMBER+r'\s*(?:像素|px)'),
)


def topic_for(query):
    for trigger, topic in TOPICS:
        if re.search(trigger, query):
            return topic
    return ''


def clause_needs(query):
    value = norm(query)
    needs = [Need(code,label,subject,proof,topic_for(value) if code in {'certificate','key_bits','algorithm','retention','storage_location','pagination','count_limit'} else '', code in {'pagination','private_deployment'})
             for code,label,trigger,subject,proof in RULES if re.search(trigger,value)]
    if any(need.code == 'key_bits' for need in needs):
        needs = [need for need in needs if need.code != 'algorithm']
    if '税率' in value:
        needs = [Need(n.code,'具体税率',n.subject,r'税率.{0,12}'+NUMBER+r'\s*%',n.topic,n.allow_negative) if n.code=='tax' else n for n in needs]
    if re.search(r'草稿|提案',value) and not re.search(r'离线|本地|编辑|同步',value):
        # Expiry of a temporary draft is its usable lifetime, not a claim about
        # general data retention or how long an account keeps permissions.
        needs = [Need(n.code,'草稿或提案的可用期限',r'草稿|提案',
                      r'(?:草稿|提案).{0,15}'+TIME+r'.{0,8}(?:过期|失效)|(?:草稿|提案).{0,8}有效期.{0,8}'+TIME,
                      '',False) if n.code=='retention' else n for n in needs]
    unit = re.search(r'(?:多少|几)(行|条|列|人|次|份|张|台|件)|(?:行数|条数|列数)',value)
    if unit:
        requested = unit[1] or unit[0][0]
        needs = [Need(n.code,'所问对象的'+requested+'数上限',n.subject,NUMBER+r'\s*'+requested,n.topic,n.allow_negative) if n.code=='count_limit' else n for n in needs]
        if re.search(r'试(?:验|用)?|验证|示例|建议|首次|初次|抽样|样本',value) and not re.search(r'上限|最多|最大|限制',value):
            needs = [replace(n,code='trial_count',label='试验或建议使用的数量',subject=r'试|验证|映射|建议|初次|首次|样本|抽样|示例',
                             proof=NUMBER+r'\s*'+requested) if n.code=='count_limit' else n for n in needs]
    time_question = r'(?:几|多少)(?:个)?(?:毫秒|秒|分钟|小时|工作日|自然日|天|日|周|月|年)'
    if (re.search(time_question,value) and not re.search(r'行数|条数|数量|大小|多少(?:行|条)|几(?:行|条)|mb|gb',value)) or any(need.code=='retention' for need in needs):
        needs = [need for need in needs if need.code != 'count_limit']
    durations = r'多久|多长时间|有效期|时限|时间上限|多长|'+time_question
    if re.search(durations,value) and not any(need.code=='retention' for need in needs):
        specific = []
        for code,trigger,subject in (
            ('lock',r'锁定|(?:输错密码|密码.{0,6}输错).{0,12}(?:挡住|锁住|限制|再试|等待)',r'锁定'),
            ('invitation',r'邀请.{0,6}链接|邀请.{0,8}(?:有效期|过期)',r'邀请|链接'),
            ('reset',r'(?:重置|密码).{0,10}链接|链接.{0,10}(?:重置|密码)',r'链接|有效期'),
            ('restore',r'恢复',r'恢复'),
            ('index',r'索引',r'索引|队列'),
        ):
            if re.search(trigger,value):
                # "恢复后的索引" must not borrow the upload/index duration.
                if code == 'index' and re.search(r'恢复.{0,8}(?:后|之后)',value):
                    subject = r'恢复'
                proof = r'锁定(?:时间|时长)?(?:为|是|持续|约|不超过|最多)?\s*'+TIME if code=='lock' else TIME
                specific.append(Need('duration_'+code,{'lock':'账号锁定时长','invitation':'邀请链接有效时长','reset':'重置链接有效时长','restore':'恢复完成时长','index':'索引完成时长'}[code],subject,proof,
                                     r'邀请' if code=='invitation' else r'密码|重置' if code=='reset' else '',True))
        if not specific:
            subject = temporal_subject(value)
            proof = r'(?:'+subject+r').{0,30}'+TIME+'|'+TIME+r'.{0,30}(?:'+subject+')'
            specific = [Need('duration','明确时长（'+query.strip('？?。 ')[:50]+'）',subject,proof,topic_for(value),True,event=subject)]
        needs.extend(specific)
    if re.search(r'保证|承诺|一定|确保|sla|最迟',value):
        needs.append(Need('guarantee','是否存在所要求的保证或承诺',r'保证|承诺|sla',r'(?:保证|承诺|sla).{0,32}'+TIME,topic_for(value),True,event=temporal_subject(value)))
    return needs


def temporal_subject(query):
    # Prefer the event being timed over generic words like "小时/时间".
    events = r'续期|刷新|接管|核对|审批|交接|审阅|响应|索引|恢复|上传|导出|下载|归档|停用|同步|释放|重试|退款|开票|生效|发送|送达|收到|过期|有效|保留|留存|保存|等待|完成|处理'
    for event in events.split('|'):
        if re.search(event,query):
            return event
    terms = salient_terms(query)
    return '|'.join(map(re.escape,terms)) if terms else r'(?!)'


def salient_terms(query):
    plain = re.sub(r'请问|请|告诉我|说明|需要|应该|可以|如何|怎么|步骤|流程|怎样|哪些|什么|哪里|哪一天|哪个|多久|多少|多长时间|几分钟|几小时|几天|有效期|时限|时间|间隔|上限|最大|最多|具体|是多少|是否|是|的|吗|呢|了|要|应|最晚|至少|提前|启动', ' ',norm(query))
    return sorted(set(tokenize(plain)))


def needs_for(query):
    clauses = [part.strip() for part in re.split(r'[，,；;？?。]|以及|另外|同时',query) if part.strip()]
    groups = []
    for index,clause in enumerate(clauses):
        resolved = clause
        # Resolve a short time question only against an explicit immediately
        # preceding event in this same request; never invent conversation state.
        if index and re.search(r'^(?:那|那么|现在|还|过|要|需|需要|应|应该|得)?(?:等)?(?:多久|多长时间|几分钟|几小时)',clause):
            previous = clauses[index-1]
            if re.search(r'锁定|(?:密码.{0,6}输错|输错密码).{0,12}(?:挡住|锁住)|草稿|提案|邀请链接|重置链接|上传|恢复|索引',previous) and not clause_needs(previous):
                resolved = previous+'，'+clause
        groups.append((resolved,clause_needs(resolved)))
    result = [replace(need,source_query=clause) for clause,needs in groups for need in needs]
    if result:
        for clause,needs in groups:
            if needs or not re.search(r'如何|怎么|步骤|流程|怎样|哪些|什么|哪一天|哪个|谁',clause):
                continue
            terms = salient_terms(clause)
            if terms:
                result.append(Need('context_fact','已有原文可回答的部分（'+clause[:50]+'）','|'.join(map(re.escape,terms)),
                                   r'请|使用|通过|进入|打开|提交|检查|确认|记录|选择|先|然后|按|由|为|是|包含|应|须', source_query=clause))
    return result


def sentences(row):
    content = row.get('content','')
    for match in re.finditer(r'[^。！？!?；;\n]+[。！？!?；;]?', content):
        text = match.group().strip()
        # Markdown headings and source labels are not factual assertions.
        if not text or text.startswith(('#','来源：','来源:')):
            continue
        start = match.start()+len(match.group())-len(match.group().lstrip())
        yield text,start,start+len(text)


def reference(row, text, start, end):
    base = row.get('start_offset',0)
    return {'doc_id':row.get('doc_id',row.get('id')), 'revision':row.get('revision',1),
            'chunk_id':row.get('chunk_id',row.get('id')), 'start_offset':base+start,
            'end_offset':base+end, 'quote':text}


def evidence_for(need, rows, query):
    found = []
    value = norm(need.source_query or query)
    for row in rows:
        context = norm(str(row.get('title',''))+' '+str(row.get('heading','')))
        previous = ''
        for text,start,end in sentences(row):
            sentence = norm(text)
            earlier = previous
            previous = sentence
            if not re.search(need.subject,sentence):
                continue
            limited_scope = r'试验|试点|测试|抽样|样本|首批|首次|首轮|初次|先(?:用|选|取|导入)|核对映射|示例|例如|建议'
            if need.code == 'count_limit' and re.search(limited_scope,sentence) and not re.search(limited_scope,value):
                continue
            if need.code == 'context_fact':
                tokens = set(salient_terms(value))
                if len(tokens & set(tokenize(sentence))) < max(1,min(2,len(tokens)),math.ceil(len(tokens)*.45)):
                    continue
            if not relevant_subject(need,sentence,context,earlier):
                continue
            if not matching_edition(value,sentence,context):
                continue
            # "确认价格/保留多久" or an explicit unknown is not a supplied fact.
            if UNKNOWN.search(sentence) or re.search(r'(?:请|需|需要|应|要).{0,8}(?:确认|核实|咨询|询问).{0,20}(?:'+need.subject+')',sentence):
                continue
            negative = NEGATIVE.search(sentence)
            if negative and need.allow_negative:
                # A broad inability to guarantee restoration also rules out a
                # requested one-hour guarantee, without inventing a time value.
                if need.code.startswith('duration') and not re.search(r'保证|承诺|一定|确保|最迟',value):
                    continue
                # Negative evidence must concern this requirement within one
                # clause, not an unrelated hotline/permission elsewhere.
                clauses = re.split(r'[，,]',sentence)
                if not any(NEGATIVE.search(part) and re.search(need.subject,part) and event_matches(need,part,value)
                           and not (need.code.startswith('duration') and re.search(r'不是|不等于|不构成',part)) for part in clauses):
                    continue
                found.append(reference(row,text,start,end))
            elif re.search(need.proof,sentence):
                if need.code.startswith('duration_') and not any(re.search(need.subject,part) and re.search(need.proof,part) for part in re.split(r'[，,]',sentence)):
                    continue
                if need.code in {'duration','guarantee'} and not any(re.search(need.proof,part) and event_matches(need,part,value) for part in re.split(r'[，,]',sentence)):
                    continue
                if need.code in {'pricing','tax','retention','count_limit','trial_count','resolution','accuracy','pagination','key_bits','algorithm','certificate','storage_location'}:
                    if not any(re.search(need.subject,part) and re.search(need.proof,part) for part in re.split(r'[，,]',sentence)):
                        continue
                if need.code.startswith('duration') and re.search(r'保证|承诺|一定|确保|最迟',value) and not re.search(r'保证|承诺|最迟|最多',sentence):
                    continue
                found.append(reference(row,text,start,end))
            if found:
                return found
    return found


def event_matches(need, clause, query):
    if not need.event:
        return True
    if re.search(need.event,clause):
        return True
    # A general work-order SLA disclaimer can rule out a promised response,
    # but cannot prove unrelated takeover, recovery or deletion guarantees.
    return (need.event == '响应' and '工单' in query
            and bool(re.search(r'sla|服务等级协议',clause))
            and bool(NEGATIVE.search(clause)))


def relevant_subject(need, sentence, title, previous):
    if not need.topic or re.search(need.topic,sentence):
        return True
    subjects = (r'审计',r'聊天|会话',r'邀请',r'密码|重置',r'证书|saml|sso',r'csv',r'上传',r'恢复',r'备份')
    # Never let a broad title lend its subject to another explicitly named
    # object's value, e.g. "审计与聊天" + "聊天记录保留30天".
    if any(re.search(subject,sentence) and not re.search(need.topic,re.search(subject,sentence)[0]) for subject in subjects):
        return False
    for context in (previous,title):
        if not re.search(need.topic,context):
            continue
        matched = [subject for subject in subjects if re.search(subject,context)]
        if all(re.search(need.topic,re.search(subject,context)[0]) for subject in matched):
            return True
    return False


def matching_edition(query, sentence, title):
    editions = r'标准版|专业版|企业版|基础版|免费版|旗舰版|enterprise|standard|business|\bpro\b'
    requested = set(re.findall(editions,query))
    if not requested:
        return True
    stated = set(re.findall(editions,sentence))
    if stated:
        return requested <= stated
    title_editions = set(re.findall(editions,title))
    return title_editions == requested


def assessment(query, ranked, *, clarify=None):
    result = {'policy_version':POLICY_VERSION,'assessment_scope':SCOPE,'coverage':'none',
              'supported_claims':[],'missing_information':[],'missing_requirements':[],
              'missing_fields':[], 'clarification_questions':[]}
    if clarify:
        result.update(missing_fields=clarify['missing_fields'], clarification_questions=clarify['questions'])
        return result
    needs = needs_for(query)
    for need in needs:
        evidence = evidence_for(need,ranked,query)
        if evidence:
            result['supported_claims'].append({'requirement':need.code,'label':need.label,'evidence':evidence})
        else:
            result['missing_requirements'].append(need.code)
            result['missing_information'].append('尚缺“'+need.label+'”的直接原文依据。')
    if not ranked:
        result['missing_information'] = result['missing_information'] or ['当前资料未检索到相关证据，请补充具体业务信息或有权使用的资料。']
        return result
    if not needs:
        # General instructions remain unverified candidates. full is expressly
        # only completion of detected checks, not proof of the entire answer.
        result['coverage'] = 'full'
        result['assessment_scope'] = 'no_precise_fact_need_detected_candidates_require_verification'
    elif not result['missing_requirements']:
        result['coverage'] = 'full'
    elif result['supported_claims']:
        result['coverage'] = 'partial'
    return result
