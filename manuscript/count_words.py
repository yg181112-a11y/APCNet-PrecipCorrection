# -*- coding: utf-8 -*-
"""AMS 词数口径复核：正文 body + acknowledgments + appendixes ≤ 7500。
排除：标题页、摘要、Significance Statement、Data Availability、参考文献、
图表题注、表格内容、关键词。"""
import docx, re

P = r'D:\liaohe\论文三\04_定稿投稿代_2026_R3投稿包与归档\投稿系统上传\Manuscript_R3_WAF.docx'
d = docx.Document(P)

SECTIONS = ('ABSTRACT', 'SIGNIFICANCE STATEMENT', 'SIGNIFICANCE', 'DATA AVAILABILITY',
            'ACKNOWLEDGMENTS', 'ACKNOWLEDGEMENTS', 'APPENDIX', 'REFERENCES')
skip_state = 'none'  # none | abstract | sig | avail | refs
words = 0
ack_words = 0
counted = []
skipped = []

def clean(t):
    t = re.sub(r'\[EQ:[^\]]*\]', ' ', t)  # OMML 公式占位
    return t

for p in d.paragraphs:
    xml = p._p.xml
    ts = re.findall(r'<w:t(?:\s[^>]*)?>(.*?)</w:t>', xml, re.S)
    ms = re.findall(r'<m:t(?:\s[^>]*)?>(.*?)</m:t>', xml, re.S)
    text = ''.join(ts) + (' ' + ' '.join(ms) if ms else '')
    t = text.strip()
    upper = t.upper()
    # 状态切换
    if upper.startswith('ABSTRACT') and len(t) < 60:
        skip_state = 'abstract'; continue
    if (upper.startswith('SIGNIFICANCE STATEMENT') or upper.startswith('SIGNIFICANCE')) and len(t) < 60:
        skip_state = 'sig'; continue
    if upper.startswith('DATA AVAILABILITY') and len(t) < 60:
        skip_state = 'avail'; continue
    if upper.startswith('ACKNOWLEDG') and len(t) < 60:
        skip_state = 'ack'; 
    if upper.startswith('APPENDIX') and len(t) < 60:
        skip_state = 'appx'
    if upper.startswith('REFERENCES') and len(t) < 60:
        skip_state = 'refs'; continue
    if skip_state in ('refs', 'abstract', 'sig', 'avail'):
        continue
    # 标题段落（Heading 样式）不计入
    style = (p.style.name or '')
    if style.startswith('Heading'):
        continue
    # 图表题注不计入
    if re.match(r'^(Table\s+\d+|Fig\.\s*\d+|Figure\s+\d+)', t, re.I):
        continue
    # 表内文字已由表格对象承载，段落跳过表格单元格（python-docx 表格单元格段落不在 d.paragraphs 中）
    if not t:
        continue
    n = len(clean(t).split())
    words += n
    if skip_state == 'ack':
        ack_words += n
    counted.append((skip_state, n, t[:60]))

print('BODY+ACK+APPENDIX words =', words)
print('  of which acknowledgments =', ack_words)
print('AMS limit 7500; headroom =', 7500 - words)
