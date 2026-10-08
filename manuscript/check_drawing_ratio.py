# -*- coding: utf-8 -*-
"""核查 docx 中每张图的显示尺寸比例 vs 实际像素比例，找出被拉伸的图。"""
import zipfile, re
from xml.etree import ElementTree as ET
from PIL import Image

DOCX = r'D:\liaohe\论文三\04_定稿投稿代_2026_R3投稿包与归档\投稿系统上传\Manuscript_R3_WAF.docx'
NSW = '{http://schemas.openxmlformats.org/wordprocessingml/2006/main}'
NSP = '{http://schemas.openxmlformats.org/drawingml/2006/wordprocessingDrawing}'

z = zipfile.ZipFile(DOCX)
root = ET.fromstring(z.read('word/document.xml'))
rels = z.read('word/_rels/document.xml.rels').decode('utf-8')
rid2media = dict(re.findall(r'Id="(rId\d+)"[^>]*Target="media/(image\d+\.png)"', rels))

# drawing 顺序
order = re.findall(r'r:embed="(rId\d+)"', z.read('word/document.xml').decode('utf-8'))
media_order = [rid2media[r] for r in order if r in rid2media]
seen, media_seq = set(), []
for m in media_order:
    if m not in seen:
        seen.add(m); media_seq.append(m)

print('embedded media order:', media_seq)

# 每张图 extent（EMU: cx/cy）与像素比
for i, m in enumerate(media_seq, 1):
    # 找到该 media 对应的 drawing（按顺序取第 i 个 inline/drawing）
    # 简单法：遍历所有 wp:extent 按出现顺序
    pass

# 遍历所有 wp:inline 中的 extent + blip rId，按文档顺序配对
extents = []
for inline in root.iter(NSP + 'inline'):
    ext = inline.find(NSP + 'extent')
    blip = inline.find('.//{http://schemas.openxmlformats.org/drawingml/2006/main}blip')
    if ext is None or blip is None:
        continue
    rid = blip.get('{http://schemas.openxmlformats.org/officeDocument/2006/relationships}embed')
    extents.append((rid, int(ext.get('cx')), int(ext.get('cy'))))

print('\n%-6s %-12s %-10s %-10s %-10s %-8s' % ('#', 'media', 'cx(EMU)', 'cy(EMU)', 'disp_ratio', 'px_ratio'))
for i, (rid, cx, cy) in enumerate(extents, 1):
    m = rid2media.get(rid)
    if not m:
        continue
    img = Image.open(z.extract(m, temp_dir if False else None)) if False else None
    data = z.read('word/media/' + m)
    im = Image.open(__import__('io').BytesIO(data))
    w, h = im.size
    disp_ratio = cx / cy
    px_ratio = w / h
    diff = abs(disp_ratio - px_ratio) / px_ratio * 100
    flag = ' <<< STRETCHED' if diff > 0.5 else ''
    print('%-6d %-12s %-10d %-10d %-10.3f %-8.3f diff %.1f%%%s' % (i, m, cx, cy, disp_ratio, px_ratio, diff, flag))
