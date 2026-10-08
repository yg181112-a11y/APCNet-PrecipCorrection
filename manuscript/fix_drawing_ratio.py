# -*- coding: utf-8 -*-
"""修正 docx 中所有图片显示比例：保持 cx(宽度) 不变，cy 按实际像素比重算，
消除 Word 显示时的横向/纵向拉伸（线条变斜、字体变形的根源）。"""
import zipfile, shutil, os, sys, time
from io import BytesIO
from lxml import etree
from PIL import Image

DOCX = r'D:\liaohe\论文三\04_定稿投稿代_2026_R3投稿包与归档\投稿系统上传\Manuscript_R3_WAF.docx'
BAKDIR = r'D:\liaohe\论文三\04_定稿投稿代_2026_R3投稿包与归档\投稿系统上传\_backups'
os.makedirs(BAKDIR, exist_ok=True)
stamp = time.strftime('%Y%m%d_%H%M%S')
bak = os.path.join(BAKDIR, 'Manuscript_R3_WAF_bak_ratio_%s.docx' % stamp)
shutil.copy2(DOCX, bak)
print('backup ->', bak)

NSW = 'http://schemas.openxmlformats.org/wordprocessingml/2006/main'
NSWP = 'http://schemas.openxmlformats.org/drawingml/2006/wordprocessingDrawing'
NSA = 'http://schemas.openxmlformats.org/drawingml/2006/main'
NSR = 'http://schemas.openxmlformats.org/officeDocument/2006/relationships'
NSPIC = 'http://schemas.openxmlformats.org/drawingml/2006/picture'

z = zipfile.ZipFile(DOCX)
media = {n: z.read(n) for n in z.namelist() if n.startswith('word/media/')}
xml = z.read('word/document.xml')

root = etree.fromstring(xml)
changed = 0
for inline in root.iter('{%s}inline' % NSWP):
    ext = inline.find('{%s}extent' % NSWP)
    blip = inline.find('.//{%s}blip' % NSA)
    if ext is None or blip is None:
        continue
    rid = blip.get('{%s}embed' % NSR)
    if not rid:
        continue
    # rId -> media
    m = None
    for name in media:
        pass
    # 通过 rels 映射
    rels = z.read('word/_rels/document.xml.rels').decode('utf-8')
    import re as _re
    mm = _re.search(r'Id="%s"[^>]*Target="media/(image\d+\.png)"' % rid, rels)
    if not mm:
        continue
    fname = 'word/media/' + mm.group(1)
    im = Image.open(BytesIO(media[fname]))
    w, h = im.size
    cx = int(ext.get('cx')); cy = int(ext.get('cy'))
    new_cy = round(cx * h / w)
    if abs(new_cy - cy) < 2:
        continue
    ext.set('cy', str(new_cy))
    # 同步 pic:spPr/a:xfrm/a:ext
    for aext in inline.iter('{%s}ext' % NSA):
        if aext.get('cx') == str(cx):
            aext.set('cy', str(new_cy))
    changed += 1
    print('%-12s cx=%d cy=%d -> cy=%d (px %dx%d)' % (mm.group(1), cx, cy, new_cy, w, h))

print('changed drawings:', changed)

# 序列化 + 重建 zip
data = etree.tostring(root, xml_declaration=True, encoding='UTF-8', standalone=True)
tmp = DOCX + '.tmp.docx'
with zipfile.ZipFile(tmp, 'w', zipfile.ZIP_DEFLATED) as zout:
    for item in z.infolist():
        if item.filename == 'word/document.xml':
            zout.writestr(item, data)
        else:
            zout.writestr(item, z.read(item.filename))
z.close()

with zipfile.ZipFile(tmp) as zt:
    bad = zt.testzip()
    if bad is not None:
        print('TESTZIP FAIL:', bad); sys.exit(1)
shutil.copy2(tmp, DOCX)
os.remove(tmp)
print('docx updated OK')
