# -*- coding: utf-8 -*-
"""公式编号修正：从公式 OMML 中移除编号 (N)，改放公式行最右侧右对齐（tab @ 9360 twips）。
AMS 规范：公式居中，编号行末右对齐。安全模式：备份 + TMP + testzip + copy2。"""
import zipfile, shutil, os, re, sys, time
from lxml import etree

DOCX = r'D:\liaohe\论文三\04_定稿投稿代_2026_R3投稿包与归档\投稿系统上传\Manuscript_R3_WAF.docx'
BAKDIR = r'D:\liaohe\论文三\04_定稿投稿代_2026_R3投稿包与归档\投稿系统上传\_backups'
os.makedirs(BAKDIR, exist_ok=True)
bak = os.path.join(BAKDIR, 'Manuscript_R3_WAF_bak_eq_%s.docx' % time.strftime('%Y%m%d_%H%M%S'))
shutil.copy2(DOCX, bak)
print('backup ->', bak)

W = 'http://schemas.openxmlformats.org/wordprocessingml/2006/main'
M = 'http://schemas.openxmlformats.org/officeDocument/2006/math'

z = zipfile.ZipFile(DOCX)
xml_bytes = z.read('word/document.xml')
z.close()

parser = etree.XMLParser(remove_blank_text=False)
root = etree.fromstring(xml_bytes, parser)

p_count = 0
for p in root.iter('{%s}p' % W):
    oms = list(p.iter('{%s}oMath' % M))
    if not oms:
        continue
    om = oms[0]
    runs = om.findall('.//{%s}r' % M)
    if not runs:
        continue
    last = runs[-1]
    ts = last.findall('{%s}t' % M)
    t_last = ''.join(x.text or '' for x in ts)

    n = None
    # 情况A：整 run 为 (N)
    mf = re.fullmatch(r'\((\d{1,2})\)', t_last)
    if mf:
        n = mf.group(1)
        last.getparent().remove(last)
    else:
        # 情况B：编号拆为 '(' 与 'N)'
        m2 = re.fullmatch(r'(\d{1,2})\)', t_last)
        if m2:
            n = m2.group(1)
            prev = runs[-2]
            t_prev = ''.join(x.text or '' for x in prev.findall('{%s}t' % M))
            if t_prev == '(':
                prev.getparent().remove(prev)
            last.getparent().remove(last)
        else:
            # 情况C：编号混在文本末尾
            m3 = re.search(r'\((\d{1,2})\)$', t_last)
            if m3:
                n = m3.group(1)
                if ts:
                    ts[-1].text = t_last[:m3.start()]
                else:
                    last.getparent().remove(last)
    if n is None:
        print('WARN: 未识别编号 run: %r' % t_last)
        continue

    # pPr 加右对齐制表位 @ 9360（文本宽 6.5in）
    pPr = p.find('{%s}pPr' % W)
    if pPr is None:
        pPr = etree.Element('{%s}pPr' % W)
        p.insert(0, pPr)
    if pPr.find('{%s}tabs' % W) is None:
        tabs = etree.Element('{%s}tabs' % W)
        tab = etree.SubElement(tabs, '{%s}tab' % W)
        tab.set('{%s}val' % W, 'right')
        tab.set('{%s}pos' % W, '9360')
        anchor = pPr.find('{%s}snapToGrid' % W)
        if anchor is not None:
            anchor.addprevious(tabs)
        else:
            sp = pPr.find('{%s}spacing' % W)
            if sp is not None:
                sp.addprevious(tabs)
            else:
                pPr.insert(0, tabs)

    # 段尾追加：tab + (N)
    r_tab = etree.SubElement(p, '{%s}r' % W)
    etree.SubElement(r_tab, '{%s}tab' % W)
    r_num = etree.SubElement(p, '{%s}r' % W)
    rpr = etree.SubElement(r_num, '{%s}rPr' % W)
    rf = etree.SubElement(rpr, '{%s}rFonts' % W)
    rf.set('{%s}ascii' % W, 'Times New Roman')
    rf.set('{%s}hAnsi' % W, 'Times New Roman')
    rf.set('{%s}cs' % W, 'Times New Roman')
    sz = etree.SubElement(rpr, '{%s}sz' % W); sz.set('{%s}val' % W, '24')
    szc = etree.SubElement(rpr, '{%s}szCs' % W); szc.set('{%s}val' % W, '24')
    t = etree.SubElement(r_num, '{%s}t' % W)
    t.text = '(%s)' % n
    t.set('{http://www.w3.org/XML/1998/namespace}space', 'preserve')
    p_count += 1
    print('段公式 %s: 编号移至行尾 (n=%s)' % (p_count, n))

new_xml = etree.tostring(root, xml_declaration=True, encoding='UTF-8', standalone=True)

# 重建 zip
tmp = DOCX + '.tmp.docx'
with zipfile.ZipFile(tmp, 'w', zipfile.ZIP_DEFLATED) as zout:
    with zipfile.ZipFile(DOCX) as zin:
        for item in zin.infolist():
            data = zin.read(item.filename)
            if item.filename == 'word/document.xml':
                data = new_xml
            zout.writestr(item, data)

with zipfile.ZipFile(tmp) as zt:
    bad = zt.testzip()
    if bad is not None:
        print('TESTZIP FAIL:', bad); sys.exit(1)
shutil.copy2(tmp, DOCX)
os.remove(tmp)
print('docx updated OK; 公式段处理数 =', p_count)
