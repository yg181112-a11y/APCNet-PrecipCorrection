# -*- coding: utf-8 -*-
"""P3: 将 v1 手稿写入 TARGET（Manuscript_R3_WAF_draft.docx），AMS 小节风格。
保留：标题/作者/单位/通讯/Acknowledgments/Author Contributions/Availability。
替换：标题文本、Abstract、Significance Statement、正文(1-5节含表图)、References。
"""
import os, copy
from docx import Document
from docx.shared import Pt, Inches, RGBColor
from docx.enum.text import WD_ALIGN_PARAGRAPH
from docx.oxml.ns import qn
from docx.oxml import OxmlElement

TARGET = r'C:\Users\yg181\Desktop\论文三\Manuscript_R3_WAF_draft.docx'
FIG_DIR = r'D:\liaohe\校正优化过程\第三阶段\12优化\fig_p3_pub'
FIG1 = r'C:\Users\yg181\Desktop\论文三\13.0修复重跑\r3_media\word\media\image1.png'

import shutil
SOURCE = r'C:\Users\yg181\Desktop\论文三\Manuscript_R2_revised_final_backup_figs.docx'
shutil.copy(SOURCE, TARGET)  # 每次从原始 R2 稿重建，保证可重复运行
doc = Document(TARGET)
body = doc.element.body

# ---------- 0. fix preserved header areas (author line, Availability GitHub link) ----------
def fix_run_text(doc, old, new):
    n = 0
    for p in doc.paragraphs:
        for r in p.runs:
            if old in r.text:
                r.text = r.text.replace(old, new)
                n += 1
    return n

fix_run_text(doc, 'Xin Huang,ab ,Yuhan Jiang', 'Xin Huang,ab, Yuhan Jiang')
fix_run_text(doc, 'https://github.com/TianLin-CCIT/APCNet-PrecipCorrection',
             'https://github.com/yg181112-a11y/APCNet-PrecipCorrection')
fix_run_text(doc, ' ,Yuhan Jiang', ', Yuhan Jiang')

# ---------- helpers ----------
def find_para(text_prefix, start=0):
    for i, p in enumerate(doc.paragraphs):
        if i < start:
            continue
        t = p.text.strip()
        if t.startswith(text_prefix):
            return i, p
    return None, None

def clear_para(p):
    for r in list(p.runs):
        r._r.getparent().remove(r._r)

def set_para_text(p, text):
    clear_para(p)
    p.add_run(text)

def new_para_before(anchor_p, text='', style=None, align=None, bold=False, size=None, italic=False):
    p = anchor_p.insert_paragraph_before()
    if style is not None:
        try:
            p.style = style
        except Exception:
            pass
    if text:
        run = p.add_run(text)
        if bold:
            run.bold = True
        if italic:
            run.italic = True
        if size:
            run.font.size = Pt(size)
    if align is not None:
        p.alignment = align
    return p

def add_table_before(anchor_p, headers, rows, style='Table Grid', font_size=9):
    tbl = doc.add_table(rows=1 + len(rows), cols=len(headers))
    tbl.style = style
    hdr = tbl.rows[0].cells
    for j, h in enumerate(headers):
        hdr[j].text = h
        for para in hdr[j].paragraphs:
            for r in para.runs:
                r.bold = True
                r.font.size = Pt(font_size)
    for i, row in enumerate(rows):
        cells = tbl.rows[i + 1].cells
        for j, v in enumerate(row):
            cells[j].text = str(v)
            for para in cells[j].paragraphs:
                for r in para.runs:
                    r.font.size = Pt(font_size)
    # move tbl before anchor
    anchor_p._p.addprevious(tbl._tbl)
    return tbl

def add_figure_before(anchor_p, img_path, width_in=5.2, caption=None, cap_before=False):
    p = anchor_p.insert_paragraph_before()
    p.alignment = WD_ALIGN_PARAGRAPH.CENTER
    run = p.add_run()
    run.add_picture(img_path, width=Inches(width_in))
    if caption:
        cp = anchor_p.insert_paragraph_before()
        cp.alignment = WD_ALIGN_PARAGRAPH.CENTER
        r = cp.add_run(caption)
        r.font.size = Pt(9)
        r.italic = False
    return p

def delete_between(start_el, end_el):
    """删除 start_el 与 end_el 之间的所有 w:p / w:tbl / w:sdt 元素（不含两端）"""
    el = start_el.getnext()
    while el is not None and el is not end_el:
        nxt = el.getnext()
        if el.tag in (qn('w:p'), qn('w:tbl'), qn('w:sdt')):
            body.remove(el)
        el = nxt

# ---------- 1. locate anchors ----------
idx_title, p_title = find_para('Physics-Guided and Neighborhood-Aware')
idx_abs, p_abs_label = find_para('Abstract')
abs_content_idx = None
if p_abs_label is not None:
    abs_content_idx, p_abs_content = find_para('Numerical weather prediction (NWP) models', idx_abs + 1)
idx_sig, p_sig_label = find_para('Significance Statement')
sig_content_idx, p_sig_content = find_para('Numerical weather forecasts over mountainous terrain', idx_sig + 1)
idx_intro, p_intro = find_para('1. Introduction')
idx_author, p_author = find_para('Author Contributions', idx_intro + 1)
idx_ref, p_ref = find_para('References', idx_author + 1)

print('anchors:', idx_title, idx_abs, abs_content_idx, idx_sig, sig_content_idx, idx_intro, idx_ref)

# ---------- 2. replace title / abstract / significance ----------
set_para_text(p_title, 'Skill limits and transferability of deep-learning precipitation post-processing trained on reanalysis targets: consistent failure across ERA5, CHM, and GPM verification over Northeast China')

ABSTRACT = ('Deep-learning (DL) post-processing of NWP precipitation increasingly uses reanalysis products such as '
            'ERA5 as training targets, implicitly assuming that the reanalysis is an unbiased learning reference. Using 11 '
            'years of GFS f003 forecasts and ERA5 targets over the Liaohe basin, Northeast China, we test this assumption. '
            'First, we discovered and corrected a cumulation-window mismatch in the ERA5 target (1-hour accumulations sampled '
            'at 3-hourly steps treated as 3-hour accumulations) that had produced a spurious 3x scale offset and invalidated '
            'all previous quantitative conclusions; every result here is recomputed from the corrected, gate-verified target. '
            'Second, a controlled experiment in which the training target is GFS plus controllable Gaussian noise shows that '
            'the same network learns a noise-free identity almost perfectly (correlation 0.998) but degrades monotonically as '
            'target noise increases, reaching -27.2% at the observed residual level: negative skill is a property of the '
            'target, not of the architecture. Third, the failure is consistent across every reference. A bin-based '
            'conditional-expectation benchmark improves MSE by +27.3% on the ERA5 reference and +22.6% against GPM IMERG, '
            'while the DL network degrades MSE by -32.6% to -63.1% on the reference and -29.3% on GPM. Against CHM daily '
            'rates and GPM 3-hour accumulations alike, the DL networks are the only methods with significantly negative skill '
            '(-29% to -59%), while quantile mapping (+3.6%/+8.3%) and linear regression (+12.2%/+19.6%) remain significantly '
            'positive. Reanalysis-trained DL correction inherits the smoothness and noise of its reference; simple grid-point '
            'baselines remain the strongest operational choice.')

set_para_text(p_abs_content, ABSTRACT)
p_kw = new_para_before(p_sig_label, 'KEYWORDS: precipitation post-processing; deep learning; reanalysis training target; '
                         'bias correction; quantile mapping; independent verification; Northeast China.',
                       style=None, size=9)


SIGNIFICANCE = ('Numerical weather forecasts over Northeast China often carry systematic precipitation errors, and deep-learning '
                '(DL) correction trained on reanalysis targets such as ERA5 has been proposed to fix them. Using 11 years of GFS '
                'forecasts over the Liaohe basin, we show that after correcting a target-data error in the ERA5 reference, a '
                'correction network that almost perfectly learns a noise-free mapping degrades forecasts whenever the reanalysis '
                'target is noisy, and that on two independent verification products the DL networks (APCNet and a standard U-Net) are the only methods that make '
                'forecasts significantly worse, while simple quantile mapping and linear regression improve them. '
                'Reanalysis-trained DL post-processing inherits the smoothness and noise of its reference and, in this setup, '
                'provides no benefit over simple grid-point baselines. We recommend that DL post-processing studies verify '
                'training targets, report simple baselines, and evaluate on independent verification products before operational deployment.')
set_para_text(p_sig_content, SIGNIFICANCE)

# ---------- 3. delete old body between intro and Author Contributions ----------
delete_between(p_intro._p, p_author._p)

# ---------- 4. delete old reference entries (after References label) ----------
# 收集 References 之后的段落并删除
el = p_ref._p.getnext()
while el is not None:
    nxt = el.getnext()
    if el.tag in (qn('w:p'), qn('w:tbl'), qn('w:sdt')):
        body.remove(el)
    el = nxt

# ---------- 5. insert new body before References ----------
A = p_author  # anchor; 所有新内容 addprevious 到 Author Contributions 前

def H1(text):
    return new_para_before(A, text, style=None, bold=True, size=12)

def H2(text):
    return new_para_before(A, text, style=None, bold=True, size=11)

def P(text, italic=False, size=10):
    return new_para_before(A, text, style=None, size=size, italic=italic)

def FIG(img, width, cap):
    add_figure_before(A, os.path.join(FIG_DIR, img), width_in=width, caption=cap)

def TAB(headers, rows):
    add_table_before(A, headers, rows)

def NOTE(text):
    p = new_para_before(A, text, style=None, size=9)
    return p

# ---- 1. Introduction ----
P('Numerical weather prediction (NWP) precipitation suffers from systematic biases in intensity and location, and statistical '
  'post-processing has long been the standard remedy in operational forecasting (Vannitsem et al. 2021). Classical approaches-quantile mapping (QM) '
  '(Zhu and Luo 2015; Hamill et al. 2017; Hamill and Scheuerer 2018), model output statistics (MOS), and grid-point '
  'regression-remain widely used because they are cheap, robust, and improve skill scores on the data they are calibrated on. '
  'Over the past few years, deep learning (DL) has been proposed as a more powerful alternative: convolutional and U-Net '
  'architectures, adversarial training, and generative models have been applied to correct NWP precipitation, with reported '
  'improvements over raw forecasts and over classical baselines on reanalysis or analysis targets (Harris et al. 2022; Wang '
  'et al. 2023; Hess and Boers 2022; Sha et al. 2022; Hu et al. 2021; Chen et al. 2023). Adversarial and generative '
  'variants have recently been extended to kilometre-scale extreme-precipitation post-processing over China (Yang et al. 2025; '
  'Fang et al. 2025; Xu et al. 2026).')
P('A shared design choice in most DL precipitation post-processing studies is the training target: because high-resolution '
  'gridded observations are often unavailable over long periods or over complex terrain, reanalysis products, most commonly '
  'ERA5, are used as the learning reference. The justification is pragmatic: reanalyses are spatially complete, physically '
  'consistent, and share the grid and variables of the model being corrected (Tarek et al. 2020). ERA5 reproduces the relative '
  'hourly and daily distribution of precipitation over China reasonably well, although its absolute values carry '
  'region-dependent biases (Wu et al. 2024), and its capacity to capture Chinese precipitation extremes is bounded (Lei et al. '
  '2022). The implicit assumption is that the reanalysis is an adequate, unbiased surrogate for the truth, and that a model '
  'trained against it will improve the forecast in a way that transfers to the real world.')
P('This assumption deserves scrutiny. Reanalysis precipitation is itself a model product: smooth in space and time, and '
  'carrying its own systematic differences from gauge- and satellite-merged observations. Several recent studies report that '
  'DL precipitation correction trained on reanalyses performs worse than expected on independent verification products, or that simple '
  'baselines remain competitive in specific regions, and CNN-based downscaling of daily precipitation over China has been reported to be outperformed by quantile mapping and regression-based correction (Hess and Boers 2022; Worsnop et al. 2024; Sun and Lan 2021). Conversely, when trained on '
  'long reforecast-observation pairs, CNN post-processing has been reported to beat classical MOS on the same reference '
  '(Badrinath et al. 2023). Whether the claimed DL advantage survives the choice of training reference-reanalysis versus '
  'observation-based targets-and whether the learned correction is valid at the observation scale, remains an open question. '
  'Two questions are rarely answered quantitatively: (i) can a well-designed DL architecture learn anything from a reanalysis '
  'target, and what exactly limits its skill? (ii) when a DL correction fails on the training reference, does it fail, or '
  'succeed, at the observation scale?')
P('We address both questions with a case study over the Liaohe basin in Northeast China, a region dominated by the Northeast '
  'China cold vortex (NECV) and its organized precipitation (Yang et al. 2024). We use 11 years (2015-2025) of operational GFS f003 3-hourly '
  'forecasts as input, ERA5 as the training reference, and two independent verification products: the CHM '
  'gauge-merged daily product (2024-2025) and GPM IMERG half-hourly satellite precipitation aggregated to 3 h (2024-2025).')
P('Our main contributions are threefold. (1) Data integrity. During final quality assurance we discovered that the ERA5 '
  'target used in earlier versions of this study suffered from a cumulation-window mismatch (1-h accumulation values sampled '
  'at 3-hourly steps paired against true 3-h GFS accumulations), producing a spurious 3x scale offset that invalidated all '
  'previous quantitative claims (including a previously reported 46.9% MSE reduction). We corrected the target (hourly '
  're-download, true 3-h reconstruction), verified it with a six-item acceptance gate, and recomputed every result in this '
  'paper from the corrected data. Section 2d and the supplementary tables document the problem, the evidence, and the fix. '
  '(2) Causal diagnosis of negative skill. Through a controlled experiment in which the training target is the GFS input plus '
  'controllable Gaussian noise, we show that the same network that learns an identity mapping almost perfectly (spatial '
  'correlation 0.998) exhibits monotonically negative skill as target noise increases, and that at the noise level matching '
  'the observed GFS-ERA5 residual (sigma = 1.16 mm/3h) the MSE improvement over raw GFS is -27.2%, consistent with the '
  '-32.6% to -63.1% range of the real task across three seeds. The negative skill of DL correction on its training reference '
  'is a property of the target signal-to-noise ratio, not of the architecture or of training choices. (3) A transferability '
  'boundary that is consistent across all references. On the training reference, a bin-based conditional-expectation '
  'benchmark (E[ERA5|GFS], +27.3%) and simple baselines dominate: grid-point-wise linear regression (+22.2%) and QM (+11.1%) '
  'beat the network (-32.6% to -63.1%). On two independent verification products the ranking is identical: against CHM daily '
  'rates the network degrades RMSE by -31.9% to -58.7% (all seeds, monthly block bootstrap 95% CI [-41.3, -26.3], '
  'P(negative) = 1.000) while QM improves by +3.6% and OLS by +12.2%; against GPM IMERG 3-h accumulations the network '
  'degrades MSE by -29.3% while the bin benchmark improves by +22.6%, QM by +8.3% and OLS by +19.6%. The DL correction '
  'over-wets the domain (predicted domain-mean rate 0.58 mm/h vs observed 0.25 on the GPM scale) and its extreme-precipitation '
  'skill is at best equal to raw GFS. Reanalysis-trained DL post-processing inherits the reference\'s smoothness and noise; '
  'the reference\'s conditional structure is learnable by simple nonparametric estimators but is not captured by the network, '
  'and grid-point simple baselines remain the strongest operational choice on both references and observations.')
P('The remainder of the paper is organized as follows. Section 2 describes the data, the target-correction procedure, the '
  'network, the baselines, and the evaluation protocol. Section 3 presents the results: target verification, the main skill '
  'comparison, the controlled experiment, spatial and daily-scale diagnostics, independent verification against CHM and GPM, '
  'extreme-precipitation behavior, and probabilistic post-processing. Section 4 discusses the implications for operational DL '
  'post-processing and Section 5 concludes.')

# ---- 2. Data and methods ----
H1('2. Study Area, Data, and Methods')
H2('a. Study Area and Data')
P('The Liaohe basin (40-46 N, 117-126 E), 25x37 grid points at 0.25 resolution. Precipitation is strongly modulated by the '
  'Northeast China cold vortex (NECV) and organized mesoscale systems (Fig. 1); both convective extremes and long-duration '
  'light rain occur, and deterministic NWP precipitation still carries systematic errors over China (Zhang et al. 2021; Zhu et '
  'al. 2018), and verification against dense observation networks highlights persistent near-surface biases in operational '
  'models such as GFS (Gaudet et al. 2024).')
add_figure_before(A, FIG1, width_in=4.6, caption='Fig. 1. Topographic map of the Liaohe River Basin showing the spatial '
  'domain (40-46 N, 117-126 E). Color shading denotes normalized elevation (relative to 1.0 km); the complex terrain '
  'configuration strongly modulates local precipitation processes in this typical East Asian monsoon region.')

P('Input: GFS f003. Operational GFS forecasts at 0.25, f003 (3-h lead), 2015-2025, 16,056 files (4 cycles/day, 00/06/12/18Z). '
  'The precipitation variable is A_PCP_L1_Accum_1 (kg m-2), a true 3-h accumulation (time_bnds = [0,3], cell_methods = time: '
  'sum over hours), valid at 03/09/15/21Z. Input channels (8): CAPE, PWAT, U850, V850, U500, V500, VVEL, and precipitation, '
  'plus sequence context of 6 input times at 6-h intervals covering t-30h to t (matched to the available 03/09/15/21Z cycle '
  'times). Table 1 lists the variables and units.')
TAB(['GFS Variable', 'Unit', 'ERA5 Variable', 'Unit'],
    [['Convective Available Potential Energy (CAPE)', 'J kg-1', 'CAPE', 'J kg-1'],
     ['Precipitable Water (PWAT)', 'kg m-2', 'Total Column Water Vapour (tcwv)', 'kg m-2'],
     ['U/V wind at 850 hPa', 'm s-1', 'U/V wind at 850 hPa', 'm s-1'],
     ['U/V wind at 500 hPa', 'm s-1', 'U/V wind at 500 hPa', 'm s-1'],
     ['Total Precipitation', 'mm (3h)-1', 'Total Precipitation', 'mm (3h)-1'],
     ['Vertical Velocity', 'hPa s-1', 'Vertical Velocity', 'hPa s-1']])
NOTE('Table 1. Meteorological forecast variables and units of GFS and ERA5 data.')
P('Target: ERA5 (reconstructed). ERA5 total precipitation tp (Hersbach et al. 2020) was downloaded at hourly resolution from '
  'the CDS and reconstructed into true 3-h accumulations by summing the three hourly values within each window, TP_3h(t) = '
  'tp(t-2h) + tp(t-1h) + tp(t), for t in {03, 09, 15, 21}Z (Section 2d).')
P('Independent verification 1: CHM. The China Meteorological Administration Multisource Precipitation Merging System (CHM) '
  'daily product (0.1; a gauge-satellite merged analysis in the family of Shen et al. 2014), 2024-2025, bilinearly '
  'interpolated to the study grid. Coverage mismatch: model/GFS/ERA5 fields cover 12 h per day (four 3-h windows) whereas CHM '
  'is a full 24-h daily accumulation; CHM comparisons are made on a rate basis (mm h-1) with relative, not absolute, '
  'interpretation (caveat in Section 4d).')
P('Independent verification 2: GPM IMERG. NASA GPM IMERG Final half-hourly precipitation (0.1; Huffman et al. 2019), '
  '2024-01 to 2025-09, aggregated to 3-h accumulations matching the GFS f003 valid times (six 30-min files summed per 3-h '
  'window). This verification has no 12-h/24-h coverage assumption and is the primary observation-scale test at the native '
  'forecast scale.')
P('Data partitioning. The 2015-2025 record is split strictly by year: training 2015-2021, validation 2022-2023, test '
  '2024-2025 (Table 2). The validation period is used only for learning-rate scheduling and early stopping (training stops '
  'when the validation composite score or validation POD20 has not improved for 8 consecutive epochs, after a minimum number '
  'of epochs); no test-period information enters model selection. Normalization statistics are estimated on the training '
  'period only, and the validation and test sets are never oversampled.')
TAB(['Dataset', 'Period', 'Purpose', 'Samples'],
    [['Training', 'Jan 2015 - Dec 2021', 'Establish climatological mapping', '10,157'],
     ['Validation', 'Jan 2022 - Dec 2023', 'Hyperparameter and gating calibration', '2,841'],
     ['Test', 'Jan 2024 - Dec 2025', 'Two-year independent blind test', '2,839']])
NOTE('Table 2. Dataset partitioning by year for training, validation, and independent testing.')
P('Input preprocessing. The GFS precipitation channel is preprocessed before feature construction: values are clipped to '
  '[0, 100] mm, non-zero grid points are capped at the 99.9th percentile of the non-zero values, and when the field maximum '
  'is below 10 mm a light Gaussian smoothing (sigma = 0.6) is applied to reduce noise. This cleaning is applied to the GFS '
  'input only; the ERA5 target is not cleaned, and the other input channels are standardized only.')
H2('b. The Correction Network (APCNet)')
P('The precipitation correction network (APCNet) is a two-branch encoder with kinematics-oriented channel grouping: a '
  'motion/topology branch processes wind and vertical velocity fields, and a moisture/precipitation branch processes CAPE, '
  'PWAT, and precipitation; branch outputs are fused with squeeze-excitation weighting and a gated residual connection adds '
  'the correction to the input precipitation (Fig. 2). We explicitly do not claim physics-constrained learning; the architecture uses '
  'channel grouping and attention, not conservation constraints. The network is trained to predict the residual between the '
  'ERA5 target and the GFS precipitation. Training uses batch size 64, Adam optimizer, ~12 epochs, exponential moving average '
  '(EMA) of weights, intensity-weighted sampling, and three seeds (42/40/41). A residual-mask gate at evaluation applies a '
  'soft threshold on the added correction.')
P('Loss. The default loss is an asymmetric intensity-weighted L1/L2 combination with higher penalty on misses of heavy '
  'precipitation than on false alarms. Because our earlier analysis showed that the asymmetry itself induces a positive-bias '
  'artefact (over-wetting of dry areas), we also train the network with a symmetric variant in which the miss/false-alarm '
  'asymmetry is removed while intensity weighting is retained. The symmetric version is the primary reported network; '
  'asymmetric results are reported for comparison.')
FIG('fig3_architecture.png', 5.4, 'Fig. 2. Architecture of the precipitation correction network (APCNet). '
    'A kinematics-oriented encoder processes the dynamics channel group through temporal attention and atrous spatial '
    'pyramids; a thermodynamic branch extracts the last-step moist variables and modulates the decoder features through '
    'FiLM (gamma, beta); the adaptive gated fusion combines the network residual with raw GFS precipitation. Grid 25 x 37; '
    'about 1.38 million parameters.')
H2('c. Baselines')
P('BinCM (bin conditional mean): the empirical conditional-expectation benchmark E[ERA5|GFS]. The GFS precipitation range '
  'is discretized into 11 bins with edges [0, 0.05, 0.1, 0.2, 0.5, 1, 2, 5, 10, 20, 40, 100] mm; for each bin the '
  'training-period (2015-2021) mean of the ERA5 target is stored, and the test-period prediction is the bin mean of each '
  'input grid point. A per-grid-point variant (PerGrid-BinCM) fits the bin means independently at each of the 925 grid '
  'points, falling back to the global bin table where fewer than 30 training samples are available; because the extreme bins (>=20 mm) contain fewer than 30 training samples at most grid points, the per-grid-point variant effectively falls back to the global table there, so the two variants converge at extreme thresholds. BinCM is deliberately '
  'the simplest possible estimator of the learnable structure in the training reference; it has no spatial context, no loss '
  'weighting, and no free parameters besides the bin table.')
P('QM: grid-point quantile mapping (Zhu and Luo 2015; Cannon et al. 2015; Hamill and Scheuerer 2018; Jiang and Johnson 2023), calibrated on the '
  'training period (2015-2021), applied blind to the test period (2024-2025). OLS: grid-point-wise linear regression ERA5 = '
  'a*GFS + b, fitted on the training period, applied blind to the test period. U-Net: standard U-Net (hidden channels 32), '
  'same data, loss, and seeds, retrained on the corrected target with the symmetric loss (seed 42) for fair comparison. '
  'Clim: training-period grid-point climatology (weak reference, supplementary only).')
H2('d. Target Correction and Acceptance Gate (Data Transparency)')
P('Discovery and evidence. Raw-file attributes showed the mismatch: GFS A_PCP_L1_Accum_1 has time_bnds = [0,3] and '
  'cell_methods = "time: sum over hours" (a true 3-h accumulation), whereas the ERA5 monthly tp files carry GRIB_stepType = '
  'accum with GRIB_stepUnits = 1, i.e., 1-hour accumulation values stored only at 3-hourly steps (00/03/06...Z). Direct '
  'comparison over July 2024 (124 common times): domain-mean GFS 0.9064 mm/3h vs ERA5 tp 0.2984 mm/h, ratio 3.038, '
  'time-series correlation 0.936; scaling ERA5 by 3 restores ratio 1.013 and monthly totals GFS 224.8 vs ERA5 222.0 mm '
  '(+1.3%). Before correction the apparent annual basin precipitation was 235 mm yr-1 (vs GFS 671, CHM 642); after '
  'correction it is 727 mm yr-1.')
P('Correction procedure. ERA5 hourly tp was re-downloaded from the CDS and true 3-h accumulations were reconstructed by '
  'summing the three hourly values within each window (TP_3h(t) = tp(t-2h) + tp(t-1h) + tp(t)). Old monthly files were '
  'archived. The corrected target passes a six-item acceptance gate (Table S1): (V1) domain-mean GFS/ERA5 ratio 0.961 (was '
  '3.038); (V2) annual basin precipitation 727 mm yr-1 (was 235); (V3) grid-point ratio at >=20 mm/3h 1.27 (was 0.012); '
  '(V4) domain-mean time-series correlation 0.940; (M1/M2) metadata and grid consistency. Table S2 lists the superseded '
  'numbers. The erroneous target was used for all results in the earlier versions of this manuscript, including the version previously submitted to this journal; every number in the present version is recomputed from the corrected target.')
H2('e. Evaluation Protocol')
P('Deterministic: MSE and RMSE improvement relative to unprocessed GFS; spatial correlation (all-point flattening and '
  'per-sample space-averaged); categorical POD/FAR/ETS at 0.1/3/10/20 mm per 3 h; area bias; daily aggregation (12-h '
  'coverage, Section 3d). Independent verification: CHM daily rates (rate basis, caveat above) and GPM IMERG 3-h '
  'accumulations (native scale). Probabilistic: CRPS for deterministic predictions (equals MAE) and for dressed '
  'probabilistic predictions (zero-inflated Gaussian for the network; Gaussian dressing for QM); Brier score and BSS for '
  'precipitation occurrence; reliability. Significance: block bootstrap with block resampling (2000 resamples) for CHM '
  'verification improvements, accounting for temporal autocorrelation; monthly blocks (24 blocks) are the primary choice, '
  'and 60- and 90-day blocks are reported as a sensitivity check; 95% percentile intervals. Spatial correlation is computed '
  'in two ways: all-point correlation (all grid points and times pooled) and per-sample spatial correlation (the correlation '
  'of the predicted and reference fields at each time, averaged over all times with non-zero spatial variance).')
H2('f. Controlled Experiment')
P('To isolate the effect of target signal-to-noise ratio (SNR), we train the identical network on a synthetic target Y = '
  'GFS + epsilon, epsilon ~ N(0, sigma^2), for sigma in {0.5, 1.16, 2.0} mm/3h, with sigma = 1.16 equal to the observed std '
  'of the GFS-ERA5 residual in the corrected data. An identity run (target = GFS) provides the upper bound of learnability. '
  'The zero-correction baseline (output = GFS) defines the improvement metric. If the negative skill of the real task is '
  'caused by target SNR, the synthetic runs should reproduce the real-task magnitude at the matching sigma.')

# ---- 3. Results ----
H1('3. Results')
H2('a. Corrected Target and the Vanishing "Wet Bias" of GFS')
P('After target correction (Section 2d), the domain-mean GFS/ERA5 ratio over common 03/09/15/21Z times is 0.961, the annual '
  'basin precipitation is 727 mm yr-1 (vs 235 before correction; consistent with GFS 671 and CHM 642), and the number of '
  'grid points >=20 mm/3h is of the same order in both fields (ratio 1.27; 2,027 vs 2,001). The time-series correlation of '
  'domain means is 0.940. There is no systematic wet bias of GFS relative to ERA5 once the accumulation windows are aligned: '
  'the previously reported "systematic wet bias" was entirely an artefact of the mismatched target. Fig. 3 documents the '
  'correction evidence (July 2024: the raw target ratio 3.038, the corrected ratio 1.016 with correlation 0.946, and the '
  'monthly totals 112.4 vs 110.6 mm).')
FIG('fig2_target_evidence.png', 5.4, 'Fig. 3. Target-correction evidence for July 2024: (a) domain-mean GFS/ERA5 ratio '
    'before correction (3.038); (b) ratio after correction (1.016, correlation 0.946); (c) scatter of corrected ERA5 vs GFS; '
    '(d) monthly totals GFS 112.4 vs corrected ERA5 110.6 mm (+1.6%) vs old target 37.0 mm.')
P('The multi-year mean spatial bias fields (Fig. 4) show that the raw GFS is nearly unbiased against ERA5 (-0.011 mm/3h '
  'domain mean, spatial std 0.028), the corrected quantile mapping is likewise nearly unbiased (-0.021, std 0.033), whereas '
  'the DL network adds a systematic +0.347 mm/3h with strong spatial structure (std 0.196, local maxima +1.42). The '
  'network\'s correction is a spatially non-uniform wetting of the field, not a uniform rescaling toward the reference.')
FIG('fig3_bias.png', 5.4, 'Fig. 4. Multi-year mean spatial bias fields (mm/3h) of (left) raw GFS, (middle) QM-corrected, and '
    '(right) APCNet-corrected precipitation against the ERA5 reference. The DL network adds a systematic, spatially '
    'structured wet bias (+0.347 mm/3h domain mean, local maxima +1.42).')
H2('b. Main Experiment: Skill on the Training Reference')
P('Table 3 reports the 3-h MSE improvement over raw GFS on the ERA5 reference (test period 2024-2025, 2,839 samples).')
TAB(['Method', 'MSE', 'Improvement', 'All-point CC', 'Per-sample space CC'],
    [['GFS (raw)', '1.3462', '-', '0.5987', '0.470'],
     ['BinCM', '0.9791', '+27.3%', '0.6235', '0.387'],
     ['PerGrid-BinCM', '0.9781', '+27.3%', '0.6243', '0.373'],
     ['OLS', '1.0472', '+22.2%', '0.5964', '0.354'],
     ['QM', '1.1970', '+11.1%', '0.6114', '0.451'],
     ['U-Net (retrained)', '2.0885', '-55.1%', '0.6332', '0.232'],
     ['APCNet sym (S42/S40/S41)', '1.785/2.064/2.195', '-32.6/-53.3/-63.1%', '-', '0.267']])
NOTE('Table 3. Deterministic skill on the ERA5 reference (2024-2025 test period). MSE improvement relative to raw GFS; '
     'spatial correlation in two definitions (Section 2e).')
P('Simple baselines dominate the network on the training reference. The bin-based conditional-expectation benchmark '
  '(BinCM), a table of training-period bin means with no spatial context and no free parameters, achieves the largest MSE '
  'reduction (+27.3%), slightly ahead of per-grid-point linear regression (+22.2%) and quantile mapping (+11.1%); the APCNet '
  'reductions are robustly negative across seeds, and the symmetric-loss variant is the best of the network configurations '
  'but still degrades MSE. Monthly block bootstrap on the test period (24 month-blocks, 2000 resamples) confirms the '
  'significance of every contrast: BinCM 95% CI [+20.7, +33.1]% (P(improvement < 0) = 0.000), OLS [+16.1, +27.7]% '
  '(P = 0.000), QM [+2.6, +16.5]% (P = 0.003), APCNet [-54.5, -24.2]% (P(improvement > 0) = 0.000), U-Net [-79.7, -46.4]% '
  '(P = 0.000). The conditional structure of the ERA5 residual with respect to GFS is therefore strongly learnable by the '
  'simplest nonparametric estimator, and the deep network fails to capture it. Notably, the per-sample spatial correlation '
  'of every correction method is below the raw GFS (GFS 0.470; QM 0.451; OLS 0.354; BinCM 0.387; APCNet 0.267; U-Net '
  '0.232): grid-point-wise corrections and the network alike degrade the spatial structure of the field, a systematic '
  'property of point-based post-processing rather than a specific deficiency of the network.')
H2('c. Controlled Experiment: Target SNR Determines Learnable Skill')
P('Fig. 5 and Table 4 show the controlled experiment (seed 42, zero-correction baseline):')
FIG('fig4_snr.png', 4.2, 'Fig. 5. Controlled experiment: MSE improvement over the zero-correction baseline as a function of '
    'the added Gaussian target noise (sigma). The identity run (sigma = 0) learns the mapping almost perfectly (correlation '
    '0.998); skill degrades monotonically with noise and reaches -27.2% at the observed residual level (sigma = 1.16 mm/3h).')
TAB(['Target', 'sigma (mm/3h)', 'val MSE convergence', 'MSE improvement', 'r(pred, noise)'],
    [['GFS (identity)', '0', '0.0083', 'learnable (CC 0.998)', '-'],
     ['GFS + noise', '0.5', '0.26', '-2.7%', '0.0000'],
     ['GFS + noise', '1.16 (= observed residual std)', '1.71', '-27.2%', '0.0009'],
     ['GFS + noise', '2.0', '4.39', '-9.6%', '0.0007']])
NOTE('Table 4. Controlled experiment (Section 2f). MSE improvement relative to the zero-correction baseline (output = GFS).')
P('Three facts follow. (i) The architecture is not the bottleneck: with a noise-free target it learns the identity mapping '
  'almost perfectly (correlation 0.9984, mean |residual| 0.055). (ii) The network learns essentially nothing of an '
  'independent noise component (r(pred, noise) ~ 0 in all noisy runs). (iii) Despite learning nothing, it degrades MSE '
  'relative to the zero-correction baseline, most strongly at sigma = 1.16 (-27.2%, within the -32.6% to -63.1% range of '
  'the real task). The non-monotonicity is not an artifact of early stopping: re-running every noise level with a relaxed '
  'stopping criterion (patience 8, minimum 8 epochs, up to 24) leaves the results essentially unchanged (sigma = 0.5: '
  '-2.7% vs -3.2%; sigma = 1.16: -27.2% vs -28.5%; sigma = 2.0: -9.6% vs -9.5%, with the run now extending from 7 to 12 '
  'epochs), and the validation curves converge within a few epochs and remain flat thereafter. The non-monotonicity is '
  'quantitatively explained by a ratio effect. The network output is essentially a small-mean correction that is '
  'uncorrelated with the injected noise (r ~ 0), so the MSE of the corrected output against the noisy target equals '
  'var(pred) + sigma^2 and the improvement is approximately -var(pred)/sigma^2. The variance of the learned correction '
  'grows with noise and saturates (0.007 at sigma = 0.5, 0.366 at sigma = 1.16, 0.385 at sigma = 2.0), while sigma^2 keeps '
  'growing, so the ratio first deepens and then recovers: -0.007/0.25 = -2.7%, -0.366/1.35 = -27.2%, -0.385/4.00 = -9.6%, '
  'matching the observed values. The negative skill of DL correction on its training reference is thus causally tied to '
  'the target SNR: the loss-weighted network adds a structured, noise-amplified bias instead of a noise-free correction, '
  'and this bias costs more than it corrects.')
H2('d. Spatial and Daily-Scale Diagnostics')
P('The spatial bias fields (Section 3a, Fig. 4) already show that the network wets the domain non-uniformly. At the daily '
  'scale (12-h coverage; four 3-h windows summed per UTC day; 706 complete days), the conclusions are stronger (Table 5): '
  'the network degrades daily MSE by -101% (domain-mean daily rate 2.12 mm/12h vs target 0.96), while QM (+9.5%) and OLS '
  '(+11.3%) retain positive skill; the U-Net (retrained) degrades daily MSE by -111%. The daily-scale degradation of the '
  'network is the same over-wetting seen at 3-h scale, integrated over the day.')
TAB(['Method', 'MSE', 'Improvement', 'CC', 'Daily-mean rate (mm/h)', '>=10 mm bias'],
    [['GFS', '5.7205', '-', '0.7355', '0.918', '0.87'],
     ['APCNet (sym S42)', '11.506', '-101%', '0.7389', '2.12', '2.19'],
     ['QM', '5.1784', '+9.5%', '0.7465', '0.881', '0.90'],
     ['OLS', '5.0726', '+11.3%', '0.7324', '0.885', '0.45'],
     ['U-Net (retrained)', '12.0700', '-111%', '0.7561', '1.674', '2.35']])
NOTE('Table 5. Daily-scale deterministic skill (12-h coverage; four 3-h windows per UTC day; 706 complete days).')
H2('e. Independent Verification against CHM (Observation Scale, Daily)')
P('Table 6 reports the CHM daily verification (713 common days, 2024-2025, mm h-1 rate basis; model fields cover 12 h, CHM '
  '24 h, rate comparisons only):')
TAB(['Method', 'RMSE (mm/h)', 'MAE', 'CC', 'Bias (fcst-obs)', 'RMSE improvement vs GFS'],
    [['GFS', '0.2720', '0.0801', '0.4924', '+0.0055', '-'],
     ['APCNet (sym, S42)', '0.3589', '0.1507', '0.5122', '+0.1052', '-31.94%'],
     ['APCNet (sym, S40/S41)', '0.3640/-', '0.1455/-', '0.4936/-', '+0.0965/-', '-33.82%/-40.84%'],
     ['APCNet (asym, S42)', '-', '-', '-', '-', '-58.69%'],
     ['U-Net (retrained)', '0.3722', '0.1240', '0.5205', '+0.0681', '-36.84%'],
     ['QM', '0.2622', '0.0790', '0.4962', '+0.0024', '+3.58%'],
     ['OLS', '0.2389', '0.0840', '0.4869', '+0.0027', '+12.15%']])
NOTE('Table 6. CHM daily verification (rate basis, 713 common days). RMSE improvement relative to raw GFS; negative means '
     'degradation.')
P('Monthly block bootstrap (24 month-blocks, 2000 resamples): APCNet observed improvement -31.94%, 95% CI [-41.31, '
  '-26.27]%, P(improvement > 0) = 0.000. The same bootstrap applied to QM gives observed +3.58%, 95% CI [+1.47, +5.12]%, '
  'P(improvement < 0) = 0.001, and to OLS +12.15%, 95% CI [+9.80, +15.86]%, P(improvement < 0) = 0.000: both classical '
  'baselines are significantly positive at the observation scale, and the DL networks (APCNet and U-Net) are the only methods with significantly '
  'negative skill. The conclusion is insensitive to block length: with 60-day blocks (12 blocks) the APCNet CI is [-42.65, '
  '-27.23]% and with 90-day blocks (8 blocks) [-39.84, -27.78]%, P(negative) = 1.000 in both cases; QM and OLS remain '
  'significantly positive under both block lengths. The retrained U-Net is even worse at the observation scale (-36.84% RMSE, bias +0.0681 mm/h), so the negative skill is not specific to the APCNet architecture. The network more than doubles the mean absolute bias (from +0.006 to '
  '+0.105 mm/h, i.e., over-wetting by a factor of ~2.5 in domain-mean rate: 0.176 vs CHM 0.071).')
FIG('fig5_chm.png', 5.6, 'Fig. 6. CHM verification: (a) monthly mean precipitation rate of GFS, APCNet, and CHM; (b) RMSE '
    'improvement relative to GFS with 95% monthly-block bootstrap intervals. The DL networks (APCNet and U-Net) are the only methods with significantly '
    'negative skill at the observation scale (-31.9%, CI [-41.3, -26.3]); QM (+3.6%) and OLS (+12.2%) are significantly '
    'positive.')
P('Error decomposition by observed intensity (Fig. 6b, Table 7). Binning the errors by the CHM observed rate: the network '
  'improves only the heaviest bin (>=1.0 mm/h; RMSE -0.195) and degrades every other bin, with the dry/trace bin (<0.02 '
  'mm/h; 78% of samples) suffering a MAE increase of +0.016 mm/h from over-wetting. Improvement samples are 15.8% of the '
  'total, degraded samples 63.7%. The network\'s only measurable positive contribution is a reduction of large errors in '
  'heavy-precipitation events, paid for by systematic over-wetting elsewhere.')
TAB(['Observed rate bin (mm/h)', 'Samples', 'GFS RMSE', 'APCNet RMSE', 'Delta RMSE'],
    [['0-0.02', '514,427', '0.1450', '0.2533', '+0.108'],
     ['0.02-0.1', '59,517', '0.2498', '0.4268', '+0.177'],
     ['0.1-0.3', '42,682', '0.3380', '0.5203', '+0.182'],
     ['0.3-1.0', '32,538', '0.5461', '0.6794', '+0.133'],
     ['>=1.0', '10,361', '1.3783', '1.1834', '-0.195']])
NOTE('Table 7. CHM error decomposition by observed intensity bin. Delta RMSE = APCNet - GFS; the network improves only the '
     'heaviest bin and degrades all others.')
P('Interpretation. The raw GFS is nearly unbiased against CHM (+8% in rate), not under-forecasting; the network\'s learned '
  'amplification therefore moves the field in the wrong direction at the observation scale. ERA5 itself is close to CHM in '
  'climatological mean (QM-corrected domain-mean rate 0.0732 vs CHM 0.0708 mm/h, within ~3%), which is why QM transfers well; the '
  'network\'s non-uniform amplification destroys this climatological fidelity.')
H2('f. Independent Verification against GPM IMERG (Observation Scale, 3-h)')
P('Table 8 reports the GPM verification at the native 3-h scale (2,475 samples, 2024-01-15 to 2025-09-30; six 30-min IMERG '
  'files summed per 3-h window; no coverage-mismatch assumption):')
TAB(['Method', 'MSE', 'RMSE', 'MAE', 'CC', 'Bias (fcst-obs)', 'MSE improvement vs GFS'],
    [['GFS', '2.2635', '1.5045', '0.3014', '0.4682', '+0.0106', '-'],
     ['ERA5 (target)', '2.1593', '1.4695', '0.3021', '0.4733', '+0.0226', '+4.60%'],
     ['BinCM (trained on ERA5)', '1.7521', '1.3237', '0.3039', '0.4846', '-0.0075', '+22.60%'],
     ['APCNet (sym, S42)', '2.9270', '1.7108', '0.5610', '0.4619', '+0.3367', '-29.31%'],
     ['U-Net (retrained)', '3.3338', '1.8259', '0.4560', '0.4796', '+0.2243', '-47.29%'],
     ['QM', '2.0767', '1.4411', '0.2963', '0.4774', '+0.0002', '+8.25%'],
     ['OLS', '1.8193', '1.3488', '0.3150', '0.4695', '-0.0069', '+19.63%']])
NOTE('Table 8. GPM IMERG verification at the native 3-h scale (2,475 samples). MSE improvement relative to raw GFS.')
P('The GPM result confirms the CHM result at the native forecast scale and without any daily-coverage assumption: the '
  'bin-based conditional-expectation benchmark trained on ERA5 transfers almost perfectly to the observation scale (+22.6%, '
  'bias -0.008), OLS and QM improve MSE by +19.6% and +8.3% respectively, the ERA5 target itself is slightly better than '
  'GFS (+4.6%), and the network degrades MSE by -29.3%, over-wetting the domain (predicted rate 0.582 mm/h vs observed '
  '0.246, a 2.4x factor). A standard U-Net retrained under the same protocol is even worse (-47.3%), so the failure is shared by the regression-based DL family, not a quirk of APCNet. Fig. 7 summarizes the consistency across references: on the ERA5 reference and on the two '
  'independent verification products the ranking OLS > QM > GFS > APCNet holds, with BinCM ahead of all of them on the two 3-h-scale '
  'references it applies to (+27.3% on ERA5, +22.6% on GPM; CHM is daily and BinCM is not defined at that scale). The ERA5 '
  'target is itself slightly higher than GPM (+9% in rate), i.e., the reanalysis reference does not underestimate the '
  'observations in this region, and the network amplifies this small positive bias into a large one. The domain-mean '
  'time-series correlation is 0.90 for GFS/QM/OLS and 0.868 for the network.')
P('Significance under autocorrelation-aware inference. Because consecutive 3-h samples are strongly autocorrelated '
  '(a single convective system spans many windows), the contrasts in Table 8 were re-tested with block bootstrap over '
  'consecutive samples (main block length 16 windows = 48 h, 2,000 resamples; sensitivity to 8- and 32-window blocks '
  'reported in the supplement). Point estimates are unchanged (-29.3%, +8.3%, +19.6%), but the 95% intervals are wide: '
  'APCNet [-65.0, +2.9]%, QM [-18.8, +32.6]%, OLS [-6.2, +42.2]% (P(improvement < 0) = 0.962, 0.257, 0.060). On the '
  '2,475-sample GPM record, with its strong serial correlation, no contrast reaches conventional significance; the '
  'CHM daily record, with its longer independent block structure, is where the significance statements in Section 3e '
  'hold (APCNet 95% CI [-41.3, -26.3]%, P < 0.001; QM [+1.5, +5.1]%; OLS [+9.8, +15.9]%). The GPM result should '
  'therefore be read as a directional confirmation of the CHM verdict at the native forecast scale, not as an '
  'independent significance test.')
FIG('fig6_three_refs.png', 5.6, 'Fig. 7. MSE/RMSE improvement relative to raw GFS across the three references (ERA5 '
    'training reference, CHM daily observations, GPM IMERG 3-h observations). The ranking BinCM > OLS > QM > GFS > APCNet '
    'holds on every 3-h-scale reference; the DL networks (APCNet and U-Net) are the only methods with negative skill on all references.')
P('Categorical scores at 3-h scale (Table 9): at >=1 and >=3 mm/3h the network has the highest POD but also the highest '
  'FAR, and the lowest ETS; at >=10 mm/3h the network ETS (0.151) is statistically indistinguishable from GFS (0.149); at '
  '>=20 mm/3h it is significantly worse (0.063 vs 0.071; block-bootstrap delta = -0.0089, P = 0.018). The modest '
  'extreme-detection advantage seen on the ERA5 reference (Section 3g) does not transfer to observations. All categorical scores are computed on 2,475 samples x 925 grid points; the >=10 and >=20 mm/3h classes contain 11,779 and 2,478 observed grid-point events (in 476 and 225 samples), respectively. All GPM '
  'conclusions are robust to the interpolation choice: repeating the full evaluation with area-average remapping instead of '
  'bilinear interpolation changes the improvement estimates only marginally (BinCM +19.2%, OLS +16.5%, QM +8.0%, APCNet '
  '-24.2%, ERA5 target +5.2%) and leaves the ranking and the sign of every extreme-scale comparison unchanged.')
TAB(['Threshold (mm/3h)', 'POD (GFS/APCNet)', 'FAR (GFS/APCNet)', 'ETS (GFS/APCNet)'],
    [['>=1', '0.521/0.686', '0.573/0.764', '0.282/0.178'],
     ['>=3', '0.416/0.572', '0.579/0.738', '0.254/0.204'],
     ['>=10', '0.226/0.301', '0.688/0.760', '0.149/0.151'],
     ['>=20', '0.120/0.105', '0.851/0.863', '0.071/0.063']])
NOTE('Table 9. Categorical scores at 3-h scale against GPM IMERG. At >=10 and >=20 mm the network ETS is equal to or '
     'significantly worse than GFS. Observed events: 11,779 (>=10) and 2,478 (>=20) grid-point-hours.')
P('Focusing the GPM verification on the extreme samples themselves preserves the ranking. The 476 samples that contain at '
  'least one >=10 mm/3h grid point (observed domain-mean rate 1.03 mm/3h, sample-mean peak 23.2 mm/3h) and the 225 '
  'samples that contain >=20 mm/3h (observed rate 1.48, peak 32.8) give per-sample RMSE improvements over raw GFS of '
  '+9.6%/+7.5% for OLS, +2.8%/+4.5% for QM, +5.6%/+4.9% for U-Net, and -13.1%/-6.4% for APCNet at the >=10/>=20 '
  'thresholds: the network is the only method whose error grows on the extreme samples, consistent with its ETS at '
  '>=20 mm (Table 9). Its domain-mean rate on these samples is 1.81 and 2.34 mm/3h versus 1.03 and 1.48 observed (+76% '
  'and +57%), while GFS is already near the observed rate (1.01 and 1.36) and QM/OLS/U-Net under-forecast. All methods '
  'underestimate the domain peak, with APCNet closest at >=10 mm (-5.8 mm/3h) but only at the cost of the largest areal '
  'over-forecasting; its extreme-sample "skill" is therefore again an intensity-shape artefact rather than a '
  'localization improvement.')
H2('g. Extreme Precipitation')
P('The ERA5 reference test period (2,839 samples x 925 grid points) contains 8,935 and 1,402 grid-point events at >=10 and >=20 mm/3h (in 333 and 111 samples), respectively. On the ERA5 reference (Table 10, symmetric APCNet S42): the network achieves the highest ETS at >=10 and >=20 mm/3h '
  '(0.199 vs GFS 0.193; 0.104 vs 0.086) with higher POD, at the cost of a 3.1x over-wetting of the light-rain area (area '
  'bias 3.12 at >=0.1 mm). Monthly-block bootstrap on the test period shows the >=20 mm ETS advantage is significant '
  '(delta = +0.0185, P(improvement > 0) = 0.997) while the >=10 mm advantage is not (P = 0.793). This reference-relative '
  'advantage, however, does not survive observation-scale verification: against GPM IMERG the network ETS is equal to GFS at '
  '>=10 mm (delta = +0.0016, P = 0.656) and significantly worse at >=20 mm (delta = -0.0089, P(improvement > 0) = 0.018; '
  'Section 3f). The extreme-skill pattern on the ERA5 reference is thus best understood as a reference-internal artefact: '
  'the network broadens the wet area toward the ERA5 target, which improves ERA5-relative categorical scores while degrading '
  'all observation-scale measures. Event-level extreme verification at the native scale remains future work with longer '
  'records.')
TAB(['Threshold (mm/3h)', 'POD (GFS/APCNet)', 'FAR (GFS/APCNet)', 'ETS (GFS/APCNet)', 'Area bias (GFS/APCNet)'],
    [['>=0.1', '0.693/0.794', '0.292/0.745', '0.486/0.127', '0.98/3.12'],
     ['>=3', '0.497/0.683', '0.432/0.646', '0.351/0.290', '0.88/1.93'],
     ['>=10', '0.319/0.445', '0.667/0.732', '0.193/0.199', '0.96/1.66'],
     ['>=20', '0.192/0.223', '0.866/0.836', '0.086/0.104', '1.43/1.36']])
NOTE('Table 10. Categorical scores on the ERA5 reference (test period). The network\'s ETS advantage at >=20 mm is '
     'significant on the reference but does not transfer to observations (Section 3f). Events: 8,935 (>=10) and 1,402 (>=20) grid-point-hours.')
H2('h. Probabilistic Post-Processing')
P('Table 11 reports probabilistic scores on the second half of the test period (n = 1,420; symmetric APCNet S42; ERA5 '
  'reference):')
TAB(['Method', 'CRPS'],
    [['GFS (deterministic)', '0.2026'],
     ['QM (deterministic)', '0.2029'],
     ['APCNet (deterministic)', '0.4276'],
     ['QM + Gaussian dressing', '0.3377'],
     ['APCNet + ZIG (const p0)', '0.3138'],
     ['APCNet + ZIG (adaptive p0)', '0.3895']])
NOTE('Table 11. Probabilistic scores (second half of test period, n = 1,420; ERA5 reference).')
FIG7 = os.path.join(FIG_DIR, 'fig7_reliability.png')
add_figure_before(A, FIG7, width_in=5.4, caption='Fig. 8. (a) Reliability diagram for the APCNet precipitation-occurrence '
  'probability (>=0.1 mm) after adaptive ZIG calibration; point size scales with bin sample count (labels: counts), the '
  'dotted line marks the climatological wet frequency. (b) Sharpness: histogram of forecast probabilities.')
P('Deterministic CRPS (=MAE) confirms the deterministic findings: the network\'s point predictions are worse than GFS. '
  'Probabilistic dressing degrades CRPS relative to the deterministic forecasts in all cases: the estimated spread is not '
  'well calibrated, a calibration bottleneck reported for DL-based probabilistic post-processing in other settings as well '
  '(Worsnop et al. 2024). The reliability of the network\'s precipitation-occurrence probability is systematically '
  'over-forecasting: at every forecast-probability bin the observed wet frequency falls below the forecast probability, '
  'from 0.048 vs 0.065 in the lowest bin to 0.864 vs 0.962 in the highest (Fig. 8a), i.e., the network predicts wet '
  'conditions more often than they occur even after the adaptive ZIG calibration. Its sharpness is nonetheless '
  'weak-to-moderate: the forecast probability has a standard deviation of 0.21 (10th-90th percentile 0.05-0.49), but the '
  'Gaussian spread used for dressing is more than three times wider than QM\'s (3.12 vs 0.86 mm/3h), so the probabilistic '
  'forecasts are simultaneously over-forecasting and over-spread. The occurrence Brier score is 0.0864 with BSS +0.256 '
  'against the climatological frequency of 0.134, but at extreme thresholds (>=10, >=20 mm) the occurrence-probability BSS '
  'is negative (-0.49, -0.73): probabilistic post-processing does not recover extreme skill. Positive but modest '
  'probabilistic skill relative to climatology has been reported for ANN-based post-processing in other regions (Scheuerer '
  'et al. 2020); here the deterministic and probabilistic forms of the network both fail to recover extreme skill.')
P('Probabilistic verification against GPM IMERG at the native 3-h scale (2,475 samples; dressing parameters calibrated on the ERA5 reference as in Table 11, with the observations exchanged) confirms the failure at the observation scale (Table 12). Deterministic CRPS equals the MAE column of Table 8 (GFS 0.3014, QM 0.2964, APCNet 0.5610). Dressed probabilistic predictions degrade CRPS relative to the GFS deterministic forecast in every case (QM + Gaussian dressing 0.4185; APCNet + ZIG with constant p0 0.3435; APCNet + ZIG with adaptive p0 0.4792; GFS 0.3014). The occurrence Brier skill against the climatological frequency drops from +0.256 on the ERA5 reference to -0.025 against GPM, and the extreme-threshold BSS remains strongly negative (-0.37 at >=10 mm and -0.39 at >=20 mm). Reliability is again systematically over-forecasting at every bin, from 0.064 vs 0.042 in the lowest bin to 0.963 vs 0.658 in the highest: probabilistic post-processing does not recover extreme skill at the observation scale either.')
TAB(['Method (GPM IMERG, 3-h)', 'CRPS'],
    [['GFS (deterministic)', '0.3014'],
     ['QM (deterministic)', '0.2964'],
     ['APCNet (deterministic)', '0.5610'],
     ['QM + Gaussian dressing', '0.4185'],
     ['APCNet + ZIG (const p0)', '0.3435'],
     ['APCNet + ZIG (adaptive p0)', '0.4792']])
NOTE('Table 12. Probabilistic scores against GPM IMERG at the native 3-h scale (2,475 samples; dressing parameters calibrated on the ERA5 reference). Occurrence (>=0.1 mm) Brier 0.0937, BSS -0.025; >=10 mm BSS -0.367; >=20 mm BSS -0.390; reliability over-forecasts at every bin (0.963 vs 0.658 in the highest).')

NOTE('Table 12. Probabilistic scores against GPM IMERG at the native 3-h scale (2,475 samples; dressing parameters calibrated on the ERA5 reference). Occurrence (>=0.1 mm) Brier 0.0937, BSS -0.025; >=10 mm BSS -0.367; >=20 mm BSS -0.390; reliability over-forecasts at every bin (0.963 vs 0.658 in the highest).')

H2('i. Terrain Stratification and Physical-Consistency Diagnostics')
P('Because the study region is justified in the title by complex terrain, we stratified both independent verifications by '
  'elevation. SRTM 90-m elevation aggregated to the GFS grid (25x37) ranges 0-1608 m over the 764 valid land points; the '
  'valid points are split into three equal one-third quantile tiers (low <=252 m, mid 257-835 m, high 840-1608 m). The CHM '
  '713 common days and the GPM 2,475 samples are reused. Table 13 reports RMSE per tier.')
TAB(['Terrain tier / reference', 'Observed rate', 'GFS', 'QM', 'OLS', 'APCNet', 'U-Net'],
    [['CHM low (<=252 m)', '0.0821', '0.2872', '0.2780', '0.2586', '0.3760', '0.3951'],
     ['CHM mid (257-835 m)', '0.0666', '0.2348', '0.2284', '0.2082', '0.2976', '0.3404'],
     ['CHM high (840-1608 m)', '0.0599', '0.2138', '0.2057', '0.1793', '0.2768', '0.2981'],
     ['GPM low (<=252 m)', '0.2706', '1.5523', '1.4649', '1.3726', '1.7903', '1.9120'],
     ['GPM mid (257-835 m)', '0.2259', '1.3376', '1.2924', '1.2047', '1.4993', '1.6977'],
     ['GPM high (840-1608 m)', '0.1830', '1.2166', '1.1751', '1.0458', '1.3695', '1.5116']])
NOTE('Table 13. RMSE against each observation product, stratified by SRTM elevation tier on the GFS grid (764 valid land '
     'points split into equal one-third quantiles). CHM rows: daily rate, mm/h; GPM rows: 3-h accumulation, mm/3h. In every '
     'tier and on both references the DL networks have the largest RMSE; QM and OLS are the only methods with systematically '
     'lower RMSE than GFS, and the DL over-forecasting bias is largest relative to the observed rate in the high '
     '(complex-terrain) tier.')
P('Diurnal and area diagnostics confirm structural damage at the observation scale (Fig. 9). Against the GPM samples '
  'grouped by valid time, GFS and QM follow the observed diurnal cycle (domain-mean 0.21-0.28 mm/3h), whereas APCNet '
  'raises every time slot to 0.56-0.62 mm/3h and U-Net to 0.44-0.54, flattening the cycle (daytime 09Z peak 0.617 vs '
  '0.261 observed) while over-forecasting by a factor of about two at all hours. Wet-area fractions behave similarly: at '
  '>=0.1 mm/3h the observed domain fraction is 0.102, GFS 0.145 and QM 0.140, but APCNet 0.436 (4.3x); at >=10 mm/3h '
  'U-Net doubles the observed fraction (0.011 vs 0.005). A channel-level network trained on a smooth reanalysis '
  'reference therefore not only over-amplifies intensity but flattens temporal structure and inflates light-rain area: '
  'the physics-guided design does not preserve temporal or spatial structure at the observation scale.')
FIG('fig8_diurnal.png', 5.4, 'Fig. 9. Diurnal cycle of the domain-mean precipitation rate over the GPM-verification samples '
    '(2,475 samples, 2024-2025): GPM IMERG (black), raw GFS (blue), QM (green), OLS (orange), APCNet (red), U-Net '
    '(purple). The DL networks over-forecast by a factor of about two at every valid time and flatten the daytime peak.')
P('At the event scale the same structural signature is visible. Fig. 10 shows a typical NECV heavy-rain episode '
  '(peak window 2024-08-09 21Z, domain-mean GPM 5.83 mm/3h): GFS, QM and OLS reproduce the observed precipitation core '
  '(domain means 3.4-3.5), while APCNet matches the observed domain total (5.60) by inflating the areal extent of '
  'light-to-moderate precipitation rather than the core intensity, and U-Net remains below the observed total (3.11). '
  'The areal inflation is the same signature quantified in Fig. 9 and is consistent with the positive domain-mean bias '
  'of the network in the multi-year bias fields (Fig. 4).')
FIG('fig10_case.png', 5.4, 'Fig. 10. Typical NECV heavy-rain event (2024-08-09 21Z, domain 25 x 37): GPM IMERG '
    'observation and corrected forecasts at the 3-h scale. Top: 3-h domain-mean precipitation around the event. '
    'GFS, QM and OLS reproduce the observed precipitation core; APCNet matches the domain total (5.60 vs 5.83 mm/3h) '
    'by inflating the areal extent rather than the core intensity; U-Net stays below the observed total (3.11).')

# ---- 3i 补：CHM 独立验证期（2022-2023）----
P('We additionally evaluated the same trained networks over an earlier, independent CHM period '
  '(2022-2023, 712 common days, 12-h coverage, Table 14). This window is not fully out-of-sample: it '
  'served as the early-stopping and hyperparameter-selection window, so its numbers carry an optimistic '
  'selection bias and should be read as a favourable-case bound rather than an independent verdict. Even '
  'under this favourable condition the DL networks do not overtake the grid-point baselines. With '
  '60-day block bootstrap, every method has significantly positive skill over this window: APCNet -9.4% '
  'RMSE (95% CI [-11.2, -7.5]), U-Net -6.5% ([-7.8, -4.8]), QM -3.9% ([-6.4, -0.8]), and OLS the largest '
  '-12.0% ([-13.7, -9.9]); P(no improvement) < 0.01 in all cases. The DL rate bias is again the largest '
  '(-32% deficit in the domain-mean rate: 0.0441 vs 0.0648 mm/h observed, against OLS +2.6% and QM '
  '-2.9%), and the network spatial correlation degrades in both metrics: daily CC (0.520/0.528) stays '
  'below GFS (0.547) and QM (0.549), and per-sample spatial CC against the ERA5 reference is 0.228 '
  '(APCNet) and 0.174 (U-Net), below OLS (0.313), QM (0.299) and GFS (0.349). The validation-window '
  'ranking in RMSE is therefore OLS > APCNet > U-Net > QM > GFS, but in bias and spatial-correlation '
  'terms it is OLS > QM > GFS > APCNet > U-Net. Crucially, the same networks degrade to strongly '
  'negative skill (-32% to -59%) on the fully independent 2024-2025 test period (Section 3d), while OLS '
  'and QM retain their positive skill (+12.2% and +3.6%): DL skill is not stable across windows, '
  'exactly the transferability boundary developed in Section 4b.')
TAB(['Method (CHM 2022-2023, daily rate, mm/h)', 'RMSE', 'dRMSE vs GFS (95% CI, 60-d block)', 'CC', 'Bias', 'Domain-mean rate'],
    [['GFS', '0.2414', '-', '0.547', '+0.0012', '0.0660'],
     ['QM', '0.2320', '-3.9% (-6.4, -0.8)', '0.549', '-0.0019', '0.0629'],
     ['OLS', '0.2126', '-12.0% (-13.7, -9.9)', '0.544', '+0.0017', '0.0665'],
     ['APCNet', '0.2188', '-9.4% (-11.2, -7.5)', '0.520', '-0.0207', '0.0441'],
     ['U-Net', '0.2258', '-6.5% (-7.8, -4.8)', '0.528', '-0.0192', '0.0456']])
NOTE('Table 14. Independent-period CHM verification (2022-2023, 712 common days, 12-h daily coverage, '
     'domain mean over the 884 CHM-valid GFS grid points). Observed domain-mean rate 0.0648 mm/h. dRMSE '
     'intervals are 95% block-bootstrap confidence intervals (60-day blocks, 2,000 draws; 90-day blocks '
     'give intervals within 0.3 points); P(no improvement) < 0.01 for every method. Per-sample spatial '
     'CC against the ERA5 reference over the same window: GFS 0.349, QM 0.299, OLS 0.313, APCNet 0.228, '
     'U-Net 0.174. The 2022-2023 window participated in early stopping and hyperparameter selection, so '
     'these numbers are a favourable-case bound, not an out-of-sample verdict; the fully independent test '
     'period (2024-2025) is reported in Tables 6 and 8.')

# ---- 4. Discussion ----
H1('4. Discussion')
H2('a. Why a DL Correction Cannot Exceed Its Training Reference')
P('Two experiments bound the mechanism. The controlled experiment (Section 3c) gives the target-noise part: in the '
  'corrected data, the GFS-ERA5 residual has std ~1.16 mm/3h, comparable to the std of the precipitation field itself '
  '(SNR ~ 1 per point, despite an all-point correlation of 0.60). The unpredictable component of the residual is large; a '
  'loss-minimizing network that amplifies part of it increases error unless the signal-to-noise of the learned component is '
  'high, and the intensity-weighted loss biases the correction toward amplification (over-wetting), which the sigma = 2.0 '
  'arm reproduces as a ratio effect (Section 3c).')
P('The bin benchmark (Sections 3b and 3f) adds the structure part: the conditional expectation E[ERA5|GFS] is strongly '
  'learnable - a table of training-period bin means with no spatial context reduces MSE by +27.3% on the reference and '
  '+22.6% on GPM, the best of all methods. The learnable structure is therefore not absent; it is simply not captured by '
  'the network, whose residual-amplifying regression with intensity weighting trades away exactly the conditional-mean '
  'behavior that BinCM implements by construction. The failure is thus a failure of the learning procedure under a noisy '
  'reference, not an absence of learnable signal and not an architectural ceiling in the sense of capacity: a nonlinear '
  'network with eight input channels and a learned loss is outperformed by a one-dimensional bin table on the same data. '
  'This is the practical sense in which the reference sets the ceiling: with a reference this noisy, the marginal value of '
  'a flexible learned mapping over the conditional-mean table is negative, because the flexible mapping uses its capacity '
  'to fit and amplify noise.')
H2('b. Why Simple Baselines Win Everywhere, and Why the DL Correction Fails Everywhere')
P('QM and OLS implement the conditional climatology of the training reference: they remap each input value to the '
  'reference\'s conditional distribution or regression line. BinCM is the same idea in its purest form - it stores the '
  'conditional mean exactly. Three properties make these mappings robust at the observation scale. First, they are monotone '
  'value-to-value mappings, so they cannot create the large non-uniform over-wetting that a nonlinear network produces. '
  'Second, in this region the ERA5 reference is climatologically close to the observations (QM-corrected domain-mean rate 0.0732 vs CHM 0.0708 mm/h, '
  'within ~3%; ERA5 exceeds GPM by only +9% in rate), so a mapping fitted to ERA5 remains approximately '
  'correct against observations; BinCM\'s near-zero bias on GPM (-0.008 mm/3h) is the direct evidence. Third, their '
  'simplicity prevents overfitting to the reference\'s noise: the bin table has no free parameters beyond its means, OLS '
  'has two per grid point, and neither can amplify the residual the way a flexible network does. The network, by contrast, '
  'learns spatial and dynamical context but applies it through an amplification that is calibrated to the ERA5 residual, '
  'and the ERA5 residual is mostly noise plus a small positive bias; the network amplifies both, producing a 2.4-2.5x '
  'over-wetting of the domain-mean rate at the observation scale.')
P('The practical lesson for operational post-processing: evaluate on independent verification products, and always report simple '
  'baselines, a recommendation that echoes long-standing guidance in the statistical post-processing literature (Vannitsem et '
  'al. 2021; Cannon et al. 2015). In this case the DL network fails on its training reference and on both independent verification products, while QM '
  'and OLS remain the strongest choices everywhere. This pattern is not unique to deep learning: even early statistical '
  'calibration comparisons found that complex methods (BCSD, analog) could not beat a simple spatial-interpolation '
  'benchmark in deterministic skill (Voisin et al. 2010), CNN-based statistical downscaling of daily precipitation over China has likewise been reported to be outperformed by quantile mapping and regression-based correction (Sun and Lan 2021), and the relative value of ANN versus quantile mapping remains '
  'region-dependent in operational probabilistic post-processing (Ghazvinian et al. 2022). Quantile mapping continues to '
  'anchor operational systems such as the U.S. National Blend of Models (Hamill et al. 2017; Stovern et al. 2023). This is '
  'the transferability boundary of reanalysis-trained DL correction.')
P('The cross-window comparison (Table 14) sharpens the point. On the window used for early stopping and '
  'hyperparameter selection the networks do reach significantly positive skill (APCNet -9.4% RMSE, 95% '
  'CI [-11.2, -7.5]), yet even there they remain below OLS (-12.0%, [-13.7, -9.9]), and their rate '
  'deficit and below-GFS spatial correlation persist. On the fully independent test window the same '
  'networks turn strongly negative (-32% to -59%), while OLS and QM keep their positive skill. '
  'Significant positive skill in the selection window, below-baseline skill there, and strongly negative '
  'skill out-of-sample is the fingerprint of learning the reference\'s noise rather than of a robust '
  'correction.')

H2('c. The Reanalysis-Training Ceiling as a Design Principle')
P('Our results bound what can be achieved by training a DL precipitation correction on ERA5 in this region: the correction '
  'inherits the reference\'s smoothness and noise (a smoothness characteristic of reanalysis precipitation over China; Wu et '
  'al. 2024; Lei et al. 2022), cannot exceed the reference on the reference\'s own metric, is '
  'outperformed by a one-dimensional conditional-mean table, and the small climatological fidelity of the reference is '
  'destroyed by the network\'s amplification at the observation scale. Reanalysis-trained DL post-processing should '
  'therefore be viewed with caution: it provides no demonstrated benefit over grid-point baselines in this setup, and its '
  'skill ceiling is set by the reference, not by the architecture. Closing the gap to observations requires training '
  'targets closer to the truth (gauge-merged products, radar QPE, or a hybrid target), and every DL post-processing study '
  'should (i) verify targets with acceptance gates, (ii) report simple grid-point baselines including a '
  'conditional-mean/regression benchmark, since they are cheap and bound the value of the learned mapping, and (iii) '
  'evaluate against independent verification products, since skill on the training reference does not predict skill at the '
  'observation scale.')
H2('d. Limitations')
P('CHM verification covers 2 years (713 days) with a 12-h vs 24-h daily coverage mismatch, forcing rate-based comparisons; '
  'GPM verification (2,475 samples at 3-h scale) removes the mismatch but covers only 2024-01 to 2025-09. Single region, '
  'single NWP system (GFS), single family of regression architectures; transferability of the quantitative numbers to other '
  'regions and models is untested. The causal diagnosis (Section 3c), however, is architecture-independent by construction. '
  'In regions where long, high-quality observation-based training targets exist, DL post-processing has been reported to '
  'beat MOS (e.g., the U.S. west coast; Badrinath et al. 2023); our results bound the reanalysis-trained setting, not DL '
  'post-processing in general. The network family is regression-based; generative/probabilistic DL (diffusion, GAN) may '
  'behave differently (Fang et al. 2025; Xu et al. 2026), though our probabilistic dressing results suggest calibration '
  'remains a bottleneck, and against GPM the occurrence Brier skill relative to climatology turns negative (Section 3h). We do not test '
  'DL trained on observation targets (CHM/radar); the negative results here bound reanalysis-trained DL, not DL '
  'post-processing in general. Whether observation-trained DL can beat QM/OLS is an open question and a natural next step. '
  'Both CHM and GPM IMERG are observationally constrained products that share underlying gauge information with the ERA5 training reference; \"independent\" here means independent of the training target, not independent of all observations. The extreme-precipitation conclusions rest on categorical scores; event-level verification against observations at 3-h '
  'scale remains future work.')
P('A further caveat applies to the 2022-2023 CHM evaluation (Table 14): that window participated in '
  'early stopping and hyperparameter selection, so its positive DL skill (APCNet -9.4% RMSE) is a '
  'favourable-case bound; it is reported for completeness and does not substitute for the fully '
  'independent 2024-2025 test period, on which the same networks are strongly negative.')

P('Operational guidance. For 3-h GFS precipitation post-processing over this '
  'region the immediate recommendation is to deploy QM or OLS rather than reanalysis-trained DL correction: QM is the '
  'safest default (smallest bias, observation-scale skill +3.6% on CHM and +8.3% on GPM), OLS the best point estimate in '
  'RMSE where a linear fit is affordable, and BinCM the best transferable benchmark on the 3-h scale. DL correction should '
  'be considered only when (i) it is trained or fine-tuned on observation-based targets, and (ii) its skill is demonstrated '
  'on independent verification products with block-bootstrap intervals, including extreme, diurnal and terrain-stratified '
  'diagnostics; skill on the ERA5 reference alone is not sufficient evidence for operational deployment.')

# ---- 5. Conclusions ----
H1('5. Conclusions')
P('We re-examined a deep-learning precipitation correction system trained on ERA5 targets over Northeast China after '
  'discovering and correcting a fundamental target-data error (a cumulation-window mismatch that had produced a spurious 3x '
  'scale offset). All results are recomputed from the corrected, gate-verified target. Three conclusions follow.')
P('1. Data integrity is a precondition for claims. The previously reported headline skill (-46.9% MSE) was an artefact of '
  'the mismatched target; after correction, GFS has no systematic wet bias relative to ERA5, and the reported skill of the '
  'DL network on the ERA5 reference is robustly negative (-32.6% to -63.1% across seeds).')
P('2. Negative skill is causal, and it is a failure of the learning procedure, not of capacity. A controlled experiment '
  'shows the same network learns an identity mapping almost perfectly and degrades monotonically with target noise, '
  'reproducing the real-task magnitude at the observed residual level. A bin-based conditional-expectation benchmark with '
  'no free parameters and no spatial context achieves the largest improvement on the reference (+27.3%) and transfers to '
  'the observation scale (+22.6% on GPM), proving that the learnable conditional structure exists and is simply not '
  'captured by the network. The target\'s noise plus the network\'s residual-amplifying bias, not model capacity, set the '
  'practical ceiling.')
P('3. The transferability boundary is consistent across all references. At the observation scale, on CHM daily rates and '
  'GPM 3-h accumulations alike, the DL networks (APCNet and U-Net) are the only methods with significantly negative skill (-29% to -59%), while '
  'QM (+3.6%/+8.3%) and OLS (+12.2%/+19.6%) remain significantly positive. The ranking OLS > QM > GFS > APCNet holds on '
  'the training reference and on both independent verification products, with BinCM ahead of all methods on the two 3-h-scale '
  'references. Reanalysis-trained DL precipitation correction inherits the smoothness and noise of its reference: the '
  'reference\'s conditional structure is learnable by simple nonparametric estimators but not by the network, its '
  'climatological fidelity is destroyed by over-amplification at the observation scale, and simple grid-point baselines '
  'remain the strongest operational choice.')
P('We recommend that DL precipitation post-processing studies (i) verify targets and document data provenance with '
  'acceptance gates, (ii) always report simple grid-point baselines (QM, OLS), and (iii) evaluate against independent '
  'verification products, since skill on the training reference does not predict skill at the observation scale.')

# ---- 6. References ----
REFS = [
 'Badrinath, A., Delle Monache, L., Hayatbini, N., Chapman, W., Cannon, F., and M. Ralph, 2023: Improving precipitation forecasts with convolutional neural networks. Wea. Forecasting, 38, 291-306.',
  'Cannon, A.J., Sobie, S.R., and T.Q. Murdock, 2015: Bias correction of simulated precipitation by quantile mapping: How well do methods preserve relative changes in quantiles and extremes? J. Climate, 28, 6938-6959, doi:10.1175/JCLI-D-14-00754.1.',
 'Chen, Y., Huang, G., Wang, Y., Tao, W., Tian, Q., Yang, K., Zheng, J., and H. He, 2023: Improving the heavy rainfall forecasting using a weighted deep learning model. Front. Environ. Sci., 11, 1116672, doi:10.3389/fenvs.2023.1116672.',
'Ghazvinian, M., Zhang, Y., Hamill, T.M., Seo, D.-J., and N. Fernando, 2022: Improving probabilistic quantitative precipitation forecasts using short training data through artificial neural networks. J. Hydrometeor., 23, 1365-1382.',
  'Fang, Z., Zhong, Q., Chen, H., Wang, X., Zhang, Z., and H. Liang, 2025: Improving the fine structure of intense rainfall forecast by a designed generative adversarial network. Geosci. Model Dev., 18, 9723-9749, doi:10.5194/gmd-18-9723-2025.',
 'Gaudet, L.C., Sulia, K.J., Torn, R.D., and N.P. Bassill, 2024: Verification of the Global Forecast System, North American Mesoscale Forecast System, and High-Resolution Rapid Refresh model near-surface forecasts by use of the New York State Mesonet. Wea. Forecasting, 39, 369-386, doi:10.1175/WAF-D-23-0094.1.',
'Hamill, T.M., Engle, E., Myrick, D., Peroutka, M., Finan, C., and M. Scheuerer, 2017: The U.S. National Blend of Models for statistical postprocessing of probability of precipitation and deterministic precipitation amount. Mon. Wea. Rev., 145, 3441-3463.',
 'Hamill, T.M., and M. Scheuerer, 2018: Probabilistic precipitation forecast postprocessing using quantile mapping and rank-weighted best-member dressing. Mon. Wea. Rev., 146, 407-424.',
 'Harris, L., McRae, A.T.T., Chantry, M., Dueben, P.D., and T.N. Palmer, 2022: A generative deep learning approach to stochastic downscaling of precipitation forecasts. J. Adv. Model. Earth Syst., 14, e2022MS003120.',
 'Hersbach, H., Bell, B., Berrisford, P., Hirahara, S., Horanyi, A., Munoz-Sabater, J., Nicolas, J., Peubey, C., Radu, R., Schepers, D., Simmons, A., Soci, C., Abdalla, S., Abellan, X., Balsamo, G., Bechtold, P., Biavati, G., Bidlot, J., Bonavita, M., De Chiara, G., Dahlgren, P., Dee, D., Diamantakis, M., Dragani, R., Flemming, J., Forbes, R., Fuentes, M., Geer, A., Haimberger, L., Healy, S., Hogan, R.J., Holm, E., Janiskova, M., Keeley, S., Laloyaux, P., Lopez, P., Lupu, C., Radnoti, G., de Rosnay, P., Rozum, I., Vamborg, F., Villaume, S., and J.-N. Thepaut, 2020: The ERA5 global reanalysis. Quart. J. Roy. Meteor. Soc., 146, 1999-2049, doi:10.1002/qj.3803.',
 'Hess, P., and N. Boers, 2022: Deep learning for improving numerical weather prediction of heavy rainfall. J. Adv. Model. Earth Syst., 14, e2021MS002765.',
  'Hu, Y.-F., Yin, F.-K., and W.-M. Zhang, 2021: Deep learning-based precipitation bias correction approach for the Yin-He global spectral model. Meteorol. Appl., 28, e2032, doi:10.1002/met.2032.',
'Huffman, G.J., Stocker, E.F., Bolvin, D.T., Nelkin, E.J., and J. Tan, 2019: GPM IMERG Final Precipitation L3 Half Hourly 0.1 degree x 0.1 degree V06. Greenbelt, MD, Goddard Earth Sciences Data and Information Services Center (GES DISC), accessed 2025, doi:10.5067/GPM/IMERG/3B-HH/06.',
 'Jiang, Z., and F. Johnson, 2023: A new method for postprocessing numerical weather predictions using quantile mapping in the frequency domain. Mon. Wea. Rev., 151, 1909-1925.',
  'Lei, X., and Coauthors, 2022: How well does the ERA5 reanalysis capture the extreme climate events over China? Part I: Extreme precipitation. Front. Environ. Sci., 10, 921658, doi:10.3389/fenvs.2022.921658.',
'Scheuerer, M., Switanek, M.B., Worsnop, R.P., and T.M. Hamill, 2020: Using artificial neural networks for generating probabilistic subseasonal precipitation forecasts over California. Mon. Wea. Rev., 148, 3489-3506.',
 'Sha, Y., Gagne, D.J., II, West, G., and R. Stull, 2022: A hybrid analog-ensemble, convolutional-neural-network method for postprocessing precipitation forecasts. Mon. Wea. Rev., 150, 2931-2947.',
 'Shen, Y., Zhao, P., Pan, Y., and J. Yu, 2014: A high spatiotemporal gauge-satellite merged precipitation analysis over China. J. Geophys. Res. Atmos., 119, 3063-3075, doi:10.1002/2013JD020686.',
  'Shi, X., Chen, Z., Wang, H., Yeung, D.-Y., Wong, W.-K., and W.-C. Woo, 2015: Convolutional LSTM network: A machine learning approach for precipitation nowcasting. Adv. Neural Inf. Process. Syst., 28, 802-810.',
'Stovern, D.R., Hamill, T.M., and L.L. Smith, 2023: Improving National Blend of Models probabilistic precipitation forecasts using long time series of reforecasts and precipitation reanalyses. Part II: Results. Mon. Wea. Rev., 151, 1535-1550.',
 'Sun, L., and Y. Lan, 2021: Statistical downscaling of daily temperature and precipitation over China using deep learning neural models: Localization and comparison with other methods. Int. J. Climatol., 41, 1128-1147, doi:10.1002/joc.6769.',
  'Tarek, M., Brissette, F.P., and R. Arsenault, 2020: Evaluation of the ERA5 reanalysis as a potential reference dataset for hydrological modelling over North America. Hydrol. Earth Syst. Sci., 24, 2527-2544, doi:10.5194/hess-24-2527-2020.',
 'Vannitsem, S., and Coauthors, 2021: Statistical postprocessing for weather forecasts: Review, challenges, and avenues in a big data world. Bull. Amer. Meteor. Soc., 102, E681-E699, doi:10.1175/BAMS-D-19-0308.1.',
'Voisin, N., Schaake, J.C., and D.P. Lettenmaier, 2010: Calibration and downscaling methods for quantitative ensemble precipitation forecasts. Wea. Forecasting, 25, 1603-1627.',
  'Wu, G., Lv, P., Mao, Y., and K. Wang, 2024: ERA5 precipitation over China: Better relative hourly and daily distribution than absolute values. J. Climate, 37, 1581-1596, doi:10.1175/JCLI-D-23-0302.1.',
'Wang, F., Tian, D., and M. Carroll, 2023: Customized deep learning for precipitation bias correction and downscaling. Geosci. Model Dev., 16, 535-556.',
 'Worsnop, R.P., Scheuerer, M., Hamill, T.M., Smith, T.A., and J. Schlör, 2024: RUFCO: a deep learning framework to postprocess subseasonal precipitation accumulation forecasts. Artif. Intell. Earth Syst., 3, e240020.',
  'Xu, J., Dai, K., Ma, J., Zhang, Q., Chen, Y., Zhang, F., and C.-P. Ng, 2026: Postprocessing for 24-hour advanced forecasting of extreme precipitation using deep learning generative models. Wea. Forecasting, 41, 381-401, doi:10.1175/WAF-D-24-0199.1.',
 'Yang, S., Ling, F.H., Luo, J.-J., and L. Bai, 2025: Improving the seasonal forecast of summer precipitation in Southeastern China using a CycleGAN-based deep learning bias correction method. Adv. Atmos. Sci., 42, 26-35, doi:10.1007/s00376-024-4003-3.',
 'Yang, Y., Cui, X., Li, Y., Huang, L., and J. Tian, 2024: Statistical analysis of moisture sources and quantitative contribution of cold vortex rainstorms in Northeast China during the warm season. J. Hydrometeor., 25, 1027-1043, doi:10.1175/JHM-D-23-0226.1.',
'Zhang, X., Yang, Y., Chen, B., and W. Huang, 2021: Operational precipitation forecast over China using the Weather Research and Forecasting (WRF) model at a gray-zone resolution: impact of convection-permitting modeling. Wea. Forecasting, 36, 915-928.',
  'Zhu, K., Xue, M., Zhou, B., Zhao, K., Sun, Z., Fu, P., Zheng, Y., Zhang, X., and Q. Meng, 2018: Evaluation of real-time convection-permitting precipitation forecasts in China during the 2013-2014 summer season. J. Geophys. Res. Atmos., 123, 1037-1064, doi:10.1002/2017JD027445.',
'Zhu, Y., and Y. Luo, 2015: Precipitation calibration based on the frequency-matching method. Wea. Forecasting, 30, 1109-1124.',
]
# 在 References 标签后依次插入文献（addnext 保持顺序）
anchor_after = p_ref._p
for r in REFS:
    pnew = new_para_before(A, r, style=None, size=9)  # 先建段（临时挂在 author 前）
    # 移动到 References 后
    anchor_after.addnext(pnew._p)
    anchor_after = pnew._p

doc.save(TARGET)
print('SAVED OK')
