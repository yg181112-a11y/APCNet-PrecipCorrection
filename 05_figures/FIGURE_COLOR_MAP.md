# Figure Color Map — Manuscript_R3_WAF (scale-dependent skill across ERA5, CHM, and GPM)

统一"方法—颜色"映射（v4 审稿意见 N3 落实）。全部 14 幅图在最终渲染中遵守下表；灰色阶（GFS）在灰阶输出下保持可分辨。

| Method / Reference | Color (hex) | Symbol / Line | Notes |
|---|---|---|---|
| GFS (raw forecast) | `#333333` (深灰) | solid / `o` (time series), dashed in maps when 0-baseline | 灰阶安全；Taylor 图中与黑色参考星区分（星形 `*`） |
| QM | `#E69F00` (橙) | solid / `s` | |
| OLS | `#0072B2` (蓝) | solid / `^` | |
| BinCM | `#009E73` (绿) | solid / `D` | |
| APCNet | `#D55E00` (朱红) | solid / `v` | |
| U-Net | `#CC79A7` (粉) | solid / `P` | |
| ERA5 reference (Taylor 图) | black `k*` | `*` | 仅 Fig. 14 |
| GPM / CHM observations | black solid (实线+实心圆/方块) | — | 观测类一律黑色系 |

## 面板/轴辅助色（非方法）

- 偏差场 (Fig. 5)：`RdBu_r` 色标（语义 = 偏差值，非方法）；散点 `#8C8C8C` 灰、OLS fit 线 `#0072B2`（与 OLS 方法色一致）。
- 可靠性图 (Fig. 10)：APCNet `#D55E00` 曲线；锐度柱 `#0072B2`（语义 = 样本占比，非 OLS 方法）。
- 受控实验 (Fig. 6)：复合损失 (S42) `#D55E00`；纯 MSE (S42) `#0072B2` 虚线（实验臂，正文 Table 4 对应）。
- 1:1 / 参考线：黑色虚线 `k--`。

## 已落图清单（本轮 v4 图表阶段）

- Fig. 2 架构图：新 arc3 曲线 skip（16:31 版）
- Fig. 4/7（draw_fig4_fig6_run13.py）、Fig. 8（draw_fig7_ranking_run13.py）— 前轮已按映射重渲
- Fig. 5：OLS fit 蓝线 + 灰散点（16:48 版）
- Fig. 6（fix_fig6.py，11:20 版）：中性 suptitle + 纯 MSE 对照
- Fig. 9（make_fig_case.py，16:37 版）：时间序列图例 GFS `#333333` / QM `#E69F00` / OLS `#0072B2`
- Fig. 10（draw_fig10_reliability.py，11:56 版）：图例/N=2,289,375/1:1/刻度统一
- Fig. 11（draw_fig9_diurnal_run13.py，16:36 版）
- Fig. 12（draw_multi_lead_overview.py，16:34 版）
- Fig. 13（draw_qq_season.py，16:37 版）
- Fig. 14（_draw_fig11_taylor.py，16:47 版）：GFS 深灰/QM 橙/OLS 蓝/BinCM 绿/APCNet 朱红/U-Net 粉

## 验收状态

- docx 内嵌 14 图已全部替换为投稿目录最新 PNG 并逐张字节级验证匹配（replace_all_figs.py，2026-10-06 16:49）。
- 全部 14 图分辨率 ≥ 300 dpi（AMS 要求）。
