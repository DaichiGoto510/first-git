"""
イベントベース精度検証モジュール

アプローチ (公開データのみ):
    過去の台風・強風イベントを使ったプロキシ検証

    1. イベント前 (pre-window) の衛星データからリスクスコアを計算
    2. イベント後 (post-window) の NDVI 変化量を「疑似正解」として取得
    3. リスクスコア vs NDVI 降下量の相関・ROC/AUC を計算

根拠:
    - 台風通過後の NDVI 急落は倒木・枝折れ・葉の吹き飛ばしを示す
    - Rüetschi et al. (2019) が SAR で同様の代替検証を実施
    - NDVI 変化閾値 0.05 は人工林被害検出の実用閾値 (Immitzer 2016)

限界:
    - NDVI 降下の原因は倒木以外 (落葉・雲影・季節変化) も含む
    - 台風直後は雲の影響で画像品質が低下する場合がある
    - 完全な精度評価には現地点検記録との照合が必要

推奨イベント例:
    - 台風 Hagibis (令和元年台風19号): 2019-10-12 (関東・東北)
    - 台風 Jebi  (平成30年台風21号): 2018-09-04 (近畿・東海)
    - 台風 Faxai (令和元年台風15号): 2019-09-09 (関東)
"""

import os
import math
import base64
from datetime import datetime, timedelta
from typing import Dict, List, Tuple, Optional


# ─── 統計関数 (外部依存なし) ──────────────────────────────────

def _rank_list(values: List[float]) -> List[float]:
    """同値に平均順位を与えるランクリストを返す"""
    n = len(values)
    indexed = sorted(enumerate(values), key=lambda x: x[1])
    ranks = [0.0] * n
    i = 0
    while i < n:
        j = i
        while j < n - 1 and indexed[j + 1][1] == indexed[j][1]:
            j += 1
        avg_rank = (i + j) / 2.0 + 1.0
        for k in range(i, j + 1):
            ranks[indexed[k][0]] = avg_rank
        i = j + 1
    return ranks


def spearman_correlation(x: List[float], y: List[float]) -> float:
    """スピアマン順位相関係数を計算する"""
    if len(x) < 3:
        return 0.0
    rx, ry = _rank_list(x), _rank_list(y)
    n = len(x)
    mx, my = sum(rx) / n, sum(ry) / n
    num = sum((rx[i] - mx) * (ry[i] - my) for i in range(n))
    den = math.sqrt(
        sum((r - mx) ** 2 for r in rx) * sum((r - my) ** 2 for r in ry)
    )
    return num / den if den > 0 else 0.0


def pearson_correlation(x: List[float], y: List[float]) -> float:
    """ピアソン積率相関係数を計算する"""
    n = len(x)
    if n < 3:
        return 0.0
    mx, my = sum(x) / n, sum(y) / n
    num = sum((x[i] - mx) * (y[i] - my) for i in range(n))
    den = math.sqrt(
        sum((xi - mx) ** 2 for xi in x) * sum((yi - my) ** 2 for yi in y)
    )
    return num / den if den > 0 else 0.0


def compute_roc_auc(
    y_true: List[int], y_score: List[float]
) -> Tuple[float, List[float], List[float]]:
    """
    ROC 曲線と AUC を計算する (sklearn 不要の手動実装)

    Returns:
        (auc, fpr_list, tpr_list)
    """
    n_pos = sum(y_true)
    n_neg = len(y_true) - n_pos
    if n_pos == 0 or n_neg == 0:
        return 0.5, [0.0, 1.0], [0.0, 1.0]

    pairs = sorted(zip(y_score, y_true), reverse=True)
    fpr_list, tpr_list = [0.0], [0.0]
    tp, fp = 0, 0
    prev_score = None
    for score, label in pairs:
        if score != prev_score and prev_score is not None:
            fpr_list.append(fp / n_neg)
            tpr_list.append(tp / n_pos)
        if label == 1:
            tp += 1
        else:
            fp += 1
        prev_score = score
    fpr_list.append(1.0)
    tpr_list.append(1.0)

    # 台形則で AUC 計算
    auc = sum(
        (fpr_list[i + 1] - fpr_list[i]) * (tpr_list[i + 1] + tpr_list[i]) / 2.0
        for i in range(len(fpr_list) - 1)
    )
    return auc, fpr_list, tpr_list


def find_optimal_threshold(
    y_true: List[int], y_score: List[float]
) -> Tuple[float, float, float, float]:
    """
    Youden の J 統計量が最大となる最適閾値と精度指標を返す

    Returns:
        (threshold, precision, recall, f1)
    """
    n_pos = sum(y_true)
    n_neg = len(y_true) - n_pos
    best_j, best_thresh = -1.0, 50.0

    for thresh in sorted(set(y_score)):
        tp = sum(1 for s, y in zip(y_score, y_true) if s >= thresh and y == 1)
        fp = sum(1 for s, y in zip(y_score, y_true) if s >= thresh and y == 0)
        tpr = tp / n_pos if n_pos > 0 else 0.0
        fpr = fp / n_neg if n_neg > 0 else 0.0
        j = tpr - fpr
        if j > best_j:
            best_j, best_thresh = j, thresh

    tp = sum(1 for s, y in zip(y_score, y_true) if s >= best_thresh and y == 1)
    fp = sum(1 for s, y in zip(y_score, y_true) if s >= best_thresh and y == 0)
    fn = sum(1 for s, y in zip(y_score, y_true) if s < best_thresh and y == 1)
    precision = tp / (tp + fp) if (tp + fp) > 0 else 0.0
    recall = tp / (tp + fn) if (tp + fn) > 0 else 0.0
    f1 = (
        2 * precision * recall / (precision + recall)
        if (precision + recall) > 0
        else 0.0
    )
    return best_thresh, precision, recall, f1


# ─── 検証指標計算 ──────────────────────────────────────────────

def compute_validation_metrics(
    risk_scores: List[float],
    ndvi_drops: List[float],
    damage_threshold: float = 0.05,
) -> Dict:
    """
    リスクスコア vs NDVI 降下量の検証指標を計算する

    Args:
        risk_scores: モデルのリスクスコア (0-100)
        ndvi_drops: Δ NDVI (正値 = 降下 = 被害の疑い)
        damage_threshold: Δ NDVI がこの値以上で「被害あり」と判定

    Returns:
        spearman_rho, pearson_r, auc, precision, recall, f1_score 等を含む辞書
    """
    n = len(risk_scores)
    if n < 5:
        return {"error": f"サンプル数不足 (n={n}, 最低5必要)"}

    rho = spearman_correlation(risk_scores, ndvi_drops)
    r = pearson_correlation(risk_scores, ndvi_drops)

    y_true = [1 if d >= damage_threshold else 0 for d in ndvi_drops]
    n_damaged = sum(y_true)

    auc, fpr_list, tpr_list = compute_roc_auc(y_true, risk_scores)
    best_thresh, precision, recall, f1 = find_optimal_threshold(y_true, risk_scores)

    return {
        "n_segments": n,
        "n_damaged": n_damaged,
        "n_undamaged": n - n_damaged,
        "damage_rate_pct": round(n_damaged / n * 100, 1),
        "spearman_rho": round(rho, 3),
        "pearson_r": round(r, 3),
        "auc": round(auc, 3),
        "optimal_threshold": round(best_thresh, 1),
        "precision": round(precision, 3),
        "recall": round(recall, 3),
        "f1_score": round(f1, 3),
        "mean_risk_score": round(sum(risk_scores) / n, 1),
        "mean_ndvi_drop": round(sum(ndvi_drops) / n, 4),
        "damage_threshold": damage_threshold,
        "fpr_list": fpr_list,
        "tpr_list": tpr_list,
    }


# ─── 図の生成 ──────────────────────────────────────────────────

def plot_validation_figures(
    risk_scores: List[float],
    ndvi_drops: List[float],
    metrics: Dict,
    event_name: str,
    output_dir: str,
    base_name: str,
) -> Tuple[Optional[str], Optional[str]]:
    """
    散布図と ROC 曲線を PNG で保存する

    Returns:
        (scatter_path, roc_path) ─ 生成できなかった場合は None
    """
    try:
        import matplotlib
        matplotlib.use("Agg")
        import matplotlib.pyplot as plt
        from matplotlib.patches import Patch
    except ImportError:
        print("  警告: matplotlib が利用不可。グラフ生成をスキップします。")
        return None, None

    damage_threshold = metrics.get("damage_threshold", 0.05)
    rho = metrics.get("spearman_rho", 0)
    r = metrics.get("pearson_r", 0)
    auc = metrics.get("auc", 0)
    fpr_list = metrics.get("fpr_list", [0, 1])
    tpr_list = metrics.get("tpr_list", [0, 1])

    # ── 散布図 ─────────────────────────────────────────────────
    fig1, ax1 = plt.subplots(figsize=(7, 5))
    colors = [
        "#e74c3c" if d >= damage_threshold else "#3498db" for d in ndvi_drops
    ]
    ax1.scatter(
        risk_scores, ndvi_drops,
        c=colors, alpha=0.65, edgecolors="white", s=60, linewidths=0.5,
    )

    # 回帰直線
    n = len(risk_scores)
    mx, my = sum(risk_scores) / n, sum(ndvi_drops) / n
    num = sum((risk_scores[i] - mx) * (ndvi_drops[i] - my) for i in range(n))
    den = sum((x - mx) ** 2 for x in risk_scores)
    if den > 0:
        slope = num / den
        intercept = my - slope * mx
        x_line = [min(risk_scores), max(risk_scores)]
        y_line = [slope * x + intercept for x in x_line]
        ax1.plot(x_line, y_line, "k--", linewidth=1.2, alpha=0.7)

    ax1.axhline(
        y=damage_threshold, color="#e74c3c", linestyle=":", linewidth=1.0,
        alpha=0.8, label=f"被害閾値 Δ NDVI={damage_threshold}",
    )
    ax1.set_xlabel("リスクスコア (0-100)", fontsize=11)
    ax1.set_ylabel("NDVI 降下量 (Δ NDVI)", fontsize=11)
    ax1.set_title(f"{event_name}\nリスクスコア vs NDVI 降下量", fontsize=12)
    ax1.text(
        0.05, 0.95, f"Spearman ρ = {rho:.3f}\nPearson r = {r:.3f}",
        transform=ax1.transAxes, fontsize=10, va="top",
        bbox=dict(boxstyle="round", facecolor="lightyellow", alpha=0.85),
    )
    legend_elements = [
        Patch(facecolor="#e74c3c", alpha=0.65, label="被害あり (Δ NDVI ≥ 閾値)"),
        Patch(facecolor="#3498db", alpha=0.65, label="被害なし"),
    ]
    ax1.legend(handles=legend_elements, loc="lower right", fontsize=9)
    ax1.grid(True, alpha=0.3)
    plt.tight_layout()
    scatter_path = os.path.join(output_dir, f"{base_name}_scatter.png")
    plt.savefig(scatter_path, dpi=120, bbox_inches="tight")
    plt.close(fig1)

    # ── ROC 曲線 ───────────────────────────────────────────────
    fig2, ax2 = plt.subplots(figsize=(6, 5))
    ax2.plot(
        fpr_list, tpr_list, color="#2980b9", linewidth=2.0,
        label=f"ROC 曲線 (AUC = {auc:.3f})",
    )
    ax2.plot([0, 1], [0, 1], "k--", linewidth=1.0, alpha=0.5, label="ランダム (AUC = 0.500)")
    ax2.fill_between(fpr_list, tpr_list, alpha=0.1, color="#2980b9")
    ax2.set_xlabel("偽陽性率 (FPR)", fontsize=11)
    ax2.set_ylabel("真陽性率 (TPR)", fontsize=11)
    ax2.set_title(f"{event_name}\nROC 曲線", fontsize=12)
    ax2.legend(loc="lower right", fontsize=10)
    ax2.set_xlim([0, 1])
    ax2.set_ylim([0, 1.02])
    ax2.grid(True, alpha=0.3)
    plt.tight_layout()
    roc_path = os.path.join(output_dir, f"{base_name}_roc.png")
    plt.savefig(roc_path, dpi=120, bbox_inches="tight")
    plt.close(fig2)

    return scatter_path, roc_path


# ─── HTML レポート ─────────────────────────────────────────────

def _img_to_base64(path: Optional[str]) -> str:
    """PNG をインライン base64 データ URI に変換する"""
    if not path or not os.path.exists(path):
        return ""
    with open(path, "rb") as f:
        data = base64.b64encode(f.read()).decode("ascii")
    return f"data:image/png;base64,{data}"


def generate_html_report(
    segment_results: List[Dict],
    metrics: Dict,
    event_name: str,
    event_date: str,
    scatter_path: Optional[str],
    roc_path: Optional[str],
    output_path: str,
):
    """精度検証 HTML レポートを生成する"""
    scatter_uri = _img_to_base64(scatter_path)
    roc_uri = _img_to_base64(roc_path)

    auc_val = metrics.get("auc", 0.5)
    rho_val = metrics.get("spearman_rho", 0.0)
    n_seg = metrics.get("n_segments", 0)
    n_dmg = metrics.get("n_damaged", 0)
    dmg_rate = metrics.get("damage_rate_pct", 0)
    precision = metrics.get("precision", 0)
    recall = metrics.get("recall", 0)
    f1 = metrics.get("f1_score", 0)
    opt_thresh = metrics.get("optimal_threshold", 50)
    damage_threshold = metrics.get("damage_threshold", 0.05)

    def _color(val):
        if val >= 0.70:
            return "#27ae60"
        elif val >= 0.55:
            return "#f39c12"
        return "#e74c3c"

    def _rho_label(rho):
        if abs(rho) >= 0.5:
            return "強い相関"
        elif abs(rho) >= 0.3:
            return "中程度の相関"
        elif abs(rho) >= 0.1:
            return "弱い相関"
        return "相関なし"

    rows = ""
    for seg in sorted(segment_results, key=lambda x: x.get("risk_score", 0), reverse=True):
        nd = seg.get("ndvi_drop", 0.0)
        rs = seg.get("risk_score", 0.0)
        damaged = nd >= damage_threshold
        predicted = rs >= opt_thresh
        match_icon = "✓" if damaged == predicted else "✗"
        row_bg = "#fff5f5" if damaged else "#f5fff5"
        rows += (
            f'<tr style="background:{row_bg}">'
            f"<td>{seg.get('segment_id', '')}</td>"
            f"<td>{rs:.1f}</td>"
            f"<td>{nd:.4f}</td>"
            f"<td>{'🔴 あり' if damaged else '🟢 なし'}</td>"
            f"<td>{'あり' if predicted else 'なし'}</td>"
            f'<td style="text-align:center;font-size:1.1em">{match_icon}</td>'
            "</tr>"
        )

    scatter_html = (
        f'<img src="{scatter_uri}" style="max-width:100%;border-radius:8px">'
        if scatter_uri else "<p style='color:#999'>散布図なし</p>"
    )
    roc_html = (
        f'<img src="{roc_uri}" style="max-width:100%;border-radius:8px">'
        if roc_uri else "<p style='color:#999'>ROC 曲線なし</p>"
    )

    html = f"""<!DOCTYPE html>
<html lang="ja">
<head>
<meta charset="UTF-8">
<meta name="viewport" content="width=device-width, initial-scale=1.0">
<title>精度検証レポート - {event_name}</title>
<style>
  body {{ font-family: 'Helvetica Neue', Arial, sans-serif; margin: 0; background: #f8f9fa; color: #333; }}
  .header {{ background: linear-gradient(135deg, #2c3e50, #3498db); color: white; padding: 24px 32px; }}
  .header h1 {{ margin: 0; font-size: 1.6em; }}
  .header p {{ margin: 6px 0 0; opacity: 0.85; font-size: 0.92em; }}
  .container {{ max-width: 1100px; margin: 0 auto; padding: 24px 20px; }}
  .metric-grid {{ display: grid; grid-template-columns: repeat(auto-fit, minmax(150px, 1fr)); gap: 14px; margin-bottom: 28px; }}
  .metric-card {{ background: white; border-radius: 10px; padding: 16px 18px; box-shadow: 0 2px 8px rgba(0,0,0,.08); text-align: center; }}
  .metric-card .value {{ font-size: 2em; font-weight: bold; margin: 4px 0; }}
  .metric-card .label {{ font-size: 0.78em; color: #666; }}
  .section {{ background: white; border-radius: 10px; padding: 20px 24px; margin-bottom: 24px; box-shadow: 0 2px 8px rgba(0,0,0,.08); }}
  .section h2 {{ margin-top: 0; font-size: 1.1em; border-bottom: 2px solid #3498db; padding-bottom: 8px; color: #2c3e50; }}
  .fig-grid {{ display: grid; grid-template-columns: 1fr 1fr; gap: 20px; }}
  @media (max-width: 700px) {{ .fig-grid {{ grid-template-columns: 1fr; }} }}
  table {{ width: 100%; border-collapse: collapse; font-size: 0.88em; }}
  th {{ background: #2c3e50; color: white; padding: 8px 12px; text-align: left; }}
  td {{ padding: 7px 12px; border-bottom: 1px solid #eee; }}
  .note {{ background: #fffde7; border-left: 4px solid #f39c12; padding: 12px 16px; border-radius: 4px; font-size: 0.88em; line-height: 1.7; }}
</style>
</head>
<body>
<div class="header">
  <h1>🛤️ 倒木リスクモデル 精度検証レポート</h1>
  <p>イベント: {event_name} (通過日: {event_date}) &nbsp;|&nbsp; 生成日時: {datetime.now().strftime("%Y-%m-%d %H:%M")}</p>
</div>
<div class="container">

  <div class="metric-grid">
    <div class="metric-card">
      <div class="label">検証セグメント数</div>
      <div class="value">{n_seg}</div>
    </div>
    <div class="metric-card">
      <div class="label">被害セグメント数</div>
      <div class="value" style="color:#e74c3c">{n_dmg}</div>
      <div class="label">被害率 {dmg_rate:.1f}%</div>
    </div>
    <div class="metric-card">
      <div class="label">Spearman ρ</div>
      <div class="value" style="color:{_color(abs(rho_val) * 0.6 + 0.4)}">{rho_val:.3f}</div>
      <div class="label">{_rho_label(rho_val)}</div>
    </div>
    <div class="metric-card">
      <div class="label">AUC</div>
      <div class="value" style="color:{_color(auc_val)}">{auc_val:.3f}</div>
      <div class="label">{'優秀' if auc_val >= 0.75 else '良好' if auc_val >= 0.60 else '要改善'}</div>
    </div>
    <div class="metric-card">
      <div class="label">Precision</div>
      <div class="value">{precision:.3f}</div>
    </div>
    <div class="metric-card">
      <div class="label">Recall</div>
      <div class="value">{recall:.3f}</div>
    </div>
    <div class="metric-card">
      <div class="label">F1 スコア</div>
      <div class="value" style="color:{_color(f1)}">{f1:.3f}</div>
    </div>
    <div class="metric-card">
      <div class="label">最適リスク閾値</div>
      <div class="value">{opt_thresh:.0f}</div>
      <div class="label">/ 100点</div>
    </div>
  </div>

  <div class="section">
    <h2>📊 検証グラフ</h2>
    <div class="fig-grid">
      <div>{scatter_html}</div>
      <div>{roc_html}</div>
    </div>
  </div>

  <div class="section">
    <h2>📋 セグメント別結果</h2>
    <p style="font-size:0.85em;color:#666">
      被害判定閾値: Δ NDVI ≥ {damage_threshold:.3f} &nbsp;|&nbsp;
      予測閾値 (Youden 最適): リスクスコア ≥ {opt_thresh:.0f}
    </p>
    <div style="overflow-x:auto">
    <table>
      <thead><tr>
        <th>セグメントID</th><th>リスクスコア</th><th>Δ NDVI</th>
        <th>実際の被害</th><th>モデル予測</th><th>一致</th>
      </tr></thead>
      <tbody>{rows}</tbody>
    </table>
    </div>
  </div>

  <div class="section">
    <h2>⚠️ 検証手法の注意事項</h2>
    <div class="note">
      <strong>プロキシ検証 (代替検証) の限界:</strong><br>
      ① NDVI 降下は倒木以外の原因 (落葉・豪雨時の雲影・季節変化) も含みます。<br>
      ② 台風直後は雲の影響で Sentinel-2 の画像品質が低下する場合があります。<br>
      ③ 完全な精度評価には現地点検記録との照合が別途必要です。<br><br>
      <strong>解釈の目安:</strong><br>
      AUC &gt; 0.75: 実用水準 (優秀) &nbsp;|&nbsp; AUC &gt; 0.60: 有効なモデル &nbsp;|&nbsp; AUC = 0.50: ランダム予測と同等<br>
      Spearman ρ &gt; 0.40: 中〜強の正の相関 (モデルが被害パターンを捉えている)
    </div>
  </div>

</div>
</body>
</html>"""

    with open(output_path, "w", encoding="utf-8") as f:
        f.write(html)
    print(f"  → HTML レポート: {output_path}")


# ─── GEE データ取得 ────────────────────────────────────────────

def fetch_ndvi_for_period(
    roi_geojson: dict,
    start_date: str,
    end_date: str,
    cloud_threshold: int = 20,
) -> Optional[float]:
    """
    指定期間の NDVI 中央値を GEE から取得する

    Returns:
        NDVI 平均値。データなしの場合は None。
    """
    import ee

    roi = ee.Geometry(roi_geojson)
    s2 = (
        ee.ImageCollection("COPERNICUS/S2_SR_HARMONIZED")
        .filterDate(start_date, end_date)
        .filterBounds(roi)
        .filter(ee.Filter.lt("CLOUDY_PIXEL_PERCENTAGE", cloud_threshold))
    )
    if s2.size().getInfo() == 0:
        # 雲量閾値を緩和してリトライ
        s2 = (
            ee.ImageCollection("COPERNICUS/S2_SR_HARMONIZED")
            .filterDate(start_date, end_date)
            .filterBounds(roi)
            .filter(ee.Filter.lt("CLOUDY_PIXEL_PERCENTAGE", 70))
        )
        if s2.size().getInfo() == 0:
            return None

    ndvi = s2.map(
        lambda img: img.normalizedDifference(["B8", "B4"]).rename("NDVI")
    ).median()

    result = ndvi.reduceRegion(
        reducer=ee.Reducer.mean(), geometry=roi, scale=20, maxPixels=1e8
    ).getInfo()
    val = result.get("NDVI")
    return float(val) if val is not None else None
