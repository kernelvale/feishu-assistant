"""客观质检指标（review_metrics）测试：确定性、论文无关。"""

from __future__ import annotations

from feishu_assistant.modules.review_metrics import compute_metrics, format_metrics


def test_latex_rough_counted():
    m = compute_metrics("公式 $R(x,y)$ 和 \\boxed{...} 与正常文本 R(x,y)")
    assert m["latex_rough"] == 2


def test_para_walls_and_long_sentences():
    wall = "\n".join(["行"] * 6)
    text = f"{wall}\n\n短段。\n\n{'x' * 70}。短句。"
    m = compute_metrics(text)
    assert m["para_walls"] == 1
    assert m["long_sentence_ratio"] > 0


def test_figure_note_meaning_ratio():
    text = (
        "【要回答的问题】训练是否收敛？【先说结论】会收敛。"
        "【关键比较与数据】先比较 1 轮和 10 轮。\n\n"
        "【要回答的问题】奖励来自哪里？【先说结论】来自校验器。"
        "【关键比较与数据】先比较 8 个维度。"
        "【论证作用】它回答了奖励信号是否可靠的问题。"
    )
    m = compute_metrics(text)
    assert m["figure_note_total"] == 2
    assert m["figure_note_with_meaning"] == "1/2"


def test_en_common_words_blacklist():
    m = compute_metrics("与 ground truth 比对，checker 判 pass/fail。真值校验器。")
    assert m["en_common_words"] == 3


def test_subsection_stats_do_not_assume_fixed_layer_contract():
    text = (
        "## 方法为什么成立\n### 状态筛选机制\n一段较长的机制说明。\n" * 3
        + "### 证据边界\n只有一些数字。\n### Figure 1\n图表说明。"
    )
    m = compute_metrics(text)
    assert len(m["subsections"]) == 4
    assert all(not item["title"].startswith("Figure") for item in m["subsections"])
    assert "overview_contract" not in m and "learning_loops" not in m


def test_format_metrics_renders():
    out = format_metrics(compute_metrics("### 推导边界\n" + "字" * 1000))
    assert "不能据此判定" in out and "推导边界" in out
