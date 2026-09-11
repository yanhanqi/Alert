# 固定 contamination 校准

`contamination` 是孤立森林用于确定高优先级分数阈值的异常比例。它不是攻击比例，也不是一个固定的异常分数。

运行校准：

```bash
python -m alert_priority.calibrate_contamination
```

默认划分如下：

```text
开发集：shaw、wardbeck、wheeler、wilson
验证集：harrison、santos
测试集：fox、russellmitchell
```

校准器会在开发集上测试：`0.005、0.01、0.02、0.03、0.05、0.08、0.10`，按合并组的宏平均 MCC 选择一个值。标签只用于生成离线真值和比较候选参数，不能进入特征或孤立森林拟合。

当前开发集选择结果为 `contamination=0.005`。这意味着每次窗口或场景重新拟合孤立森林时，使用其异常分数分布的最高约 0.5% 作为高优先级候选；由于窗口样本量、特征分布和分数分布不同，实际分数阈值会变化。

使用固定值运行一个场景：

```bash
python -m alert_priority.main \
  --csv datasets/alerts_csv/fox_alerts.txt \
  --scenario fox \
  --contamination 0.005
```

部署代码调用：

```python
result = prioritize_alerts(window_alerts, contamination=0.005)
```

部署时不需要知道攻击组数。高优先级只表示告警相对异常，攻击确认由后续关联、召回或人工调查完成。

`results/contamination_calibration.json` 保存候选值、开发集选择过程、验证集和测试集结果；`results/aitads_csv/fixed-0.005/` 保存 8 个场景的固定值实验日志。`results/` 为本地结果目录，不提交到 Git。
