# Collision / near-miss model evaluation
Trained on `data\sim\trajectories.parquet`, 1142936 train rows, 411519 held-out-range rows, 125620 random-val rows.
Features used (metric simulator only; `overlap_iou` from the full F1 spec needs real pixel boxes and is added at runtime integration, not here).

## Held-out parameter range (B_speed>5)

### LightGBM
```
              precision    recall  f1-score   support

      NORMAL       0.98      0.81      0.89    388353
   NEAR_MISS       0.08      0.35      0.14      9903
   COLLISION       0.21      0.79      0.33     13263

    accuracy                           0.80    411519
   macro avg       0.42      0.65      0.45    411519
weighted avg       0.93      0.80      0.85    411519

```
Confusion matrix (rows=true, cols=pred, order NORMAL/NEAR_MISS/COLLISION):
```
[[314075  36914  37364]
 [  4061   3515   2327]
 [  1892    945  10426]]
```
Mean lead time before t_event: 0.80 s

### Rule (ported to metres)
```
              precision    recall  f1-score   support

      NORMAL       0.95      1.00      0.97    388353
   NEAR_MISS       0.31      0.10      0.16      9903
   COLLISION       0.17      0.01      0.01     13263

    accuracy                           0.94    411519
   macro avg       0.48      0.37      0.38    411519
weighted avg       0.91      0.94      0.92    411519

```
Confusion matrix (rows=true, cols=pred, order NORMAL/NEAR_MISS/COLLISION):
```
[[386755   1513     85]
 [  8570   1030    303]
 [ 12427    757     79]]
```
Mean lead time before t_event: -0.21 s

## Random validation split

### LightGBM
```
              precision    recall  f1-score   support

      NORMAL       0.99      0.80      0.88    119620
   NEAR_MISS       0.08      0.45      0.13      2011
   COLLISION       0.19      0.85      0.31      3989

    accuracy                           0.79    125620
   macro avg       0.42      0.70      0.44    125620
weighted avg       0.95      0.79      0.85    125620

```
Confusion matrix (rows=true, cols=pred, order NORMAL/NEAR_MISS/COLLISION):
```
[[95301 10394 13925]
 [  528   906   577]
 [  182   436  3371]]
```
Mean lead time before t_event: 1.25 s

### Rule (ported to metres)
```
              precision    recall  f1-score   support

      NORMAL       0.96      1.00      0.98    119620
   NEAR_MISS       0.32      0.16      0.22      2011
   COLLISION       0.21      0.01      0.02      3989

    accuracy                           0.95    125620
   macro avg       0.50      0.39      0.40    125620
weighted avg       0.92      0.95      0.93    125620

```
Confusion matrix (rows=true, cols=pred, order NORMAL/NEAR_MISS/COLLISION):
```
[[119194    394     32]
 [  1583    330     98]
 [  3654    300     35]]
```
Mean lead time before t_event: -0.04 s

## Feature importance (top 6, gain)
- `d`: 8712950
- `decel_a`: 1233249
- `decel_b`: 1192632
- `v_close`: 1167599
- `reversing_b`: 651471
- `speed_a`: 457524
