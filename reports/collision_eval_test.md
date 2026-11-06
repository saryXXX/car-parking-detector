# Collision / near-miss model evaluation
Trained on `data\sim\traj_small.parquet`, 18691 train rows, 4803 held-out-range rows, 1785 random-val rows.
Features used (metric simulator only; `overlap_iou` from the full F1 spec needs real pixel boxes and is added at runtime integration, not here).

## Held-out parameter range (B_speed>5)

### LightGBM
```
              precision    recall  f1-score   support

      NORMAL       0.97      0.96      0.96      4568
   NEAR_MISS       0.00      0.00      0.00         0
   COLLISION       0.29      0.31      0.30       235

    accuracy                           0.93      4803
   macro avg       0.42      0.42      0.42      4803
weighted avg       0.93      0.93      0.93      4803

```
Confusion matrix (rows=true, cols=pred, order NORMAL/NEAR_MISS/COLLISION):
```
[[4378   11  179]
 [   0    0    0]
 [ 157    4   74]]
```
Mean lead time before t_event: 0.43 s

### Rule (ported to metres)
```
              precision    recall  f1-score   support

      NORMAL       0.95      0.99      0.97      4568
   NEAR_MISS       0.00      0.00      0.00         0
   COLLISION       0.00      0.00      0.00       235

    accuracy                           0.95      4803
   macro avg       0.32      0.33      0.32      4803
weighted avg       0.91      0.95      0.93      4803

```
Confusion matrix (rows=true, cols=pred, order NORMAL/NEAR_MISS/COLLISION):
```
[[4541   27    0]
 [   0    0    0]
 [ 224   11    0]]
```
Mean lead time before t_event: -0.36 s

## Random validation split

### LightGBM
```
              precision    recall  f1-score   support

      NORMAL       0.95      0.95      0.95      1651
   NEAR_MISS       0.00      0.00      0.00         0
   COLLISION       0.43      0.40      0.42       134

    accuracy                           0.91      1785
   macro avg       0.46      0.45      0.46      1785
weighted avg       0.91      0.91      0.91      1785

```
Confusion matrix (rows=true, cols=pred, order NORMAL/NEAR_MISS/COLLISION):
```
[[1568   11   72]
 [   0    0    0]
 [  80    0   54]]
```
Mean lead time before t_event: 1.62 s

### Rule (ported to metres)
```
              precision    recall  f1-score   support

      NORMAL       0.93      1.00      0.96      1651
   NEAR_MISS       0.00      0.00      0.00         0
   COLLISION       0.00      0.00      0.00       134

    accuracy                           0.92      1785
   macro avg       0.31      0.33      0.32      1785
weighted avg       0.86      0.92      0.89      1785

```
Confusion matrix (rows=true, cols=pred, order NORMAL/NEAR_MISS/COLLISION):
```
[[1648    3    0]
 [   0    0    0]
 [ 128    6    0]]
```
Mean lead time before t_event: -0.57 s

## Feature importance (top 6, gain)
- `d`: 161647
- `decel_a`: 108579
- `decel_b`: 57664
- `heading_rel`: 55736
- `v_close`: 11779
- `speed_a`: 8609
