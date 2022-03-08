# 语义分割评价指标


## 语义分割评价指标

### 引言

语义分割是像素级别的分类，其常用评价指标：
**像素准确率**（Pixel Accuracy，**PA**）、
**类别像素准确率**（Class Pixel Accuray，**CPA**）、
**类别平均像素准确率**（Mean Pixel Accuracy，**MPA**）、
**交并比**（Intersection over Union，**IoU**）、
**平均交并比**（Mean Intersection over Union，**MIoU**），
其计算都是建立在混淆矩阵（Confusion Matrix）的基础上。因此，了解基本的混淆矩阵知识对理解上述5个常用评价指标是很有益处的！

![截图](/images/278eff16544a944f6366f0c6e4e4e6f2.png)

### 评价指标

#### PA：像素准确率

- 对应：准确率（Accuracy）
- 含义：预测类别正确的像素数占总像素数的比例
- 混淆矩阵计算：
  - 对角线元素之和 / 矩阵所有元素之和
  - PA = (TP + TN) / (TP + TN + FP + FN)

#### CPA：类别像素准确率

- 对应：精准率（Precision）
- 含义：在类别 i 的预测值中，真实属于 i 类的像素准确率，换言之：模型对类别 i 的预测值有很多，其中有对有错，预测对的值占预测总值的比例
- 混淆矩阵计算：
  - 类1：P1 = TP / (TP + FP)
  - 类2：P2 = TN / (TN + FN)
  - 类3：…

#### MPA：类别平均像素准确率

- 含义：分别计算每个类被正确分类像素数的比例，即：CPA，然后累加求平均
- 混淆矩阵计算：
  - 每个类别像素准确率为：Pi（计算：对角线值 / 对应列的像素总数）
  - MPA = sum(Pi) / 类别数

#### IoU：交并比

- 含义：模型对某一类别预测结果和真实值的交集与并集的比值
- 混淆矩阵计算：
  - 以求二分类：正例（类别1）的IoU为例
  - 交集：TP，并集：TP、FP、FN求和
  - IoU = TP / (TP + FP + FN)

#### MIoU：平均交并比

- 含义：模型对每一类预测的结果和真实值的交集与并集的比值，求和再平均的结果
- 混淆矩阵计算：
  - 以求二分类的MIoU为例
  - MIoU = (IoU正例p + IoU反例n) / 2 = [ TP / (TP + FP + FN) + TN / (TN + FN + FP) ] / 2

### 数值计算

1. 计算混淆矩阵

```python
def get_confusion_matrix(scores, labels):
    """Computes the confusion matrix of one batch

    Args:
        scores (torch.FloatTensor, shape (B?, N, C):
            raw scores for each class.
        labels (torch.LongTensor, shape (B?, N)):
            ground truth labels.

    Returns:
        Confusion matrix for current batch.
    """
    C = scores.size(-1)
    y_pred = scores.detach().cpu().numpy().reshape(-1, C)  # (N, C)
    y_pred = np.argmax(y_pred, axis=1)  # (N,)

    y_true = labels.detach().cpu().numpy().reshape(-1,)
    # 此处类似进制，C为类别数。最后对索引计数则可以得到混淆矩阵的扁平化分布
    y = np.bincount(C * y_true + y_pred, minlength=C * C) 

    if len(y) < C * C:
        y = np.concatenate([y, np.zeros((C * C - len(y)), dtype=np.long)])
    else:
        if len(y) > C * C:
            warnings.warn(
                "Prediction has fewer classes than ground truth. This may affect accuracy."
            )
        y = y[-(C * C):]  # last c*c elements.

    y = y.reshape(C, C)

  
  return y
```

```python
# 计算混淆矩阵 核心代码
def get_confusion_matrix(label_true, label_pred, n_class):
    mask = (label_true >= 0) & (label_true < n_class)
    hist = np.bincount(
        n_class * label_true[mask].astype(int) +
        label_pred[mask], minlength=n_class ** 2).reshape(n_class, n_class)
    return hist
```

2. 得到acc 与IoU

```python
tp = np.longlong(self.confusion_matrix[label, label])
fn = np.longlong(self.confusion_matrix[label, :].sum()) - tp
fp = np.longlong(self.confusion_matrix[:, label].sum()) - tp

if tp + fp + fn == 0:
    iou = float('nan')
else:
    iou = (tp) / (tp + fp + fn)
    
if tp + fn == 0:
    acc = float('nan')
else:
    acc = tp / (tp + fn)
```

