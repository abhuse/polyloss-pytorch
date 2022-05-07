# PolyLoss in Pytorch

PolyLoss implementation in Pytorch as described in:  
[[Leng et al. 2022] PolyLoss: A Polynomial Expansion Perspective of Classification Loss Functions](https://arxiv.org/abs/2204.12511)

Both Poly-Cross-Entropy and Poly-Focal losses are provided.

## Examples

```python
import torch

# Poly1 Cross-Entropy Loss
# classification task
batch_size = 10
num_classes = 5
logits = torch.rand([batch_size, num_classes])
labels = torch.randint(0, num_classes, [batch_size])
loss = Poly1CrossEntropyLoss(num_classes=num_classes, 
                             reduction='mean')
out = loss(logits, labels)
out.backward()
# optimizer.step()

# Poly1 Focal Loss
# segmentation task
H, W = 4, 7
logits = torch.rand([batch_size, num_classes, H, W])
labels = torch.randint(0, num_classes, [batch_size, H, W])
loss = Poly1FocalLoss(num_classes=num_classes, 
                      reduction='mean')
out = loss(logits, labels)
out.backward()
# optimizer.step()
```

## Parameters


### Poly1CrossEntropyLoss
* ***num_classes***, *(int)* - Number of classes
* ***epsilon***, *(float)*, *(Default=1.0)* - PolyLoss epsilon
* ***reduction***, *(str)*, *(Default='none')*  - apply reduction to the output, one of: none | sum | mean
* ***weight***, *(torch.Tensor)*, *(Default=None)*  - manual rescaling weight for each class, passed to Cross-Entropy loss

### Poly1FocalLoss
* ***num_classes***, *(int)* - Number of classes
* ***epsilon***, *(float)*, *(Default=1.0)* - PolyLoss epsilon
* ***alpha***, *(float)*, *(Default=0.25)* - Focal loss alpha 
* ***gamma***, *(float)*, *(Default=2.0)* - Focal loss gamma
* ***reduction***, *(str)*, *(Default='none')*  - apply reduction to the output, one of: none | sum | mean
* ***weight***, *(torch.Tensor)*, *(Default=None)*  - manual rescaling weight for each class, passed to binary Cross-Entropy loss
* ***label_is_onehot***, *(bool)*, *(Default=False)*  - set to True if labels are one-hot encoded


## Requirements
* Python 3.6+
* Pytorch 1.1+