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
## Case 1. labels hold class ids
# batch_size, num_classes, height, width
B, num_classes, H, W = 2, 3, 4, 7
logits = torch.rand([B, num_classes, H, W])
labels = torch.randint(0, num_classes, [B, H, W])

# optional, class-wise weights, shape must be broadcastable to [B, num_classes, H, W]
# put more weight to class id 2
pos_weight = torch.tensor([1., 1., 3.]).reshape([1, num_classes, 1, 1])

loss = Poly1FocalLoss(num_classes=num_classes,
                      reduction='mean',
                      label_is_onehot=False,
                      pos_weight=pos_weight)

out = loss(logits, labels)
# out.backward()
# optimizer.step()


## Case 2. labels are one-hot or multi-hot (in case of multi-label task) encoded
# batch_size, num_classes, height, width
B, num_classes, H, W = 2, 3, 4, 7
logits = torch.rand([B, num_classes, H, W])
labels = torch.rand([B, num_classes, H, W]) # labels are of same shape as logits

# optionally provide class-wise weights, shape must be broadcastable to [B, num_classes, H, W]
# put 3 times more weight to class id 2
pos_weight = torch.tensor([1., 1., 3.]).reshape([1, num_classes, 1, 1])
# weight tensor shape [1, num_classes, 1, 1] is broadcastable to [B, num_classes, H, W]

loss = Poly1FocalLoss(num_classes=num_classes,
                      reduction='mean',
                      label_is_onehot=True,
                      pos_weight=pos_weight)

out = loss(logits, labels)
# out.backward()
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
* ***weight***, *(torch.Tensor)*, *(Default=None)*  - manual rescaling weight given to the loss of each batch element, passed to underlying [binary_cross_entropy loss](https://pytorch.org/docs/stable/generated/torch.nn.functional.binary_cross_entropy_with_logits.html) (*)
* ***pos_weight***, *(torch.Tensor)*, *(Default=None)*  - weight of positive examples, passed to underlying [binary_cross_entropy loss](https://pytorch.org/docs/stable/generated/torch.nn.functional.binary_cross_entropy_with_logits.html) (*)
* ***label_is_onehot***, *(bool)*, *(Default=False)*  - set to True if labels are one-hot (or multi-hot) encoded


\* Check formulas in the documentation page for [BCEWithLogitsLoss](https://pytorch.org/docs/stable/generated/torch.nn.BCEWithLogitsLoss.html) to understand how `weight (w_n)` and `pos_weight (p_c)` parameters are plugged into the loss function and how they affect the loss. Detailed explanation coming soon. Further discussions can be found in [this](https://discuss.pytorch.org/t/weight-vs-pos-weight-in-nn-bcewithlogitsloss/114859) and [this](https://discuss.pytorch.org/t/weights-in-bcewithlogitsloss/27452) threads.


## Requirements
* Python 3.6+
* Pytorch 1.1+