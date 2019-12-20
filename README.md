# Nonzero & NMS

## Requirements

Download cub from [CUB Project Website](http://nvlabs.github.io/cub/) to `${HOME}/cub`.

## Nonzero

The implementation is equivalent to the following python codes.

```python
import torch
from torch.nn.functional import relu

torch.random.manual_seed(0)
scores = relu(torch.rand(20,) - 0.5)
boxes = torch.rand(80,) * 100
index = scores.nonzero().view(-1)

print(scores)
print(boxes)
print(index)
print(scores[index])
print(boxes.view(-1, 4)[index])
```

## NMS

The implementation is equivalent to the following python codes.

```python
import torch
from torchvision.ops import nms

torch.set_printoptions(sci_mode=False)

torch.random.manual_seed(0)
scores = torch.rand(100,)
centers = torch.rand(100, 2) * 100
sizes = torch.rand(100, 2) * 50
boxes = torch.cat([centers - sizes, centers + sizes], dim=1)

keep = nms(boxes.view(-1, 4), scores, 0.3)
mask = torch.zeros(100, dtype=torch.int)
mask[keep] = 1

print(scores)
print(boxes.view(-1))
print(mask)
print(keep)
print(len(keep))
```
