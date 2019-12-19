# Nonzero

## Requirements

Download cub from [CUB Project Website](http://nvlabs.github.io/cub/) to `${HOME}/cub`.

## Nonzero

The implementation is equivalent to the following python codes.

```python
import torch
from torch.nn.functional import relu

torch.random.manual_seed(0)
scores = relu(torch.rand(20, ) - 0.5)
boxes = torch.rand(80, ) * 100
index = scores.nonzero().view(-1)

print(scores)
print(boxes)
print(index)
print(scores[index])
print(boxes.view(-1, 4)[index])
```
