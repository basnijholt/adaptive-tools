# adaptive-tools
Tools for python-adaptive


## Installation
```
pip install -U https://github.com/basnijholt/adaptive-tools/archive/master.zip
```

## Usage
```
from adaptive_tools import Learner2D
learner = Learner2D(...)
learner.save('filename.p')
# do your thing


# next time
learner = Learner2D(...)
learner.load('filename.p')
```
