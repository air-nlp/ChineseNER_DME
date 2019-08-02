## Chinese NER using Dynamic Meta-Embeddings
This repository contains a DME for chinese named entity recognition.



## Requirements
- [Tensorflow=1.2.0](https://github.com/tensorflow/tensorflow)
- pyltp


## Model
We investigate a new dynamic meta-embeddings method and apply it to Chinese NER task, which utilizes attention mechanism to combine features of both character and word granularity in embedding layer. The meta-embeddings created by our method are dynamic, data-specific and task-specific, since the meta-embeddings for same characters in different sentence sequences are distinct.


### Default parameters:
- gradient clip: 5
- embedding size: 50
- optimizer: Adam
- dropout rate: 0.5
- learning rate: 0.001

Character embeddings (gigaword_chn.all.a2b.uni.ite50.vec): [Baidupan](https://pan.baidu.com/s/1pLO6T9D#list/path=%2F)
Word embeddings (ctb.50d.vec): [Baidupan](https://pan.baidu.com/s/1pLO6T9D#list/path=%2F)

### Train the model with default parameters:
```shell
$ python3 main.py --train=True --clean=True
```

### Online evaluate:
```shell
$ python3 main.py
```

## Our paper:
[Chinese NER using Dynamic Meta-Embeddings](https://ieeexplore.ieee.org/stamp/stamp.jsp?tp=&arnumber=8715481)

