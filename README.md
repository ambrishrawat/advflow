#AdvFlow

A library over TensorFlow and Keras to experimnent with Adversarial Images.
Examples included for tiny-ImageNet and CIFAR-10 data sets.

##Installation
Add repository to your PYTHONPATH

##Preprocessing the data sets
For preprocessing run the following command from the preproceesing folder
```
./make_csv.py --fpath='<path to>/tiny-imagenet-200/'
```
or
```
./make_csv.py --fpath='<path to>/cifar-10-batches-py/' --cifarpath='<destination path for saving JPEGs>'
```
##Training
```
./train.py -h (for help)
./train.py --epochs=100 --batchsize=128 --mid=madam100_128
```

##Testing
```
./test.py -h (for help)
./test.py --csvpath=preprocessing/valset.csv --batchsize=128 --mid=madam100_128
```


