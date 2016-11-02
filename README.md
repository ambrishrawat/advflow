#AdvFlow

A library over TensorFlow to experimnent with Adversarial Images for tiny-ImageNet and CIFAR data set.

##Installation
To run the scripts, add repository to the PYTHONPATH

##Preprocessing the data sets
For preprocessing run the following command from the preproceesing folder
```
./make_csv.py --fpath='<path to>/tiny-imagenet-200/'
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


