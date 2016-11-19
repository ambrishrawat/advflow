#AdvFlow

A library over TensorFlow and Keras to experimnent with Adversarial Images.
Examples included for CIFAR-10 data sets.

##Installation
Add repository to your PYTHONPATH

**Requirements**: tensorflow, keras

## Preprocessing the data sets
```
./preprcessing/load_npy.py (-h for help)
```

## Training

Different model defintions from model_defs.py can be trained using this script. 

```
./train.py (-h for help)
```

## Testing
Trained models can be evaluated with std-droput and mc-dropput interpretaions
```
./test.py (-h for help)
```

## Generate adversrial images
Adversrial images for the the CIFAR10 images can be generated and saved using this script
```
./genadv.py (-h for help)
```


