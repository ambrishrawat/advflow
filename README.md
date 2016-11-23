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

## Example of a horse

The epsilon used for FastGradientSign varies from 0.0 (top-left) to 0.1 (bottom-right).

![adv_horse](https://cloud.githubusercontent.com/assets/2141648/20572249/9d5b3e5a-b1a2-11e6-919b-20b333191e0c.jpg)

Difference from original-image

![adv_horse_diff](https://cloud.githubusercontent.com/assets/2141648/20572250/9d708512-b1a2-11e6-801a-2ad0f7f1498c.png)


**Note:** The compression algorithm/normalisation affects the imperceptibility of an image and its corresponsing adversarial image.

![adv_horse](https://cloud.githubusercontent.com/assets/2141648/20526315/6bfbe28e-b0bb-11e6-85b3-eb6f312af4b5.png)

the noisy pixels vanish when saved as jpeg

![adv_horse](https://cloud.githubusercontent.com/assets/2141648/20526317/6d52952e-b0bb-11e6-8cff-e05c860c62a5.jpg)
