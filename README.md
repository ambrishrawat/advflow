A repository to try experimnentation with Adversarial Images for tiny-ImageNet data set.

To run the scripts, add repository to the PYTHONPATH

For preprocessing run the following command from the preproceesing folder


Preprocessing
Run the make_csv.py script from the preprocessing folder
./make_csv.py --fpath='<path to>/tiny-imagenet-200/'

For training
./train.py -h (for help)
./train.py --epochs=100 --batchsize=128 --mid=madam100_128


For testing
./test.py -h (for help)
./test.py --csvpath=preprocessing/valset.csv --batchsize=128 --mid=madam100_128



