import csv
import random

with open ("preprocessing/train_tinyImageNet.csv") as f:
    l=list(csv.reader(f))

header = l[0]
body = l[1:]

print(body[0])
random.shuffle(body)
random.shuffle(body)
random.shuffle(body)
random.shuffle(body)
random.shuffle(body)
random.shuffle(body)
random.shuffle(body)
random.shuffle(body)

with open("random.csv", "w") as f:
    csv.writer(f).writerow(header)
with open("random.csv", "a") as f:
    csv.writer(f).writerows(body)
