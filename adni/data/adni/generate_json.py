import json
import numpy as np
import os, sys
from PIL import Image
import random
import pydicom
import csv

manufacturer_name_mapping={'CPS': 'CPS',
                           'GE MEDICAL SYSTEMS': 'GE_MEDICAL_SYSTEMS',
                           'GEMS': 'GEMS',
                           'HERMES': 'HERMES',
                            'MiE': 'MiE',
                           'Multiple': 'Multiple',
                           'Philips': 'Philips',
                            'Philips Medical Systems': 'Philips_Medical_Systems',
                           'SIEMENS': 'SIEMENS',
                            'Siemens ECAT': 'Siemens_ECAT',
                           'Siemens/CTI': 'Siemens_CTI'}


def rescale(image):
    im = Image.open(image)
    im = im.resize((32, 32))
    return im

def generate_data():

    x = []  # partitioned by manufacturers
    y = []  # partitioned by manufacturers

    data_path = sys.argv[1]
    label_path = './labels.txt'
    uid_mid_to_manu = {}

    with open(label_path, 'r') as f:
        labels = f.readlines()
    for label in labels:
        line = label.strip().split(",")
        if line[0] in uid_mid_to_manu:
            uid_mid_to_manu[line[0]][line[1]] = float(line[2])
        else:
            uid_mid_to_manu[line[0]] = {line[1]: float(line[2])}

    print('populated label mapping...')

    for manufacturer in os.listdir(data_path):
        xx = []
        yy = []
        images = os.listdir(os.path.join(data_path, manufacturer))[:-4]
        for img in images:
            if img.endswith('png'):  # ignore DS_Store
                uid, mid = img.strip().split("_")[3], img[:-4].strip().split("_")[-1]
                if uid in uid_mid_to_manu:
                    label = uid_mid_to_manu[uid][mid]
                    rescaled_numpy = np.asarray(rescale(os.path.join(data_path, manufacturer, img))) / 255.0
                    xx.append(rescaled_numpy.tolist())
                    yy.append(label)

        x.append(xx)
        y.append(yy)

    return x, y


def main():

    train_path = "data/train/mytrain.json"
    test_path = "data/test/mytest.json"

    all_data, all_labels = generate_data()
    print(len(all_data), len(all_labels))

    train_data = {'users': [], 'user_data': {}, 'num_samples': []}
    test_data = {'users': [], 'user_data': {}, 'num_samples': []}
    total_sample = 0
    for i, client_data in enumerate(all_data):
        if len(client_data) == 0:
            continue

        np.random.seed(666 + i)
        uname = 'f_{0:05d}'.format(i)
        train_data['users'].append(uname)
        train_data['user_data'][uname] = {'x': [], 'y': []}
        test_data['users'].append(uname)
        test_data['user_data'][uname] = {'x': [], 'y': []}

        combined = list(zip(all_data[i], all_labels[i]))
        random.shuffle(combined)
        all_data[i][:], all_labels[i][:] = zip(*combined)

        num_samples = len(all_labels[i])
        total_sample += num_samples

        unique_labels = np.unique(all_labels[i])
        num_unique_labels = len(unique_labels)

        random.shuffle(unique_labels)

        train_labels = unique_labels[:int(num_unique_labels * 0.8)]

        for j, lab in enumerate(all_labels[i]):
            if lab in train_labels:
                train_data['user_data'][uname]['x'].append(all_data[i][j])
                train_data['user_data'][uname]['y'].append(lab)
            else:
                test_data['user_data'][uname]['x'].append(all_data[i][j])
                test_data['user_data'][uname]['y'].append(lab)

        train_data['num_samples'].append(len(train_data['user_data'][uname]['y']))
        test_data['num_samples'].append(len(test_data['user_data'][uname]['y']))

    print('total samples: {}, {} samples per client'.format(total_sample, total_sample / (i+1)))
    print('begin to dump file...')

    with open(train_path, 'w') as outfile:
        json.dump(train_data, outfile)
    with open(test_path, 'w') as outfile:
        json.dump(test_data, outfile)


if __name__ == "__main__":
    main()
