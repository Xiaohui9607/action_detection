import scipy.io
import glob
import os
import numpy as np
import PIL.Image
import pickle as pkl


def extract_labels(label):
    label = scipy.io.loadmat(label)['gnd'][:,0]
    return label

def extrac_images(im_path):
    images = sorted(glob.glob(os.path.join(im_path, "rgbjpg", "*.jpg")), key=lambda x:int(x.split('/')[-1][:-4]))
    images = [np.array(PIL.Image.open(image)) for image in images]
    images = np.stack(images)
    return images

if __name__ == '__main__':
    data_dir = "/home/golf/code/data/watch-n-bot/fridge_k1"
    out_dir =  "/home/golf/code/data/watch-n-bot/fridge_k1/out"
    images_paths = glob.glob(os.path.join(data_dir, "video", "*"))
    # labels_mat = glob.glob(os.path.join(data_dir,"*.mat"))
    classes = os.path.join(data_dir, "kitchen_classname.mat")

    classes = ['none']+[str(cls[0]) if cls[0] else '' for cls in scipy.io.loadmat(classes)['kitchen_classname'][0]]
    print(classes)

    # data = {}
    for images_path in images_paths:
        datapoint_name = images_path.split('/')[-1]
        out = {}
        label_path = os.path.join(data_dir, "label", datapoint_name, "gnd.mat")
        out['labels'] = extract_labels(label_path)
        out['images'] = extrac_images(images_path)
        pkl.dump(out, open(os.path.join(out_dir, "{}.pkl".format(datapoint_name)), 'wb'))

