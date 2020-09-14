import scipy.io
import glob
import os
import numpy as np
import PIL.Image
import pickle as pkl
from configs import *
# import dill

"""
unique_labels_in_samples = [0, 1, 2, 3, 5, 7, 8, 9, 10]
np.unique(all_labels))
out['labels'] = map(lambda x : map_dict.get(x), out['labels'])
"""

"""
map_dict = {
        0: 0,
        1: 1,
        2: 2,
        3: 3,
        5: 4,
        7: 5,
        8: 6,
        9: 7,
        10: 8,
}
"""


SEQUENCE_LENGTH = 10
STEP = 4




def extract_labels(label_path):
    labels = scipy.io.loadmat(label_path)['gnd'][:,0]
    labels_list = labels[..., np.newaxis]
    ret = []
    for i in range(0, len(labels_list) - SEQUENCE_LENGTH, STEP):
        ret.append(np.concatenate(labels_list[i:i + SEQUENCE_LENGTH], axis=0))
    return ret

def extrac_images(im_path):
    images = sorted(glob.glob(os.path.join(im_path, "rgbjpg", "*.jpg")), key=lambda x:int(x.split('/')[-1][:-4]))
    images = [np.array(PIL.Image.open(image)) for image in images]
    ret = []
    for i in range(0, len(images) - SEQUENCE_LENGTH, STEP):
        ret.append(np.stack(images[i:i + SEQUENCE_LENGTH]))
    # try:
    #     images = np.stack(images)
    # except:
    #     print("need at least one array to stack")
    return ret

new_classes = classes[:3]
def generate_sample(images, labels, interaction="fridge"):
    new_labels = []
    new_images = []
    import cv2
    # last_image = np.array(PIL.Image.open(images[0]))
    X,Y,W,H = positions['campos_2']['fridge']
    for i in range(0, len(images)):
        image = np.array(PIL.Image.open(images[i]))
        label = classes[labels[i]] if 'fridge' in classes[labels[i]] else 'none'
        # cv2.imwrite("image.png",image[X:X+W, Y:Y+H])
        new_images.append(image[X:X+W, Y:Y+H].copy())
        new_labels.append(label)
    #     res = image.astype(np.float) - last_image.astype(np.float)
    #     res = np.mean(res)
    #     print(res, labels[i-1])
    #     last_image = image.copy()
    #     pass
    new_labels = [new_classes.index(label) for label in new_labels]
    new_labels = np.array(new_labels)
    new_images = np.stack(new_images)

    return new_images, new_labels

if __name__ == '__main__':
    campos = 'campos_2'

    data_dir = "../data/watch-n-bot/fridge_k1/{}".format(campos)

    out_dir = "../data/watch-n-bot/out"
    video_paths = glob.glob(os.path.join(data_dir, "videos", "*"))
    # label_paths = glob.glob(os.path.join(data_dir, "labels", "*"))
    label_paths = [path.replace("videos", "labels") for path in video_paths]

    for video, label in zip(video_paths, label_paths):
        image_paths = sorted(glob.glob(os.path.join(video, 'rgbjpg', '*')), key=lambda x:int(x.split('/')[-1][:-4]))
        labels = scipy.io.loadmat(os.path.join(label, 'gnd.mat'))['gnd'][:,0]
        images, labels = generate_sample(image_paths, labels)
        out = {}
        out['images'] = images.astype(np.uint8)
        out['labels'] = labels.astype(np.uint8)

        if not os.path.exists(out_dir):
            os.makedirs(out_dir)
        pkl.dump(out, open("{}.pkl".format(video), 'wb'))
        pass
    pass

    # labels_mat = glob.glob(os.path.join(data_dir,"*.mat"))
    # classes = os.path.join(data_dir, "kitchen_classname.mat")
    # classes = ['none']+[str(cls[0]) if cls[0] else '' for cls in scipy.io.loadmat(classes)['kitchen_classname'][0]]
    #
    #
    # for i, images_path in enumerate(images_paths):
    #     datapoint_name = images_path.split('/')[-1]
    #     label_path = os.path.join(data_dir, "labels", datapoint_name, "gnd.mat")
    #
    #     imgs_list = extrac_images(images_path)
    #     labels_list = extract_labels(label_path)
    #     for j, a_seq in enumerate(imgs_list):
    #         out = {}
    #         out['images'] = a_seq
    #         out['labels'] = labels_list[j]
    #
    #         if not os.path.exists(out_dir):
    #             os.makedirs(out_dir)
    #         pkl.dump(out, open(os.path.join(out_dir, "{}_{}.pkl".format(j, i)), 'wb'))

