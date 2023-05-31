import vc_models
from vc_models.models.vit import model_utils

import cv2
import numpy as np
import torch

import os
import pickle
import time

IMG_DIR = "/home/noisebridge/nb-pics"
IMG_FILES = [
        os.path.join(IMG_DIR, f) 
        for f in os.listdir(IMG_DIR)
]

# Assume vectorized for rh
def cosine_similarity(lh, rh):
    return np.dot(lh, rh.T)/(np.linalg.norm(lh)*np.linalg.norm(rh, axis=-1))

def raw_distance(lh, rh):
    return np.linalg.norm(lh - rh, axis=-1)


def nth_nearest(embeddings, files):
    result = dict()
    for emb, f in zip(embeddings, files):
        similarities = raw_distance(emb, embeddings)
        top_5 = np.argsort(similarities)[:5]
        result[f] = {"files": files[top_5], "score": similarities[top_5]}
    return result

    

if __name__ == "__main__":
    model,embd_size,model_transforms,model_info = model_utils.load_model(model_utils.VC1_BASE_NAME)


    imgs = []
    for img_file in IMG_FILES:
        img = cv2.imread(img_file)
        imgs.append(img) 
    imgs_np = np.array(imgs).astype(np.float32) / 255
    print(imgs_np.shape)

    imgs_np = imgs_np.transpose((0, 3, 1, 2))
    imgs_torch = torch.from_numpy(imgs_np)
    imgs_transforms =  model_transforms(imgs_torch)
    print(imgs_transforms.shape)
    before = time.time()
    embeddings = model(imgs_transforms).detach().numpy()
    after = time.time()
    embed_data = {"embeddings": embeddings, "files": np.array(IMG_FILES)}
    print(f"Inference time: {after-before}")
    with open("embeddings.pkl", "wb") as fo:
        pickle.dump(embed_data, fo)
        fo.close()
    nearest = nth_nearest(embeddings, np.array(IMG_FILES))
    with open("comparisons_raw_distance.pkl", "wb") as fo:
        pickle.dump(nearest, fo)
        fo.close()


