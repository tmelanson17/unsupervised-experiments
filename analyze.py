import pickle

from torch.utils.tensorboard import SummaryWriter 
import torch
import torchvision.transforms

from data_loader import UnsupervisedImageDataset, EAI_TRANSFORM

def create_embedding(dataset, logpath):
    writer = SummaryWriter(logpath)
    # helper function
    def select_n_random(dataset, n=100):
        '''
        Selects n random datapoints and their corresponding features from a dataset
        '''
        perm = torch.randperm(len(dataset))[:n]
        return dataset[perm]

    # select random images and their target indices
    images, features = select_n_random(dataset, n=27)

    # log embeddings
    writer.add_embedding(features,
                        label_img=images)
    writer.close()

def print_comparisons():
    with open("comparisons_raw_distance.pkl", "rb") as fo:
        comparisons = pickle.load(fo)
        for file in comparisons:
            print(f"Filename: {file}")
            top_5 = comparisons[file]["score"]
            files = comparisons[file]["files"]
            print(top_5)
            print(files)


if __name__ == "__main__":
    EMB_PATH="embeddings.pkl"
    dataset = UnsupervisedImageDataset(EMB_PATH, transform=EAI_TRANSFORM)
    create_embedding(dataset, "/home/noisebridge/tensorboard-experiment-1")

