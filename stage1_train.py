from stage1GAN import *

transform = transforms.Compose([transforms.Resize((64, 64)), transforms.PILToTensor()])

loader, dataset = get_loader(
    "data/annotations/captions_train2017.json",
    "data/train2017/",
    transform=transform,
)



# def train_GAN(loader, ):


if __name__ == "__main__":
    pass
