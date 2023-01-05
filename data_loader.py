from torchvision import transforms
from preprocessing import get_loader

transform_1 = transforms.Compose(
    [
        transforms.Resize((64, 64)),
        transforms.PILToTensor(),
        transforms.Normalize((0.5, 0.5, 0.5), (0.5, 0.5, 0.5)),
    ]
)

loader, dataset = get_loader(
    "data/annotations/captions_train2017.json",
    "data/train2017/",
    transform=transform_1,
)

VOCAB_SIZE = len(dataset.vocab)
