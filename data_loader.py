from torchvision import transforms
from preprocessing import get_loader

transform_1 = transforms.Compose(
    [
        transforms.Resize((64, 64)),
        transforms.ToTensor(),
        transforms.Normalize((0.5, 0.5, 0.5), (0.5, 0.5, 0.5)),
    ]
)

batch_size = 32

loader, dataset = get_loader(
    "data/annotations/captions_train2017.json",
    "data/train2017/",
    transform=transform_1,
    batch_size=batch_size,
)

VOCAB_SIZE = len(dataset.vocab)
