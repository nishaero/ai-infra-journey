import torch
import torchvision
import torchvision.transforms as transforms
from neural_network import Net

def test_data_loading():
    transform = transforms.Compose([transforms.ToTensor(), transforms.Normalize((0.5,), (0.5,))])
    trainset = torchvision.datasets.MNIST(root='./data', train=True, download=True, transform=transform)
    trainloader = torch.utils.data.DataLoader(trainset, batch_size=128, shuffle=True, num_workers=2)
    images, labels = next(iter(trainloader))
    assert images.shape == (128, 1, 28, 28), "Incorrect image shape"
    assert labels.shape == (128,), "Incorrect label shape"

def test_model_output():
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    model = Net().to(device)
    images = torch.randn(128, 1, 28, 28).to(device)
    outputs = model(images)
    assert outputs.shape == (128, 10), "Incorrect output shape"
    assert outputs.dtype == torch.float32, "Incorrect output dtype"

if __name__ == "__main__":
    test_data_loading()
    test_model_output()
    print("All tests passed!")
