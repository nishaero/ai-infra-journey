import torch
import torch.nn as nn
import torch.optim as optim
import torchvision
import torchvision.transforms as transforms
import matplotlib.pyplot as plt
import mlflow
import mlflow.pytorch
import psutil  # For system metrics

# Set random seed for reproducibility
torch.manual_seed(42)

# Check for GPU
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
print(f"Using device: {device}")

# Load and preprocess MNIST dataset
transform = transforms.Compose([transforms.ToTensor(), transforms.Normalize((0.5,), (0.5,))])
trainset = torchvision.datasets.MNIST(root='./data', train=True, download=True, transform=transform)
trainloader = torch.utils.data.DataLoader(trainset, batch_size=128, shuffle=True, num_workers=2)  # Increased batch size for GPU

# Define neural network
class Net(nn.Module):
    def __init__(self):
        super(Net, self).__init__()
        self.flatten = nn.Flatten()
        self.linear_relu_stack = nn.Sequential(
            nn.Linear(28 * 28, 256),  # Slightly larger for GPU
            nn.ReLU(),
            nn.Linear(256, 128),
            nn.ReLU(),
            nn.Linear(128, 10)
        )

    def forward(self, x):
        x = self.flatten(x)
        logits = self.linear_relu_stack(x)
        return logits

# Initialize model, loss function, and optimizer
model = Net().to(device)
criterion = nn.CrossEntropyLoss()
optimizer = optim.SGD(model.parameters(), lr=0.01, momentum=0.9)

# Start MLflow run
with mlflow.start_run():
    # Log parameters
    mlflow.log_param("batch_size", 128)
    mlflow.log_param("learning_rate", 0.01)
    mlflow.log_param("device", str(device))

    # Training loop
    epochs = 5
    train_losses = []
    train_accuracies = []

    for epoch in range(epochs):
        running_loss = 0.0
        correct = 0
        total = 0
        for images, labels in trainloader:
            images, labels = images.to(device), labels.to(device)
            optimizer.zero_grad()
            outputs = model(images)
            loss = criterion(outputs, labels)
            loss.backward()
            optimizer.step()

            running_loss += loss.item()
            _, predicted = torch.max(outputs.data, 1)
            total += labels.size(0)
            correct += (predicted == labels).sum().item()

        epoch_loss = running_loss / len(trainloader)
        epoch_acc = 100 * correct / total
        train_losses.append(epoch_loss)
        train_accuracies.append(epoch_acc)
        print(f'Epoch {epoch+1}, Loss: {epoch_loss:.3f}, Accuracy: {epoch_acc:.2f}%')

        # Log metrics to MLflow
        mlflow.log_metric("loss", epoch_loss, step=epoch)
        mlflow.log_metric("accuracy", epoch_acc, step=epoch)
        mlflow.log_metric("cpu_usage", psutil.cpu_percent(), step=epoch)
        if torch.cuda.is_available():
            mlflow.log_metric("gpu_memory", torch.cuda.memory_allocated() / 1024**2, step=epoch)  # MB

    # Log model to MLflow
    mlflow.pytorch.log_model(model, "model")

    # Plot training loss and accuracy
    plt.figure(figsize=(10, 5))
    plt.subplot(1, 2, 1)
    plt.plot(train_losses, 'b-', label='Training Loss')
    plt.xlabel('Epoch')
    plt.ylabel('Loss')
    plt.title('Training Loss')
    plt.legend()
    plt.savefig('loss_plot.png')
    plt.close()

    plt.figure(figsize=(10, 5))
    plt.subplot(1, 2, 2)
    plt.plot(train_accuracies, 'g-', label='Training Accuracy')
    plt.xlabel('Epoch')
    plt.ylabel('Accuracy (%)')
    plt.title('Training Accuracy')
    plt.legend()
    plt.savefig('accuracy_plot.png')
    plt.close()

    # Save sample predictions
    images, labels = next(iter(trainloader))
    images, labels = images.to(device), labels.to(device)
    outputs = model(images[:10])
    _, predicted = torch.max(outputs, 1)
    fig, axes = plt.subplots(2, 5, figsize=(12, 5))
    for i, ax in enumerate(axes.ravel()):
        ax.imshow(images[i].cpu().squeeze(), cmap='gray')
        ax.set_title(f'Pred: {predicted[i].item()}, True: {labels[i].item()}')
        ax.axis('off')
    plt.savefig('sample_predictions.png')
    plt.close()

    # Log plots to MLflow
    mlflow.log_artifact('loss_plot.png')
    mlflow.log_artifact('accuracy_plot.png')
    mlflow.log_artifact('sample_predictions.png')

print("Training complete! Plots and model logged to MLflow.")
