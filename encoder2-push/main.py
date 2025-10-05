import torch
from torch import nn
from torch.utils.data import DataLoader
from torchvision import datasets
from torchvision.transforms import ToTensor
import mlflow
import mlflow.pytorch
import matplotlib.pyplot as plt
import numpy as np
import random
from torch.utils.data import Dataset


device = torch.accelerator.current_accelerator().type if torch.accelerator.is_available() else "cpu"
print(f"Using {device} device\n\n")

#64-32-16-8-4-2-4-8-16-32-64
class SinAutoencoder(nn.Module):
    def __init__(self, bottleneck_size=2):
        super().__init__()
        self.encoder = nn.Sequential(
            nn.Linear(64, 32),
            nn.ReLU(),
            nn.Linear(32, 16),
            nn.ReLU(),
            nn.Linear(16, 8),
            nn.ReLU(),
            nn.Linear(8, 4),
            nn.ReLU(),
            nn.Linear(4, bottleneck_size),
        )
        self.decoder = nn.Sequential(
            nn.Linear(bottleneck_size, 4),
            nn.ReLU(),
            nn.Linear(4, 8),
            nn.ReLU(),
            nn.Linear(8, 16),
            nn.ReLU(),
            nn.Linear(16, 32),
            nn.ReLU(),
            nn.Linear(32, 64)
        )

    def forward(self, x):
        encoded = self.encoder(x)
        decoded = self.decoder(encoded)
        return decoded

def train(dataloader, model, loss_fn, optimizer, epoch=None, print_every=None):
    size = len(dataloader.dataset)
    model.train()
    running_loss = 0.0
    for batch, X in enumerate(dataloader):
        X = X.to(device)

        pred = model(X)
        loss = loss_fn(pred, X)

        # wstecz
        loss.backward()
        optimizer.step()
        optimizer.zero_grad()

        running_loss += loss.item()

        # mniej printów
        if print_every is not None and (batch + 1) % print_every == 0:
            print(f"Batch {batch+1}, Loss: {loss.item():.6f}")

    # strata lol
    avg_loss = running_loss / len(dataloader)
    if epoch is not None:
        print(f"Epoch {epoch+1} average training loss: {avg_loss:.6f}")
    return avg_loss



class SinDataset(Dataset):
    def __init__(self, N=2000, vector_length=64, T=20.0, phi=0.0, t_range=(0, 2*np.pi)):
        self.N = N
        self.vector_length = vector_length
        self.T = T
        self.phi = phi
        self.t_range = t_range
        self.x = np.arange(vector_length)
        self.samples = [self._make_sample() for _ in range(N)]
 
    def _make_sample(self):
        t = random.uniform(*self.t_range)
        vals = np.sin(self.x / self.T + self.phi + t).astype(np.float32)
        return vals
 
    def __len__(self):
        return self.N
 
    def __getitem__(self, idx):
        return torch.from_numpy(self.samples[idx])


def test(dataloader, model, loss_fn):
    num_batches = len(dataloader)
    model.eval()
    test_loss = 0
    with torch.no_grad():
        for X in dataloader:
            X = X.to(device)
            pred = model(X)
            test_loss += loss_fn(pred, X.view(pred.shape)).item()
    test_loss /= num_batches
    print(f"Test Avg loss: {test_loss:>8f}\n")
    return test_loss


def visualize_sin_reconstruction(model, dataloader, device, n=5):
    X = next(iter(dataloader))
    X = X.to(device)

    with torch.no_grad():
        reconstructed = model(X)

    X = X.cpu().numpy()
    reconstructed = reconstructed.cpu().numpy()

    plt.figure(figsize=(12, 6))
    for i in range(n):
        plt.subplot(n, 1, i+1)
        plt.plot(X[i], label="Original")
        plt.plot(reconstructed[i], label="Reconstructed", linestyle="dashed")
        plt.legend()
    plt.show()



def main():
    mlflow.set_experiment("encoder_2_clean")
    with mlflow.start_run():

        batch_size = 64
        learning_rate = 1e-4
        epochs = 1000

        mlflow.log_param("batch_size", batch_size)
        mlflow.log_param("learning_rate", learning_rate)
        mlflow.log_param("epochs", epochs)

        train_data = SinDataset(N=2000)
        test_data = SinDataset(N=500)
        train_dataloader = DataLoader(train_data, batch_size=batch_size, shuffle=True)
        test_dataloader = DataLoader(test_data, batch_size=batch_size, shuffle=False)

        model = SinAutoencoder().to(device)
        loss_fn = nn.MSELoss()
        optimizer = torch.optim.Adam(model.parameters(), lr=learning_rate)

        for t in range(epochs):
            if t == 0 or t == epochs-1 or (t+1) % 100 == 0:
                print(f"Epoch {t+1}\n-------------------------------")
                avg_train_loss = train(train_dataloader, model, loss_fn, optimizer, epoch=t)
                test_loss = test(test_dataloader, model, loss_fn)
                print(f"Test Avg loss: {test_loss:.6f}\n")
            else:
                avg_train_loss = train(train_dataloader, model, loss_fn, optimizer, epoch=None)
                test_loss = test(test_dataloader, model, loss_fn)

            mlflow.log_metric("train_loss", avg_train_loss, step=t)
            mlflow.log_metric("test_loss", test_loss, step=t)

        mlflow.pytorch.log_model(model, "model")
        print("logged  to MLflow")
        torch.save(model.state_dict(), "model.pth")
        print("saved  to model.pth")


        visualize_sin_reconstruction(model, test_dataloader, device)


if __name__ == "__main__":
    main()
