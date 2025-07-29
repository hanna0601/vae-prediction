
from utils import *
from torch.autograd import Variable
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim
import torch.utils.data
import matplotlib.pyplot as plt
from torch.utils.data import TensorDataset, DataLoader
import numpy as np
import torch

# v1

num_question = 1774

def load_data(base_path="../data"):
    # Load training data as a sparse matrix and convert to dense numpy array
    train_matrix = load_train_sparse(base_path).toarray()
    # Load validation and test data as dictionaries
    valid_data = load_valid_csv(base_path)
    test_data = load_public_test_csv(base_path)

    # Copy the training matrix and fill missing entries with 0
    zero_train_matrix = train_matrix.copy()
    zero_train_matrix[np.isnan(train_matrix)] = 0
    # Convert numpy arrays to PyTorch tensors for model compatibility
    zero_train_matrix = torch.FloatTensor(zero_train_matrix)
    train_matrix = torch.FloatTensor(train_matrix)

    return zero_train_matrix, train_matrix, valid_data, test_data

# Variational AutoEncoder class definition
class VariationalAutoEncoder(nn.Module):
    def __init__(self, num_question, k=100):
        super(VariationalAutoEncoder, self).__init__()

        # Encoder definition using a Sequential container
        self.encoder = nn.Sequential(
            nn.Linear(num_question, k*4),  # Linear layer with ReLU activation
            nn.ReLU(),
            nn.Linear(k*4, k*2),  # Linear layer to reduce dimensionality further
            nn.ReLU(),
            nn.Linear(k*2, k),  # Final linear layer in encoder
            nn.ReLU(),
        )

        # Layers to compute the mean and log variance for the latent distribution
        self.fc21 = nn.Linear(k, k)  # mu layer
        self.fc22 = nn.Linear(k, k)  # logvar layer

        # Decoder definition
        self.decoder = nn.Sequential(
            nn.Linear(k, k*2),  # Decoder mirrors the encoder structure
            nn.ReLU(),
            nn.Linear(k*2, k*4),
            nn.ReLU(),
            nn.Linear(k*4, num_question),  # Final layer outputs reconstruction
        )
        
    # Encoder forward pass
    def encode(self, x):
        h1 = self.encoder(x)
        return self.fc21(h1), self.fc22(h1)  # Return mu and logvar

    # Reparameterization trick to allow backpropagation through stochastic nodes
    def reparameterize(self, mu, logvar):
        std = torch.exp(0.5*logvar)
        eps = torch.randn_like(std)
        return mu + eps*std

    # Decoder forward pass
    def decode(self, z):
        return torch.sigmoid(self.decoder(z))  # Use sigmoid to output probabilities

    # Full forward pass of the VAE
    def forward(self, x):
        mu, logvar = self.encode(x.view(-1, num_question))  # Encode input
        z = self.reparameterize(mu, logvar)  # Sample from latent space
        return self.decode(z), mu, logvar  # Decode the sample


# Loss function for VAE combining reconstruction loss and KL divergence
def loss_function(recon_x, x, mu, logvar):
    BCE = F.binary_cross_entropy(recon_x, x, reduction='sum')  # Binary Cross Entropy
    KLD = -0.5 * torch.sum(1 + logvar - mu.pow(2) - logvar.exp())  # Kullback-Leibler divergence
    # KLD = 0  # for hypothesis testing
    return BCE + KLD


# Training function for the VAE
def train_vae2(model, lr, lamb, train_data, zero_train_data, valid_data, test_data, num_epoch, batch_size=128):
    model.train()  # Set the model to training mode
    optimizer = optim.Adam(model.parameters(), lr=lr)  # Define optimizer

    # Create a dataset and loader for batching
    train_dataset = TensorDataset(zero_train_data, train_data)
    train_loader = DataLoader(train_dataset, batch_size=batch_size, shuffle=True)

    highest_score = {"valid": 0, "test": 0}
    loss_ls, acc_val, acc_test = [], [], []

    for epoch in range(num_epoch):
        model.train()
        train_loss = 0
        for inputs, targets in train_loader:  # Loop over batches
            optimizer.zero_grad()
            recon_batch, mu, logvar = model(inputs)  # Forward pass

            # Mask to keep only non-NaN entries for loss calculation
            nan_mask = ~torch.isnan(targets)
            recon_batch_masked = recon_batch[nan_mask]
            targets_masked = targets[nan_mask]

            # Compute loss and backpropagate
            loss = loss_function(recon_batch_masked, targets_masked, mu, logvar)
            loss.backward()
            train_loss += loss.item()
            optimizer.step()

        # Print the average loss for the epoch
        average_loss = train_loss / len(train_loader.dataset)
        loss_ls.append(average_loss)
        print(f'Epoch: {epoch}, Average loss: {average_loss:.4f}', end=' ')

        # Evaluate on the validation set
        model.eval()  # Set model to evaluation mode
        with torch.no_grad():  # Disable gradient computation
            validation_accuracy = evaluate_vae(model, zero_train_data, valid_data)
            print(f'Valid Acc: {validation_accuracy:.4f}', end=" ")

            test_accuracy = evaluate_vae(model, zero_train_data, test_data)
            print(f"Test Acc: {test_accuracy:.4f}", end=" | ")

            if validation_accuracy > highest_score['valid']:
                highest_score['valid'] = validation_accuracy
                highest_score['test'] = test_accuracy
            print(f"Final Test: {highest_score['test']:.4f}")

        acc_val.append(validation_accuracy)
        acc_test.append(test_accuracy)
    return model, loss_ls, acc_val, acc_test


# Evaluation function for the VAE
def evaluate_vae(model, train_data, valid_data):
    model.eval()  # Set the model to evaluation mode
    total = 0
    correct = 0
    with torch.no_grad():  # Disable gradient computation for evaluation
        for i, u in enumerate(valid_data["user_id"]):  # Loop over validation data
            inputs = Variable(train_data[u]).unsqueeze(0)  # Prepare input
            recon, _, _ = model(inputs)  # Forward pass

            # Make a prediction based on reconstruction probability
            pred = recon[0][valid_data["question_id"][i]].item() >= 0.5

            # Compare prediction to actual label
            if pred == valid_data["is_correct"][i]:
                correct += 1
            total += 1
    return correct / total  # Return the accuracy


# Main function where the model is trained and evaluated
def main():
    zero_train_matrix, train_matrix, valid_data, test_data = load_data()  # Load data

    k = 5  # Set the latent dimension size
    num_questions = zero_train_matrix.shape[1]  # Set the number of questions
    model = VariationalAutoEncoder(num_questions, k)  # Initialize the VAE model

    lr = 0.01  # Set the learning rate
    num_epoch = 50  # Set the number of epochs to train for
    lamb = 0.001  # Set the regularization strength

    epochs = list(range(1, num_epoch+1))

    # Train the model
    trained_model, loss, acc_val, acc_test = train_vae2(model, lr, lamb, train_matrix, zero_train_matrix, valid_data, test_data, num_epoch)

    plt.plot(epochs, loss, label="Loss over epoch")
    plt.xlabel('Epoch')
    plt.title('Training Loss over Epoch')
    plt.savefig("loss.png")
    plt.legend()

    plt.show()

    plt.plot(epochs, acc_val, label="Valid Acc")
    plt.plot(epochs, acc_test, label="Test Acc")
    plt.xlabel('Epoch')
    plt.title('Accuracy over Epoch (VAE)')
    plt.legend()
    plt.savefig("acc_vae.png")
    plt.show()

    # Evaluate the model on the validation set
    validation_accuracy = evaluate_vae(trained_model, zero_train_matrix, valid_data)
    print(f"Validation Accuracy: {validation_accuracy:.4f}")

    # Evaluate the model on the test set
    test_accuracy = evaluate_vae(trained_model, zero_train_matrix, test_data)
    print(f"Test Accuracy: {test_accuracy:.4f}")


if __name__ == "__main__":
    main()
