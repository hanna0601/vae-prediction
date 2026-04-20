# VAE Prediction

This project uses a variational autoencoder to predict whether a student will answer a question correctly. The input is a sparse user-question response matrix, where known entries are binary correctness labels and missing entries are left as unknown. The model learns a compact latent representation for each user response vector, then reconstructs the probability of correctness for each question.

The work is centered on collaborative-filtering style prediction for educational data. Instead of using hand-built student or question features, it lets the neural network learn patterns from the response matrix directly.

## What is in the repo

- `code/vae.py` - VAE model, training loop, masked loss calculation, and validation/test evaluation.
- `code/utils.py` - data loading helpers for sparse matrices and CSV files.
- `result/acc_vae.png` - saved accuracy plot from a training run.
- `project_description.pdf` - original project brief.

## Techniques used

- Variational autoencoder (VAE) for matrix reconstruction.
- Encoder-decoder neural network built with PyTorch.
- Latent-variable modeling with learned mean and log-variance vectors.
- Reparameterization trick so the stochastic latent layer can be trained with backpropagation.
- Binary cross entropy reconstruction loss for correctness prediction.
- KL divergence regularization to keep the latent space close to a normal distribution.
- Masked loss calculation, so missing responses do not contribute to the training objective.
- Sparse matrix loading with SciPy, followed by dense tensor conversion for model training.
- Mini-batch training with `TensorDataset` and `DataLoader`.
- Adam optimizer.
- Accuracy evaluation by thresholding reconstructed probabilities at `0.5`.

## Expected data

The script expects a `data` directory next to the repo root:

```text
data/
  train_sparse.npz
  valid_data.csv
  test_data.csv
```

`train_sparse.npz` should store the user-question training matrix. The CSV files should follow this column order:

```text
question_id,user_id,is_correct
```

The loader skips a header row if one is present.

## Running it

From the repository root:

```bash
cd code
python vae.py
```

The main script trains the model for 50 epochs using:

- latent dimension `k = 5`
- learning rate `0.01`
- batch size `128`
- Adam optimization

During training it prints the average training loss, validation accuracy, test accuracy, and the best test accuracy tied to the best validation result.

## Output

The script saves two plots from a run:

- `loss.png` - training loss over epochs
- `acc_vae.png` - validation and test accuracy over epochs

The repository also includes a previous accuracy plot in `result/acc_vae.png`.

## Notes

The missing entries in the training matrix are filled with zero before being passed into the model, but they are masked out when computing the reconstruction loss. That keeps the input tensor complete while avoiding the mistake of treating every missing answer as incorrect.

The `lamb` argument is still present in the training function signature, but the current VAE loss uses binary cross entropy plus KL divergence and does not apply a separate weight decay term.
