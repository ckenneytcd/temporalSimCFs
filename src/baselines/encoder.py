from typing import List, Union
import torch.nn as nn
import numpy as np
import torch
import pandas as pd


class VariationalAutoencoder(nn.Module):
    def __init__(self, layers: List):
        """
        Parameters
        ----------
        layers:
            List of layer sizes.
        """
        super(VariationalAutoencoder, self).__init__()

        if len(layers) < 2:
            raise ValueError(
                "Number of layers have to be at least 2 (input and latent space), and number of neurons bigger than 0"
            )

        self._input_dim = layers[0]
        latent_dim = layers[-1]

        # The VAE components
        lst_encoder = []
        for i in range(1, len(layers) - 1):
            lst_encoder.append(nn.Linear(layers[i - 1], layers[i]))
            lst_encoder.append(nn.ReLU())
        encoder = nn.Sequential(*lst_encoder)

        self._mu_enc = nn.Sequential(encoder, nn.Linear(layers[-2], latent_dim))
        self._log_var_enc = nn.Sequential(encoder, nn.Linear(layers[-2], latent_dim))

        lst_decoder = []
        for i in range(len(layers) - 2, 0, -1):
            lst_decoder.append(nn.Linear(layers[i + 1], layers[i]))
            lst_decoder.append((nn.ReLU()))
        decoder = nn.Sequential(*lst_decoder)

        self.mu_dec = nn.Sequential(
            decoder,
            nn.Linear(layers[1], self._input_dim),
            nn.Sigmoid(),
        )

        device = "cuda" if torch.cuda.is_available() else "cpu"
        self.to(device)

    def encode(self, x):
        print(x)
        return self._mu_enc.float()(x.float()), self._log_var_enc.float()(x.float())

    def decode(self, z):
        return self.mu_dec(z)

    def __reparametrization_trick(self, mu, log_var):
        std = torch.exp(0.5 * log_var)
        epsilon = torch.randn_like(std)  # the Gaussian random noise
        return mu + std * epsilon

    def forward(self, x):

        # split up the input in a mutable and immutable part
        x = x.clone()

        # the mutable part gets encoded
        mu_z, log_var_z = self.encode(x)
        z = self.__reparametrization_trick(mu_z, log_var_z)
        recon = self.decode(z)

        # add the immutable features to the reconstruction
        x = recon

        return x, mu_z, log_var_z

    def predict(self, data):
        return self.forward(data)

    def kld(self, mu, logvar):
        KLD = -0.5 * torch.sum(1 + logvar - mu.pow(2) - logvar.exp())
        return KLD

    def fit(
        self,
        xtrain: Union[pd.DataFrame, np.ndarray],
        kl_weight=0.3,
        lambda_reg=1e-6,
        epochs=50,
        lr=1e-4,
        batch_size=64,
    ):
        if isinstance(xtrain, pd.DataFrame):
            xtrain = xtrain.values

        train_loader = torch.utils.data.DataLoader(
            xtrain, batch_size=batch_size, shuffle=True
        )

        optimizer = torch.optim.Adam(
            self.parameters(),
            lr=lr,
            weight_decay=lambda_reg,
        )

        criterion = nn.BCELoss(reduction="mean")

        # Train the VAE with the new prior
        ELBO = np.zeros((epochs, 1))
        print("Start training of Variational Autoencoder...")
        for epoch in range(epochs):

            beta = epoch * kl_weight / epochs

            # Initialize the losses
            train_loss = 0
            train_loss_num = 0

            # Train for all the batches
            for data in train_loader:
                data = data.view(data.shape[0], -1)
                data = data.float()

                # forward pass
                reconstruction, mu, log_var = self(data)

                recon_loss = criterion(reconstruction, data)
                kld_loss = self.kld(mu, log_var)
                loss = recon_loss + beta * kld_loss

                # Update the parameters
                optimizer.zero_grad()
                # Compute the loss
                loss.backward()
                # Update the parameters
                optimizer.step()

                # Collect the ways
                train_loss += loss.item()
                train_loss_num += 1

            ELBO[epoch] = train_loss / train_loss_num
            if epoch % 10 == 0:
                print(
                    "[Epoch: {}/{}] [objective: {:.3f}]".format(
                        epoch, epochs, ELBO[epoch, 0]
                    )
                )

            # ELBO_train = ELBO[epoch, 0].round(2)
            # print("[ELBO train: " + str(ELBO_train) + "]")

        print("... finished training of Variational Autoencoder.")

        # self.eval()