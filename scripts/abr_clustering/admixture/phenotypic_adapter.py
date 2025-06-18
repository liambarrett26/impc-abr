"""
Phenotypic adapter for Neural ADMIXTURE.
Modifies the core Neural ADMIXTURE components to handle phenotypic data.
"""

import torch
import numpy as np
from typing import Dict, List, Tuple, Optional
import logging
from sklearn.mixture import GaussianMixture
from sklearn.cluster import KMeans

logger = logging.getLogger(__name__)


class PhenotypicNeuralAdmixture:
    """
    Adapter class that modifies Neural ADMIXTURE for phenotypic clustering.
    Handles the differences between genomic and phenotypic data.
    """

    def __init__(self,
                 min_k: int,
                 max_k: int,
                 epochs: int,
                 batch_size: int,
                 learning_rate: float,
                 hidden_size: int,
                 device: torch.device,
                 random_seed: int = 42):
        """
        Initialize phenotypic Neural ADMIXTURE.

        Args:
            min_k: Minimum number of clusters
            max_k: Maximum number of clusters
            epochs: Training epochs
            batch_size: Batch size
            learning_rate: Learning rate
            hidden_size: Hidden layer size
            device: Computing device
            random_seed: Random seed
        """
        self.min_k = min_k
        self.max_k = max_k
        self.epochs = epochs
        self.batch_size = batch_size
        self.learning_rate = learning_rate
        self.hidden_size = hidden_size
        self.device = device
        self.random_seed = random_seed

        # Set random seeds
        torch.manual_seed(random_seed)
        np.random.seed(random_seed)
        if torch.cuda.is_available():
            torch.cuda.manual_seed(random_seed)

    def adapt_neural_admixture_forward(self, original_forward_method):
        """
        Create adapted forward method for phenotypic data.

        Args:
            original_forward_method: Original Q_P.forward method

        Returns:
            Adapted forward method
        """
        def phenotypic_forward(self_model, X: torch.Tensor):
            """
            Adapted forward pass for phenotypic data.

            Args:
                X: Phenotypic feature tensor [N, M] in range [0,1]

            Returns:
                Same as original forward method
            """
            # Skip genomic-specific preprocessing
            # X is already in [0,1] range from our preprocessing

            # Apply identity transformation instead of PCA
            # (V matrix should be identity for phenotypic data)
            X_processed = X @ self_model.V

            # Apply batch normalization
            X_processed = self_model.batch_norm(X_processed)

            # Continue with original encoder-decoder pipeline
            enc = self_model.common_encoder(X_processed)
            hid_states = self_model.multihead_encoder(enc)
            probs = [self_model.softmax(h) for h in hid_states]

            return self_model.return_func(probs), X

        return phenotypic_forward

    def initialize_P_matrix_phenotypic(self,
                                     data: torch.Tensor,
                                     ks_list: List[int]) -> torch.Tensor:
        """
        Initialize P matrix using k-means clustering on phenotypic data.

        Args:
            data: Phenotypic data tensor [N, M]
            ks_list: List of K values

        Returns:
            Initialized P matrix
        """
        logger.info("Initializing P matrix using k-means on phenotypic data")

        # Convert to numpy for sklearn
        data_np = data.cpu().numpy() if data.is_cuda else data.numpy()

        # Initialize P matrices for all K values
        P_matrices = []

        for k in ks_list:
            # Use k-means to find initial cluster centers
            kmeans = KMeans(
                n_clusters=k,
                init='k-means++',
                n_init=10,
                random_state=self.random_seed
            )

            # Fit k-means
            kmeans.fit(data_np)

            # Use cluster centers as initial P matrix
            # Clamp to valid probability range
            P_k = np.clip(kmeans.cluster_centers_, 0.01, 0.99)
            P_matrices.append(P_k)

        # Concatenate all P matrices
        P_combined = np.concatenate(P_matrices, axis=0)

        logger.info(f"Initialized P matrix shape: {P_combined.shape}")
        return torch.tensor(P_combined, dtype=torch.float32)

    def create_identity_projection(self, n_features: int) -> np.ndarray:
        """
        Create identity matrix for V (no PCA needed for phenotypic data).

        Args:
            n_features: Number of features

        Returns:
            Identity matrix
        """
        return np.eye(n_features, dtype=np.float32)

    def train_phenotypic_admixture(self,
                                data: torch.Tensor,
                                metadata: Dict) -> Tuple[List[np.ndarray], List[np.ndarray], object]:
        """
        Train Neural ADMIXTURE on phenotypic data.

        Args:
            data: Preprocessed phenotypic data [N, M] in range [0,1]
            metadata: Metadata dictionary

        Returns:
            Tuple of (Ps, Qs, trained_model)
        """
        logger.info(f"Training phenotypic Neural ADMIXTURE on {data.shape}")

        # Import Neural ADMIXTURE components
        from neural_admixture_original.model.neural_admixture import NeuralAdmixture, Q_P
        from neural_admixture_original.src.loaders import dataloader_admixture

        N, M = data.shape
        ks_list = list(range(self.min_k, self.max_k + 1))

        # Create identity V matrix (no PCA projection)
        V = self.create_identity_projection(M)
        V_tensor = torch.tensor(V.T, dtype=torch.float32, device=self.device)

        # Initialize P matrix using k-means
        P_init = self.initialize_P_matrix_phenotypic(data, ks_list)
        P_init = P_init.to(self.device)

        # Create modified Q_P model
        model = Q_P(
            hidden_size=self.hidden_size,
            num_features=M,
            V=V_tensor,
            P=P_init,
            ks_list=ks_list,
            is_train=True
        ).to(self.device)

        # Patch the forward method
        model.forward = self.adapt_neural_admixture_forward(model.forward).__get__(model, Q_P)

        # Setup optimizer
        optimizer = model.create_custom_adam(device=self.device, lr=self.learning_rate)

        # Loss function
        loss_function = torch.nn.BCELoss(reduction='sum').to(self.device)

        # Training loop
        model.train()
        logger.info(f"Starting training: {self.epochs} epochs")

        for epoch in range(self.epochs):
            epoch_loss = 0

            # Create dataloader
            pops = torch.zeros(N, device=self.device)  # Dummy population labels
            dataloader = dataloader_admixture(
                data, self.batch_size, 1, self.random_seed,
                torch.Generator().manual_seed(self.random_seed),
                pops, shuffle=True
            )

            for batch_x, _ in dataloader:
                batch_x = batch_x.to(self.device)

                # Forward pass
                optimizer.zero_grad()
                output_tuple, original = model(batch_x)

                # FIXED: Handle the return format correctly
                # output_tuple contains (reconstructions_list, probs_list) when training
                reconstructions_list, probs_list = output_tuple

                # Calculate loss for all reconstruction heads
                batch_loss = 0
                for reconstruction in reconstructions_list:  # Now reconstruction is a tensor
                    batch_loss += loss_function(reconstruction, original)

                # Backward pass
                batch_loss.backward()
                optimizer.step()

                # Clamp P matrix values
                model.restrict_P()

                epoch_loss += batch_loss.item()

            if epoch % 10 == 0:
                logger.info(f"Epoch {epoch}: Loss = {epoch_loss:,.0f}")

        # Inference phase
        model.eval()
        model.return_func = model._return_infer

        Qs = [torch.empty(0, k, device=self.device) for k in ks_list]

        with torch.no_grad():
            pops = torch.zeros(N, device=self.device)
            dataloader = dataloader_admixture(
                data, min(N, 5000), 1, self.random_seed,
                torch.Generator().manual_seed(self.random_seed),
                pops, shuffle=False
            )

            for batch_x, _ in dataloader:
                batch_x = batch_x.to(self.device)
                probs, _ = model(batch_x)  # In inference mode, just returns probs

                for i, prob in enumerate(probs):
                    Qs[i] = torch.cat([Qs[i], prob], dim=0)

        # Extract results
        Ps = [decoder.weight.data.cpu().numpy() for decoder in model.decoders.decoders]
        Qs = [Q.cpu().numpy() for Q in Qs]

        logger.info("Training completed successfully")

        return Ps, Qs, model

def create_phenotypic_admixture_trainer(min_k: int = 3,
                                      max_k: int = 12,
                                      epochs: int = 500,
                                      batch_size: int = 512,
                                      learning_rate: float = 1e-3,
                                      hidden_size: int = 64,
                                      device: torch.device = None,
                                      random_seed: int = 42) -> PhenotypicNeuralAdmixture:
    """
    Factory function to create phenotypic Neural ADMIXTURE trainer.

    Args:
        min_k: Minimum clusters
        max_k: Maximum clusters
        epochs: Training epochs
        batch_size: Batch size
        learning_rate: Learning rate
        hidden_size: Hidden layer size
        device: Computing device
        random_seed: Random seed

    Returns:
        PhenotypicNeuralAdmixture trainer
    """
    if device is None:
        device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    return PhenotypicNeuralAdmixture(
        min_k=min_k,
        max_k=max_k,
        epochs=epochs,
        batch_size=batch_size,
        learning_rate=learning_rate,
        hidden_size=hidden_size,
        device=device,
        random_seed=random_seed
    )


# Monkey patch function to replace the placeholder training
def patch_train_function():
    """
    Monkey patch to replace the original train function with phenotypic version.
    """
    def phenotypic_train(epochs: int, batch_size: int, learning_rate: float, K: int, seed: int,
                        data: torch.Tensor, device: torch.device, num_gpus: int, hidden_size: int,
                        master: bool, V: np.ndarray, pops: np.ndarray, min_k: int = None,
                        max_k: int = None, n_components: int = None) -> Tuple[List[np.ndarray], List[np.ndarray], object]:
        """
        Replacement train function for phenotypic data.
        """
        logger.info("Using phenotypic Neural ADMIXTURE training")

        # Create trainer
        trainer = create_phenotypic_admixture_trainer(
            min_k=min_k or K,
            max_k=max_k or K,
            epochs=epochs,
            batch_size=batch_size,
            learning_rate=learning_rate,
            hidden_size=hidden_size,
            device=device,
            random_seed=seed
        )

        # Convert uint8 back to float for training
        if data.dtype == torch.uint8:
            data = data.float() / 255.0

        # Ensure data is on correct device
        data = data.to(device)

        # Train model
        metadata = {}  # Could pass actual metadata if needed
        Ps, Qs, model = trainer.train_phenotypic_admixture(data, metadata)

        return Ps, Qs, model

    return phenotypic_train


if __name__ == "__main__":
    # Test the phenotypic adapter
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    # Create dummy phenotypic data
    n_samples, n_features = 1000, 10
    dummy_data = torch.rand(n_samples, n_features, device=device)

    # Create trainer
    trainer = create_phenotypic_admixture_trainer(
        min_k=3, max_k=5, epochs=10, device=device
    )

    # Test training
    try:
        Ps, Qs, model = trainer.train_phenotypic_admixture(dummy_data, {})
        print("Phenotypic training test successful!")
        print(f"Results: {len(Ps)} P matrices, {len(Qs)} Q matrices")
        for i, (P, Q) in enumerate(zip(Ps, Qs)):
            print(f"  K={i+3}: P shape {P.shape}, Q shape {Q.shape}")
    except Exception as e:
        print(f"Training test failed: {e}")