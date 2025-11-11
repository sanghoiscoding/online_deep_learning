from pathlib import Path

import torch
import torch.nn as nn

HOMEWORK_DIR = Path(__file__).resolve().parent
INPUT_MEAN = [0.2788, 0.2657, 0.2629]
INPUT_STD = [0.2064, 0.1944, 0.2252]


class MLPPlanner(nn.Module):
    def __init__(
        self,
        n_track: int = 10,
        n_waypoints: int = 3,
        hidden_dim: int = 256,
    ):
        """
        Args:
            n_track (int): number of points in each side of the track
            n_waypoints (int): number of waypoints to predict
            hidden_dim (int): hidden dimension for MLP layers
        """
        super().__init__()

        self.n_track = n_track
        self.n_waypoints = n_waypoints
        
        # Input: concatenated left and right track boundaries
        # Shape: (b, n_track * 2, 2) -> flatten to (b, n_track * 2 * 2)
        input_dim = n_track * 2 * 2  # 2 tracks * 2 coordinates = 40
        output_dim = n_waypoints * 2  # 3 waypoints * 2 coordinates = 6
        
        self.mlp = nn.Sequential(
            nn.Linear(input_dim, hidden_dim),
            nn.ReLU(),
            nn.Dropout(0.1),
            nn.Linear(hidden_dim, hidden_dim),
            nn.ReLU(),
            nn.Dropout(0.1),
            nn.Linear(hidden_dim, hidden_dim // 2),
            nn.ReLU(),
            nn.Linear(hidden_dim // 2, output_dim)
        )

    def forward(
        self,
        track_left: torch.Tensor,
        track_right: torch.Tensor,
        **kwargs,
    ) -> torch.Tensor:
        """
        Predicts waypoints from the left and right boundaries of the track.

        During test time, your model will be called with
        model(track_left=..., track_right=...), so keep the function signature as is.

        Args:
            track_left (torch.Tensor): shape (b, n_track, 2)
            track_right (torch.Tensor): shape (b, n_track, 2)

        Returns:
            torch.Tensor: future waypoints with shape (b, n_waypoints, 2)
        """
        # Concatenate left and right track boundaries
        # Shape: (b, n_track, 2) + (b, n_track, 2) -> (b, n_track * 2, 2)
        track_combined = torch.cat([track_left, track_right], dim=1)
        
        # Flatten for MLP input
        # Shape: (b, n_track * 2, 2) -> (b, n_track * 2 * 2)
        track_flat = track_combined.view(track_combined.shape[0], -1)
        
        # Pass through MLP
        # Shape: (b, n_track * 2 * 2) -> (b, n_waypoints * 2)
        waypoints_flat = self.mlp(track_flat)
        
        # Reshape to waypoints format
        # Shape: (b, n_waypoints * 2) -> (b, n_waypoints, 2)
        waypoints = waypoints_flat.view(-1, self.n_waypoints, 2)
        
        return waypoints


class TransformerPlanner(nn.Module):
    def __init__(
        self,
        n_track: int = 10,
        n_waypoints: int = 3,
        d_model: int = 64,
        num_layers: int = 2,
        num_heads: int = 4,
    ):
        super().__init__()

        self.n_track = n_track
        self.n_waypoints = n_waypoints
        self.d_model = d_model

        # Query embeddings for waypoints (like Perceiver)
        self.query_embed = nn.Embedding(n_waypoints, d_model)
        
        # Input projection for track boundaries
        self.input_projection = nn.Linear(2, d_model)  # (x, y) -> d_model
        
        # Transformer decoder layers for cross-attention
        decoder_layer = nn.TransformerDecoderLayer(
            d_model=d_model,
            nhead=num_heads,
            dim_feedforward=d_model * 4,
            dropout=0.1,
            batch_first=True
        )
        self.transformer_decoder = nn.TransformerDecoder(
            decoder_layer, 
            num_layers=num_layers
        )
        
        # Output projection to waypoint coordinates
        self.output_projection = nn.Linear(d_model, 2)

    def forward(
        self,
        track_left: torch.Tensor,
        track_right: torch.Tensor,
        **kwargs,
    ) -> torch.Tensor:
        """
        Predicts waypoints from the left and right boundaries of the track.

        During test time, your model will be called with
        model(track_left=..., track_right=...), so keep the function signature as is.

        Args:
            track_left (torch.Tensor): shape (b, n_track, 2)
            track_right (torch.Tensor): shape (b, n_track, 2)

        Returns:
            torch.Tensor: future waypoints with shape (b, n_waypoints, 2)
        """
        batch_size = track_left.shape[0]
        
        # Combine left and right track boundaries
        # Shape: (b, n_track * 2, 2)
        track_combined = torch.cat([track_left, track_right], dim=1)
        
        # Project track points to embedding space
        # Shape: (b, n_track * 2, 2) -> (b, n_track * 2, d_model)
        track_features = self.input_projection(track_combined)
        
        # Get query embeddings for waypoints
        # Shape: (n_waypoints, d_model) -> (b, n_waypoints, d_model)
        query_indices = torch.arange(self.n_waypoints, device=track_left.device)
        queries = self.query_embed(query_indices).unsqueeze(0).expand(batch_size, -1, -1)
        
        # Apply transformer decoder (cross-attention)
        # queries attend to track_features
        # Shape: (b, n_waypoints, d_model)
        waypoint_features = self.transformer_decoder(
            tgt=queries,  # target (queries)
            memory=track_features,  # source (track boundaries)
        )
        
        # Project to waypoint coordinates
        # Shape: (b, n_waypoints, d_model) -> (b, n_waypoints, 2)
        waypoints = self.output_projection(waypoint_features)
        
        return waypoints


class PatchEmbedding(nn.Module):
    def __init__(self, h: int = 96, w: int = 128, patch_size: int = 8, in_channels: int = 3, embed_dim: int = 64):
        """
        Convert image to sequence of patch embeddings using a simple approach

        This is provided as a helper for implementing the Vision Transformer Planner.
        You can use this directly in your ViTPlanner implementation.

        Args:
            h: height of input image
            w: width of input image
            patch_size: size of each patch
            in_channels: number of input channels (3 for RGB)
            embed_dim: embedding dimension
        """
        super().__init__()
        self.h = h
        self.w = w
        self.patch_size = patch_size
        self.num_patches = (h // patch_size) * (w // patch_size)

        # Linear projection of flattened patches
        self.projection = nn.Linear(patch_size * patch_size * in_channels, embed_dim)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """
        Args:
            x: (B, C, H, W) input images

        Returns:
            (B, num_patches, embed_dim) patch embeddings
        """
        B, C, H, W = x.shape
        p = self.patch_size

        # Reshape into patches: (B, C, H//p, p, W//p, p) -> (B, C, H//p, W//p, p, p)
        x = x.reshape(B, C, H // p, p, W // p, p).permute(0, 1, 2, 4, 3, 5)
        # Flatten patches: (B, C, H//p, W//p, p*p) -> (B, H//p * W//p, C * p * p)
        num_patches = (H // p) * (W // p)
        x = x.reshape(B, num_patches, C * p * p)

        # Linear projection
        return self.projection(x)
    

class TransformerBlock(nn.Module):
    def __init__(self, embed_dim: int = 256, num_heads: int = 8, mlp_ratio: float = 4.0, dropout: float = 0.1):
        """
        A single Transformer encoder block with multi-head attention and MLP.

        You can use the one you implemented in Homework 3.

        Hint: A transformer block typically consists of:
        1. Layer normalization
        2. Multi-head self-attention (use torch.nn.MultiheadAttention with batch_first=True)
        3. Residual connection
        4. Layer normalization
        5. MLP (Linear -> GELU -> Dropout -> Linear -> Dropout)
        6. Residual connection

        Args:
            embed_dim: embedding dimension
            num_heads: number of attention heads
            mlp_ratio: ratio of MLP hidden dimension to embedding dimension
            dropout: dropout probability
        """
        super().__init__()
        
        # Layer normalization
        self.norm1 = nn.LayerNorm(embed_dim)
        self.norm2 = nn.LayerNorm(embed_dim)
        
        # Multi-head self-attention
        self.attention = nn.MultiheadAttention(
            embed_dim=embed_dim,
            num_heads=num_heads,
            dropout=dropout,
            batch_first=True
        )
        
        # MLP block
        mlp_hidden_dim = int(embed_dim * mlp_ratio)
        self.mlp = nn.Sequential(
            nn.Linear(embed_dim, mlp_hidden_dim),
            nn.GELU(),
            nn.Dropout(dropout),
            nn.Linear(mlp_hidden_dim, embed_dim),
            nn.Dropout(dropout)
        )

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """
        Args:
            x: (batch_size, sequence_length, embed_dim) input sequence

        Returns:
            (batch_size, sequence_length, embed_dim) output sequence
        """
        # Self-attention with residual connection
        # Layer norm -> attention -> residual
        x_norm = self.norm1(x)
        attn_output, _ = self.attention(x_norm, x_norm, x_norm)
        x = x + attn_output
        
        # MLP with residual connection
        # Layer norm -> MLP -> residual
        x_norm = self.norm2(x)
        mlp_output = self.mlp(x_norm)
        x = x + mlp_output
        
        return x


class ViTPlanner(torch.nn.Module):
    def __init__(
        self,
        n_waypoints: int = 3,
        patch_size: int = 8,
        embed_dim: int = 256,
        num_layers: int = 6,
        num_heads: int = 8,
        mlp_ratio: float = 4.0,
        dropout: float = 0.1,
    ):
        """
        Vision Transformer (ViT) based planner that predicts waypoints from images.

        Args:
            n_waypoints (int): number of waypoints to predict

        Hint - you can add more arguments to the constructor such as:
            patch_size: int, size of image patches
            embed_dim: int, embedding dimension
            num_layers: int, number of transformer layers
            num_heads: int, number of attention heads

        Note: You can use the provided PatchEmbedding and TransformerBlock classes.
        The input images are of size (96, 128).

        Hint: A typical ViT architecture consists of:
        1. Patch embedding layer to convert image into sequence of patches
        2. Positional embeddings (learnable parameters) added to patch embeddings
        3. Multiple transformer encoder blocks
        4. Final normalization layer
        5. Output projection to predict waypoints

        Hint: For this task, you can either:
        - Use a classification token ([CLS]) approach like in standard ViT as global image representation
        - Use learned query embeddings (similar to TransformerPlanner)
        - Average pool over all patch features
        """
        super().__init__()

        self.n_waypoints = n_waypoints
        self.embed_dim = embed_dim
        
        # Image normalization
        self.register_buffer("input_mean", torch.as_tensor(INPUT_MEAN), persistent=False)
        self.register_buffer("input_std", torch.as_tensor(INPUT_STD), persistent=False)
        
        # Patch embedding
        self.patch_embedding = PatchEmbedding(
            h=96, w=128, patch_size=patch_size, 
            in_channels=3, embed_dim=embed_dim
        )
        
        # Positional embeddings
        # +1 for CLS token
        num_patches = self.patch_embedding.num_patches
        self.pos_embedding = nn.Parameter(torch.randn(1, num_patches + 1, embed_dim))
        
        # CLS token for global representation
        self.cls_token = nn.Parameter(torch.randn(1, 1, embed_dim))
        
        # Transformer blocks
        self.transformer_blocks = nn.ModuleList([
            TransformerBlock(
                embed_dim=embed_dim,
                num_heads=num_heads,
                mlp_ratio=mlp_ratio,
                dropout=dropout
            ) for _ in range(num_layers)
        ])
        
        # Final layer norm
        self.norm = nn.LayerNorm(embed_dim)
        
        # Output head to predict waypoints
        self.head = nn.Sequential(
            nn.Linear(embed_dim, embed_dim // 2),
            nn.GELU(),
            nn.Dropout(dropout),
            nn.Linear(embed_dim // 2, n_waypoints * 2)  # n_waypoints * 2 coordinates
        )

    def forward(self, image: torch.Tensor, **kwargs) -> torch.Tensor:
        """
        Args:
            image (torch.FloatTensor): shape (b, 3, 96, 128) and vals in [0, 1]

        Returns:
            torch.FloatTensor: future waypoints with shape (b, n_waypoints, 2)

        Hint: The typical forward pass consists of:
        1. Normalize input image
        2. Convert image to patch embeddings
        3. Add positional embeddings
        4. Pass through transformer blocks
        5. Extract features for prediction (e.g., [CLS] token or average pooling)
        6. Project to waypoint coordinates
        """
        x = image
        x = (x - self.input_mean[None, :, None, None]) / self.input_std[None, :, None, None]
        
        batch_size = x.shape[0]
        
        # Convert image to patch embeddings
        # Shape: (b, 3, 96, 128) -> (b, num_patches, embed_dim)
        patch_embeddings = self.patch_embedding(x)
        
        # Add CLS token
        # Shape: (1, 1, embed_dim) -> (b, 1, embed_dim)
        cls_tokens = self.cls_token.expand(batch_size, -1, -1)
        
        # Concatenate CLS token with patch embeddings
        # Shape: (b, 1, embed_dim) + (b, num_patches, embed_dim) -> (b, num_patches + 1, embed_dim)
        x = torch.cat([cls_tokens, patch_embeddings], dim=1)
        
        # Add positional embeddings
        x = x + self.pos_embedding
        
        # Pass through transformer blocks
        for transformer_block in self.transformer_blocks:
            x = transformer_block(x)
        
        # Final layer normalization
        x = self.norm(x)
        
        # Extract CLS token for global representation
        # Shape: (b, num_patches + 1, embed_dim) -> (b, embed_dim)
        cls_features = x[:, 0]  # First token is CLS
        
        # Project to waypoint coordinates
        # Shape: (b, embed_dim) -> (b, n_waypoints * 2)
        waypoints_flat = self.head(cls_features)
        
        # Reshape to waypoints format
        # Shape: (b, n_waypoints * 2) -> (b, n_waypoints, 2)
        waypoints = waypoints_flat.view(batch_size, self.n_waypoints, 2)
        
        return waypoints


MODEL_FACTORY = {
    "mlp_planner": MLPPlanner,
    "transformer_planner": TransformerPlanner,
    "vit_planner": ViTPlanner,
}


def load_model(
    model_name: str,
    with_weights: bool = False,
    **model_kwargs,
) -> torch.nn.Module:
    """
    Called by the grader to load a pre-trained model by name
    """
    m = MODEL_FACTORY[model_name](**model_kwargs)

    if with_weights:
        model_path = HOMEWORK_DIR / f"{model_name}.th"
        assert model_path.exists(), f"{model_path.name} not found"

        try:
            m.load_state_dict(torch.load(model_path, map_location="cpu"))
        except RuntimeError as e:
            raise AssertionError(
                f"Failed to load {model_path.name}, make sure the default model arguments are set correctly"
            ) from e

    # limit model sizes since they will be zipped and submitted
    model_size_mb = calculate_model_size_mb(m)

    if model_size_mb > 20:
        raise AssertionError(f"{model_name} is too large: {model_size_mb:.2f} MB")

    return m


def save_model(model: torch.nn.Module) -> str:
    """
    Use this function to save your model in train.py
    """
    model_name = None

    for n, m in MODEL_FACTORY.items():
        if type(model) is m:
            model_name = n

    if model_name is None:
        raise ValueError(f"Model type '{str(type(model))}' not supported")

    output_path = HOMEWORK_DIR / f"{model_name}.th"
    torch.save(model.state_dict(), output_path)

    return output_path


def calculate_model_size_mb(model: torch.nn.Module) -> float:
    """
    Naive way to estimate model size
    """
    return sum(p.numel() for p in model.parameters()) * 4 / 1024 / 1024
