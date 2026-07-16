"""
Neural field architectures for learning EMG forward solutions.

Models take (x, y, z, condition) -> scalar potential u.

Architectures:
  - MLP: Standard ReLU/GELU network with optional positional encoding
  - SIREN: Sinusoidal representation network (Sitzmann et al., 2020)

Conditioning modes:
  - "concat": Condition concatenated at input (default, original behavior)
  - "film": Feature-wise Linear Modulation — spatial coords go through the
    main trunk; condition generates per-layer modulation via lightweight
    linear projections. Shares spatial features across all electrodes.
"""

from __future__ import annotations

import math
from typing import Optional

import torch
import torch.nn as nn


# ---------------------------------------------------------------------------
# Positional Encoding
# ---------------------------------------------------------------------------

class FourierFeatures(nn.Module):
    """Fourier feature positional encoding."""

    def __init__(self, in_dim: int, n_frequencies: int = 64, scale: float = 1.0):
        super().__init__()
        self.n_frequencies = n_frequencies
        B = torch.randn(in_dim, n_frequencies) * scale
        self.register_buffer("B", B)

    @property
    def out_dim(self) -> int:
        return self.n_frequencies * 2

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        proj = x @ self.B  # (..., n_freq)
        return torch.cat([torch.sin(proj), torch.cos(proj)], dim=-1)


# ---------------------------------------------------------------------------
# FiLM Layers
# ---------------------------------------------------------------------------

class FiLMGenerator(nn.Module):
    """Generates FiLM modulation parameters (gamma, beta) from condition."""

    def __init__(self, condition_dim: int, out_features: int):
        super().__init__()
        self.linear = nn.Linear(condition_dim, 2 * out_features)
        # Initialize to identity modulation: gamma=0, beta=0
        # so (1 + gamma) * h + beta = h at initialization
        nn.init.zeros_(self.linear.weight)
        nn.init.zeros_(self.linear.bias)

    def forward(self, condition: torch.Tensor):
        """
        Parameters
        ----------
        condition : (B, C) or (B, N, C)

        Returns
        -------
        gamma, beta : each (B, out_features) or (B, N, out_features)
        """
        params = self.linear(condition)
        gamma, beta = params.chunk(2, dim=-1)
        return gamma, beta


class FiLMSineLayer(nn.Module):
    """SIREN layer with FiLM conditioning."""

    def __init__(
        self,
        in_features: int,
        out_features: int,
        condition_dim: int,
        is_first: bool = False,
        omega_0: float = 30.0,
    ):
        super().__init__()
        self.omega_0 = omega_0
        self.is_first = is_first
        self.linear = nn.Linear(in_features, out_features)
        self.film = FiLMGenerator(condition_dim, out_features)
        self._init_weights()

    def _init_weights(self):
        with torch.no_grad():
            if self.is_first:
                bound = 1.0 / self.linear.in_features
            else:
                bound = math.sqrt(6.0 / self.linear.in_features) / self.omega_0
            self.linear.weight.uniform_(-bound, bound)

    def forward(self, x: torch.Tensor, condition: torch.Tensor) -> torch.Tensor:
        """
        Parameters
        ----------
        x : (B, N, in_features) — spatial features
        condition : (B, N, C) — condition (already expanded to match points)
        """
        h = torch.sin(self.omega_0 * self.linear(x))
        gamma, beta = self.film(condition)
        return (1 + gamma) * h + beta


class FiLMMLPLayer(nn.Module):
    """MLP layer with FiLM conditioning and GELU activation."""

    def __init__(
        self,
        in_features: int,
        out_features: int,
        condition_dim: int,
        activation: str = "gelu",
    ):
        super().__init__()
        self.linear = nn.Linear(in_features, out_features)
        self.film = FiLMGenerator(condition_dim, out_features)
        act_map = {"relu": nn.ReLU, "gelu": nn.GELU}
        self.act = act_map[activation]()

    def forward(self, x: torch.Tensor, condition: torch.Tensor) -> torch.Tensor:
        h = self.act(self.linear(x))
        gamma, beta = self.film(condition)
        return (1 + gamma) * h + beta


# ---------------------------------------------------------------------------
# MLP
# ---------------------------------------------------------------------------

class MLP(nn.Module):
    """Standard MLP with ReLU/GELU and optional Fourier features."""

    def __init__(
        self,
        in_dim: int = 6,
        hidden_dim: int = 256,
        n_layers: int = 8,
        out_dim: int = 1,
        activation: str = "gelu",
        use_fourier: bool = False,
        n_frequencies: int = 64,
        fourier_scale: float = 1.0,
        skip_connections: bool = True,
        conditioning_mode: str = "concat",
        condition_dim: int = 3,
    ):
        super().__init__()
        self.skip_connections = skip_connections
        self.conditioning_mode = conditioning_mode
        self._condition_dim = condition_dim

        act_fn = {"relu": nn.ReLU, "gelu": nn.GELU}[activation]

        # Determine spatial input dim
        if conditioning_mode == "film":
            spatial_in = in_dim - condition_dim  # just spatial coords (3)
        else:
            spatial_in = in_dim  # spatial + condition concatenated

        # Optional Fourier encoding on spatial coords
        if use_fourier:
            fourier_in = 3 if conditioning_mode == "film" else 3
            self.fourier = FourierFeatures(fourier_in, n_frequencies, fourier_scale)
            if conditioning_mode == "film":
                actual_in = self.fourier.out_dim
            else:
                actual_in = self.fourier.out_dim + (spatial_in - 3)
        else:
            self.fourier = None
            actual_in = spatial_in

        if conditioning_mode == "film":
            # FiLM: build layers with FiLM modulation
            self.film_layers = nn.ModuleList()
            self.final = nn.Linear(hidden_dim, out_dim)

            # First layer
            self.film_layers.append(
                FiLMMLPLayer(actual_in, hidden_dim, condition_dim, activation)
            )
            # Hidden layers
            for i in range(n_layers - 2):
                if skip_connections and i == (n_layers - 2) // 2:
                    self.film_layers.append(
                        FiLMMLPLayer(hidden_dim + actual_in, hidden_dim, condition_dim, activation)
                    )
                else:
                    self.film_layers.append(
                        FiLMMLPLayer(hidden_dim, hidden_dim, condition_dim, activation)
                    )
        else:
            # Concat mode: original behavior
            layers = [nn.Linear(actual_in, hidden_dim), act_fn()]
            for i in range(n_layers - 2):
                if skip_connections and i == (n_layers - 2) // 2:
                    layers.append(nn.Linear(hidden_dim + actual_in, hidden_dim))
                else:
                    layers.append(nn.Linear(hidden_dim, hidden_dim))
                layers.append(act_fn())
            layers.append(nn.Linear(hidden_dim, out_dim))

            self.layers = nn.ModuleList(
                [l for l in layers if isinstance(l, nn.Linear)]
            )
            self.acts = nn.ModuleList(
                [l for l in layers if not isinstance(l, nn.Linear)]
            )

        self._actual_in = actual_in
        self._skip_layer = (n_layers - 2) // 2 + 1 if skip_connections else -1

    def forward(self, coords: torch.Tensor, condition: torch.Tensor) -> torch.Tensor:
        """
        Parameters
        ----------
        coords : (B, N, 3) or (B, 3)
        condition : (B, C) — broadcast across points

        Returns
        -------
        u : (B, N, 1) or (B, 1)
        """
        squeeze = coords.dim() == 2
        if squeeze:
            coords = coords.unsqueeze(1)

        B, N, _ = coords.shape

        # Encode spatial coords
        if self.fourier is not None:
            enc = self.fourier(coords)  # (B, N, fourier_dim)
        else:
            enc = coords

        # Expand condition to match points
        if condition.dim() == 1:
            condition = condition.unsqueeze(0)
        cond_expanded = condition.unsqueeze(1).expand(B, N, -1)

        if self.conditioning_mode == "film":
            x = enc  # spatial-only input
            x_input = x

            for i, film_layer in enumerate(self.film_layers):
                if self.skip_connections and i == self._skip_layer:
                    x = torch.cat([x, x_input], dim=-1)
                x = film_layer(x, cond_expanded)

            x = self.final(x)
        else:
            x = torch.cat([enc, cond_expanded], dim=-1)  # (B, N, in_dim)
            x_input = x

            for i, linear in enumerate(self.layers):
                if self.skip_connections and i == self._skip_layer:
                    x = torch.cat([x, x_input], dim=-1)
                x = linear(x)
                if i < len(self.acts):
                    x = self.acts[i](x)

        if squeeze:
            x = x.squeeze(1)
        return x


# ---------------------------------------------------------------------------
# SIREN
# ---------------------------------------------------------------------------

class SineLayer(nn.Module):
    """Linear layer with sine activation (SIREN)."""

    def __init__(
        self,
        in_features: int,
        out_features: int,
        is_first: bool = False,
        omega_0: float = 30.0,
    ):
        super().__init__()
        self.omega_0 = omega_0
        self.is_first = is_first
        self.linear = nn.Linear(in_features, out_features)
        self._init_weights()

    def _init_weights(self):
        with torch.no_grad():
            if self.is_first:
                bound = 1.0 / self.linear.in_features
            else:
                bound = math.sqrt(6.0 / self.linear.in_features) / self.omega_0
            self.linear.weight.uniform_(-bound, bound)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return torch.sin(self.omega_0 * self.linear(x))


class SIREN(nn.Module):
    """
    Sinusoidal Representation Network.

    Good for smooth continuous fields like electric potentials.
    Reference: Sitzmann et al., "Implicit Neural Representations with
    Periodic Activation Functions", NeurIPS 2020.

    Supports two conditioning modes:
      - "concat": condition concatenated at input (default)
      - "film": FiLM modulation at every hidden layer, spatial-only input
    """

    def __init__(
        self,
        in_dim: int = 6,
        hidden_dim: int = 256,
        n_layers: int = 8,
        out_dim: int = 1,
        omega_0: float = 30.0,
        omega_hidden: float = 30.0,
        conditioning_mode: str = "concat",
        condition_dim: int = 3,
    ):
        super().__init__()
        self.conditioning_mode = conditioning_mode
        self._condition_dim = condition_dim

        if conditioning_mode == "film":
            spatial_in = in_dim - condition_dim  # just spatial coords (3)

            # FiLM SIREN: spatial-only input, condition modulates each layer
            self.film_layers = nn.ModuleList()
            self.film_layers.append(
                FiLMSineLayer(spatial_in, hidden_dim, condition_dim,
                              is_first=True, omega_0=omega_0)
            )
            for _ in range(n_layers - 2):
                self.film_layers.append(
                    FiLMSineLayer(hidden_dim, hidden_dim, condition_dim,
                                  omega_0=omega_hidden)
                )

            self.final = nn.Linear(hidden_dim, out_dim)
            with torch.no_grad():
                bound = math.sqrt(6.0 / hidden_dim) / omega_hidden
                self.final.weight.uniform_(-bound, bound)
        else:
            # Concat mode: original behavior
            layers = []
            layers.append(SineLayer(in_dim, hidden_dim, is_first=True, omega_0=omega_0))
            for _ in range(n_layers - 2):
                layers.append(SineLayer(hidden_dim, hidden_dim, omega_0=omega_hidden))

            self.final = nn.Linear(hidden_dim, out_dim)
            with torch.no_grad():
                bound = math.sqrt(6.0 / hidden_dim) / omega_hidden
                self.final.weight.uniform_(-bound, bound)

            self.layers = nn.ModuleList(layers)

    def forward(self, coords: torch.Tensor, condition: torch.Tensor) -> torch.Tensor:
        """
        Parameters
        ----------
        coords : (B, N, 3) or (B, 3)
        condition : (B, C)

        Returns
        -------
        u : (B, N, 1) or (B, 1)
        """
        squeeze = coords.dim() == 2
        if squeeze:
            coords = coords.unsqueeze(1)

        B, N, _ = coords.shape

        # Expand condition
        if condition.dim() == 1:
            condition = condition.unsqueeze(0)
        cond_expanded = condition.unsqueeze(1).expand(B, N, -1)

        if self.conditioning_mode == "film":
            x = coords  # spatial-only (B, N, 3)
            for film_layer in self.film_layers:
                x = film_layer(x, cond_expanded)
            x = self.final(x)
        else:
            x = torch.cat([coords, cond_expanded], dim=-1)
            for layer in self.layers:
                x = layer(x)
            x = self.final(x)

        if squeeze:
            x = x.squeeze(1)
        return x


# ---------------------------------------------------------------------------
# Unified wrapper
# ---------------------------------------------------------------------------

class NeuralField(nn.Module):
    """
    Wrapper that picks architecture by name and handles I/O consistently.

    Usage:
        model = NeuralField(arch="siren", condition_dim=3, hidden_dim=256)
        u_pred = model(batch)  # batch from PointCloudDataset

        # FiLM conditioning:
        model = NeuralField(arch="siren", condition_dim=3, conditioning_mode="film")
    """

    def __init__(
        self,
        arch: str = "siren",
        condition_dim: int = 3,
        hidden_dim: int = 256,
        n_layers: int = 8,
        conditioning_mode: str = "concat",
        **kwargs,
    ):
        super().__init__()
        in_dim = 3 + condition_dim
        if arch == "siren":
            self.net = SIREN(
                in_dim=in_dim,
                hidden_dim=hidden_dim,
                n_layers=n_layers,
                conditioning_mode=conditioning_mode,
                condition_dim=condition_dim,
                **kwargs,
            )
        elif arch == "mlp":
            self.net = MLP(
                in_dim=in_dim,
                hidden_dim=hidden_dim,
                n_layers=n_layers,
                conditioning_mode=conditioning_mode,
                condition_dim=condition_dim,
                **kwargs,
            )
        else:
            raise ValueError(f"Unknown architecture: {arch}")

        self.arch = arch
        self.condition_dim = condition_dim
        self.conditioning_mode = conditioning_mode

    def forward(
        self,
        batch: dict = None,
        coords: torch.Tensor = None,
        condition: torch.Tensor = None,
    ) -> torch.Tensor:
        """
        Forward pass. Either pass a batch dict or coords + condition directly.

        Returns
        -------
        u_pred : (B, N) predicted potential
        """
        if batch is not None:
            coords = batch["points"]
            condition = batch["condition"]

        u = self.net(coords, condition)  # (B, N, 1)
        return u.squeeze(-1)  # (B, N)

    def count_parameters(self) -> int:
        return sum(p.numel() for p in self.parameters() if p.requires_grad)
