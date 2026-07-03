"""
BioRetEcho: Biologically-Inspired Low-Light Enhancement with Retinal Echo and Ganglion Cell Feedback
PyTorch implementation
"""
import torch
import torch.nn as nn
import torch.nn.functional as F
import numpy as np


class RetinalEchoFilter(nn.Module):
    """
    Retinal Echo Filter (REF) Module
    Inspired by bat echolocation mechanism.
    I_f = I ⊙ (1 + w_echo) + β · EchoFilter(I)
    """
    def __init__(self, beta=0.1, sigma_g=1.0):
        super().__init__()
        self.beta = beta
        self.sigma_g = sigma_g

        # Sobel operators for edge extraction
        sobel_x = torch.tensor([[-1, 0, 1], [-2, 0, 2], [-1, 0, 1]], dtype=torch.float32).unsqueeze(0).unsqueeze(0)
        sobel_y = torch.tensor([[-1, -2, -1], [0, 0, 0], [1, 2, 1]], dtype=torch.float32).unsqueeze(0).unsqueeze(0)
        self.register_buffer('sobel_x', sobel_x)
        self.register_buffer('sobel_y', sobel_y)

        # Dynamic echo weight predictor (lightweight CNN)
        self.weight_predictor = nn.Sequential(
            nn.Conv2d(3, 16, 3, padding=1),
            nn.ReLU(inplace=True),
            nn.Conv2d(16, 3, 3, padding=1),
            nn.Sigmoid()
        )

    def echo_filter(self, x):
        """Gaussian blur + Sobel edge extraction + normalization"""
        # Gaussian blur (5x5, sigma=1.0)
        b, c, h, w = x.shape
        x_blur = F.gaussian_blur(x, kernel_size=5, sigma=self.sigma_g)

        # Sobel edge detection
        edges = []
        for i in range(c):
            ch = x_blur[:, i:i+1, :, :]
            edge_x = F.conv2d(ch, self.sobel_x, padding=1)
            edge_y = F.conv2d(ch, self.sobel_y, padding=1)
            edge = torch.sqrt(edge_x**2 + edge_y**2 + 1e-6)
            edges.append(edge)
        edges = torch.cat(edges, dim=1)

        # Normalize to [0, 1]
        edges = (edges - edges.min()) / (edges.max() - edges.min() + 1e-6)
        return edges

    def forward(self, x):
        """
        Args:
            x: Input low-light image, normalized to [0, 1], shape (B, 3, H, W)
        Returns:
            I_f: Pre-filtered feature, shape (B, 3, H, W)
        """
        # Dynamic echo weight (higher weight for darker regions)
        w_echo = self.weight_predictor(x)  # (B, 3, H, W)

        # Echo filter
        echo_feat = self.echo_filter(x)  # (B, 3, H, W)

        # Combine: I_f = I ⊙ (1 + w_echo) + β · EchoFilter(I)
        I_f = x * (1 + w_echo) + self.beta * echo_feat
        return torch.clamp(I_f, 0, 1)


class CenterSurroundConv(nn.Module):
    """Center-surround differential convolution for ON/OFF bipolar simulation"""
    def __init__(self, in_ch, out_ch, kernel_size=3):
        super().__init__()
        self.center = nn.Conv2d(in_ch, out_ch, kernel_size, padding=kernel_size//2, bias=True)
        self.surround = nn.Conv2d(in_ch, out_ch, kernel_size, padding=kernel_size//2, bias=True)

    def forward(self, x):
        c = self.center(x)
        s = self.surround(x)
        return c * torch.sigmoid(s)  # Element-wise interaction


class RGCsGRU(nn.Module):
    """
    Spatio-Temporal Dual-Dimensional RGCs-GRU Unit
    Simulates retinal ganglion cell spike dynamics.
    """
    def __init__(self, in_ch=64, hidden_ch=64, T=3, alpha=0.1):
        super().__init__()
        self.T = T
        self.alpha = alpha
        self.hidden_ch = hidden_ch

        # Spatial: Center-surround differential convolution
        self.conv_surround = nn.Conv2d(in_ch, hidden_ch, 3, padding=1)
        self.conv_center = nn.Conv2d(in_ch, hidden_ch, 3, padding=1)

        # Temporal: GRU gates
        self.conv_z = nn.Conv2d(in_ch + hidden_ch, hidden_ch, 3, padding=1)  # Update gate
        self.conv_r = nn.Conv2d(in_ch + hidden_ch, hidden_ch, 3, padding=1)  # Reset gate
        self.conv_h = nn.Conv2d(in_ch + hidden_ch, hidden_ch, 3, padding=1)  # Candidate hidden

        # Dynamic threshold predictor (Sobel contrast gradient)
        self.theta_conv = nn.Sequential(
            nn.Conv2d(in_ch, 16, 3, padding=1),
            nn.ReLU(inplace=True),
            nn.Conv2d(16, hidden_ch, 3, padding=1),
            nn.Sigmoid()
        )

    def forward(self, x_t):
        """
        Args:
            x_t: Pre-filtered feature from REF, shape (B, C, H, W)
        Returns:
            spike manifold: List of spike tensors for t=1..T
        """
        b, c, h, w = x_t.shape
        h_prev = torch.zeros(b, self.hidden_ch, h, w, device=x_t.device)

        spikes_list = []

        for t in range(self.T):
            # Spatial interaction: F_spatial = Conv_surround ⊙ Conv_center
            f_surround = self.conv_surround(x_t)
            f_center = self.conv_center(x_t)
            f_spatial = f_surround * torch.sigmoid(f_center)

            # GRU gates
            x_h = torch.cat([x_t, h_prev], dim=1)
            z_t = torch.sigmoid(self.conv_z(x_h))  # Update gate
            r_t = torch.sigmoid(self.conv_r(x_h))  # Reset gate

            x_rh = torch.cat([x_t, r_t * h_prev], dim=1)
            h_tilde = torch.tanh(self.conv_h(x_rh))  # Candidate hidden

            # Hidden state update
            h_t = (1 - z_t) * h_prev + z_t * h_tilde

            # Dynamic threshold: θ_t = 0.5 + α · ∇_contrast(x_t)
            theta_base = 0.5
            theta_var = self.alpha * self.theta_conv(x_t)
            theta_t = theta_base + theta_var

            # Spike generation: spikes_t = ReLU(h_t - θ_t)
            spikes = F.relu(h_t - theta_t)
            spikes_list.append(spikes)

            h_prev = h_t

        return spikes_list  # List of T tensors


class ONOFFLateralInhibition(nn.Module):
    """
    ON/OFF Lateral Inhibition Branch
    Bipolar pulse-driven R/S separation logic.
    """
    def __init__(self, ch=64, epsilon=1e-6):
        super().__init__()
        self.epsilon = epsilon

        # Reflection (R) and Illumination (S) estimation networks
        self.decom_r = nn.Sequential(
            nn.Conv2d(ch, ch, 3, padding=1),
            nn.ReLU(inplace=True),
            nn.Conv2d(ch, 3, 3, padding=1),
            nn.Sigmoid()
        )

        self.illum_s = nn.Sequential(
            nn.Conv2d(ch, ch, 3, padding=1),
            nn.ReLU(inplace=True),
            nn.Conv2d(ch, 3, 3, padding=1),
            nn.Sigmoid()
        )

        # Gamma adaptation predictor
        self.gamma_pred = nn.Sequential(
            nn.AdaptiveAvgPool2d(1),
            nn.Flatten(),
            nn.Linear(ch, 16),
            nn.ReLU(inplace=True),
            nn.Linear(16, 1),
            nn.Sigmoid()
        )

    def forward(self, enc_f, spike_manifold):
        """
        Args:
            enc_f: Encoder feature, shape (B, C, H, W)
            spike_manifold: List of spike tensors from RGCs-GRU
        Returns:
            R: Reflection component, S: Illumination component, gamma
        """
        # Aggregate spikes from T steps (mean fusion)
        s = torch.stack(spike_manifold, dim=0).mean(dim=0)  # (B, C, H, W)

        # ON/OFF bipolar response
        ON = F.relu(s)
        OFF = F.relu(-s)

        # R and S decomposition
        R = self.decom_r(enc_f) * (ON + OFF + self.epsilon)
        S = self.illum_s(enc_f) * (1 - OFF / (ON + OFF + self.epsilon))
        S = torch.clamp(S, 1e-3, 1.0)

        # Adaptive gamma
        gamma = self.gamma_pred(enc_f).view(-1, 1, 1, 1) * 2.0 + 1.0  # Range [1, 3]

        return R, S, gamma


class Encoder(nn.Module):
    """Simple CNN encoder"""
    def __init__(self, in_ch=3, base_ch=64):
        super().__init__()
        self.conv1 = nn.Sequential(
            nn.Conv2d(in_ch, base_ch, 3, padding=1),
            nn.ReLU(inplace=True),
            nn.Conv2d(base_ch, base_ch, 3, padding=1),
            nn.ReLU(inplace=True)
        )
        self.pool1 = nn.MaxPool2d(2)  # /2

        self.conv2 = nn.Sequential(
            nn.Conv2d(base_ch, base_ch*2, 3, padding=1),
            nn.ReLU(inplace=True),
            nn.Conv2d(base_ch*2, base_ch*2, 3, padding=1),
            nn.ReLU(inplace=True)
        )
        self.pool2 = nn.MaxPool2d(2)  # /4

    def forward(self, x):
        x1 = self.conv1(x)
        x = self.pool1(x1)
        x2 = self.conv2(x)
        x = self.pool2(x2)
        return x, x1, x2


class Decoder(nn.Module):
    """Simple CNN decoder"""
    def __init__(self, base_ch=64):
        super().__init__()
        self.up1 = nn.Upsample(scale_factor=2, mode='bilinear', align_corners=False)
        self.conv1 = nn.Sequential(
            nn.Conv2d(base_ch*2 + 3, base_ch*2, 3, padding=1),
            nn.ReLU(inplace=True),
            nn.Conv2d(base_ch*2, base_ch, 3, padding=1),
            nn.ReLU(inplace=True)
        )

        self.up2 = nn.Upsample(scale_factor=2, mode='bilinear', align_corners=False)
        self.conv2 = nn.Sequential(
            nn.Conv2d(base_ch + base_ch + 3, base_ch, 3, padding=1),
            nn.ReLU(inplace=True),
            nn.Conv2d(base_ch, 3, 3, padding=1),
            nn.Sigmoid()
        )

    def forward(self, x, x1, x2, R, S, gamma):
        """
        Args:
            x: Deepest feature
            x1, x2: Skip connections
            R, S: Reflection and Illumination
            gamma: Adaptive gamma
        Returns:
            Enhanced image
        """
        # Initial reconstruction: R ⊙ S^γ
        x_recon = R * (S ** gamma)
        x_recon = torch.clamp(x_recon, 0, 1)

        # Decode with skip connections
        x = self.up1(x)
        x = torch.cat([x, x2, F.interpolate(x_recon, scale_factor=2, mode='bilinear')], dim=1)
        x = self.conv1(x)

        x = self.up2(x)
        x = torch.cat([x, x1, F.interpolate(x_recon, scale_factor=4, mode='bilinear')], dim=1)
        x = self.conv2(x)

        return x


class BioRetEcho(nn.Module):
    """
    BioRetEcho: Encode-Feedback-Decode Framework
    """
    def __init__(self, T=3, alpha=0.1, beta=0.1, epsilon=1e-6):
        super().__init__()
        self.T = T

        # Encoder-Filter stage
        self.ref = RetinalEchoFilter(beta=beta)
        self.encoder = Encoder(in_ch=3, base_ch=64)

        # Feedback core: RGCs-GRU
        self.rgc_gru = RGCsGRU(in_ch=128, hidden_ch=64, T=T, alpha=alpha)

        # ON/OFF lateral inhibition + R/S separation
        self.onoff = ONOFFLateralInhibition(ch=128, epsilon=epsilon)

        # Decoder
        self.decoder = Decoder(base_ch=64)

    def forward(self, x_low):
        """
        Args:
            x_low: Low-light input, shape (B, 3, H, W), range [0, 1]
        Returns:
            enhanced: Enhanced image, shape (B, 3, H, W)
            R: Reflection component
            S: Illumination component
            spikes: Spike manifold from RGCs-GRU
        """
        # REF pre-filtering
        I_f = self.ref(x_low)

        # Encode
        feat, x1, x2 = self.encoder(I_f)

        # RGCs-GRU feedback (spatio-temporal pulse generation)
        spike_manifold = self.rgc_gru(feat)

        # ON/OFF lateral inhibition + R/S separation
        R, S, gamma = self.onoff(feat, spike_manifold)

        # Decode
        enhanced = self.decoder(feat, x1, x2, R, S, gamma)

        return enhanced, R, S, spike_manifold


class PseudoLabelGenerator(nn.Module):
    """
    Self-Generated Pseudo-Label Training Framework
    Generates pseudo low-light labels from normal-light images.
    """
    def __init__(self, noise_level=0.1):
        super().__init__()
        self.noise_level = noise_level

    def forward(self, x_normal):
        """
        Args:
            x_normal: Normal-light image, (B, 3, H, W), [0, 1]
        Returns:
            x_pseudo_low: Pseudo low-light image
        """
        # Simulate non-uniform illumination degradation
        b, c, h, w = x_normal.shape

        # Random illumination map (non-uniform)
        illum = torch.rand(b, 1, h//4, w//4, device=x_normal.device) * 0.5 + 0.1  # [0.1, 0.6]
        illum = F.interpolate(illum, size=(h, w), mode='bilinear', align_corners=False)
        illum = illum.repeat(1, 3, 1, 1)

        # Gamma degradation
        gamma = torch.rand(b, 1, 1, 1, device=x_normal.device) * 1.5 + 1.5  # [1.5, 3.0]
        x_degraded = x_normal ** gamma

        # Apply illumination reduction
        x_degraded = x_degraded * illum

        # Add shot/read noise (Gaussian-Poisson composite)
        noise = torch.randn_like(x_normal) * self.noise_level
        x_pseudo_low = x_degraded + noise

        return torch.clamp(x_pseudo_low, 0, 1)


if __name__ == "__main__":
    # Quick test
    model = BioRetEcho(T=3)
    x = torch.randn(2, 3, 400, 600)
    out, R, S, spikes = model(x)
    print(f"Input: {x.shape}, Output: {out.shape}, R: {R.shape}, S: {S.shape}, Spikes: {len(spikes)}")

    # Count parameters
    total = sum(p.numel() for p in model.parameters())
    print(f"Total parameters: {total/1e3:.1f}K")
