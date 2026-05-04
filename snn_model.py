"""
SEW-ResNet18 for Facial Emotion Recognition
Spiking Neural Network implementation using SpikingJelly

Reference:
    Fang et al. (2021) "Deep Residual Learning in Spiking Neural Networks"
    NeurIPS 2021. https://arxiv.org/abs/2102.04159
"""

import torch
import torch.nn as nn
from spikingjelly.activation_based import layer, neuron, surrogate, functional
from spikingjelly.activation_based.model import sew_resnet

__all__ = [
    "SEWResNet18EmotionRecognizer",
    "create_sew_resnet18",
    "reset_snn_state",
    "test_sew_resnet",
    "attach_spike_hooks"
]


class SEWResNet18EmotionRecognizer(nn.Module):
    """
    SEW-ResNet18 for facial emotion recognition on static grayscale images.

    Implements the Spike-Element-Wise ResNet18 architecture (Fang et al., 2021)
    with direct encoding for static image datasets, ATan surrogate gradient
    (Fang et al., ICCV 2021), and multi-step temporal processing.

    Args:
        num_classes  (int): Number of emotion classes. Default: 7.
        num_timesteps (int): Number of simulation timesteps T. Default: 8.
        encoding_type (str): Encoding label for logging. Default: 'direct'.
    """

    def __init__(self, num_classes=7, num_timesteps=8, encoding_type="direct"):
        super().__init__()
        self.num_classes   = num_classes
        self.num_timesteps = num_timesteps
        self.encoding_type = encoding_type

        try:
            self._build_spikingjelly(num_classes)
        except Exception as e:
            print(f"[SEWResNet18] SpikingJelly init failed: {e}")
            print("[SEWResNet18] Falling back to standard ResNet18.")
            self._build_fallback(num_classes)

    def _build_spikingjelly(self, num_classes):
        """Build SEW-ResNet18 with SpikingJelly backend."""
        self.backbone = sew_resnet.sew_resnet18(
            pretrained=False,
            progress=False,
            cnf="ADD",
            spiking_neuron=neuron.LIFNode,
            surrogate_function=surrogate.ATan(alpha=2.0),
            detach_reset=True
        )

        # Adapt first conv for single-channel (grayscale) input
        self.backbone.conv1 = layer.Conv2d(
            1, 64, kernel_size=7, stride=2, padding=3, bias=False
        )

        # Adapt classifier head
        self.backbone.fc = layer.Linear(
            self.backbone.fc.in_features, num_classes
        )

        # Enable multi-step mode for temporal processing
        functional.set_step_mode(self.backbone, step_mode='m')

        self.backbone_type  = "SEW-ResNet18 (SpikingJelly)"
        self.is_spikingjelly = True

    def _build_fallback(self, num_classes):
        """Fallback to standard ResNet18 when SpikingJelly is unavailable."""
        import torchvision.models as models
        self.backbone = models.resnet18(weights=None)
        self.backbone.conv1 = nn.Conv2d(1, 64, 7, 2, 3, bias=False)
        self.backbone.fc    = nn.Linear(512, num_classes)
        self.backbone_type  = "ResNet18 (fallback)"
        self.is_spikingjelly = False

    def forward(self, x):
        """
        Forward pass using direct (constant) encoding.

        Input x of shape (B, C, H, W) is min-max normalized to [0, 1]
        and replicated across T timesteps before being fed to the SNN.
        LIF neurons generate spikes through their internal membrane dynamics.
        Output logits are obtained by averaging over timesteps.

        Reference: Fang et al. (2021), Section 3.2 — direct encoding
        for static image classification datasets.
        """
        if x.ndim != 4:
            raise ValueError(f"Expected (B, C, H, W), got {x.shape}")

        if self.is_spikingjelly:
            # Per-sample min-max normalization to [0, 1]
            x_min  = x.flatten(1).min(dim=1)[0].view(-1, 1, 1, 1)
            x_max  = x.flatten(1).max(dim=1)[0].view(-1, 1, 1, 1)
            x_norm = (x - x_min) / (x_max - x_min + 1e-8)

            # Direct encoding: replicate over T timesteps → [T, B, C, H, W]
            spike_seq = x_norm.unsqueeze(0).repeat(self.num_timesteps, 1, 1, 1, 1)

            # Multi-step forward pass
            out_seq = self.backbone(spike_seq)   # [T, B, num_classes]
            out     = out_seq.mean(dim=0)        # temporal average

            functional.reset_net(self.backbone)
            return out

        return self.backbone(x)

    def get_architecture_info(self):
        """Return model metadata as a dictionary."""
        total     = sum(p.numel() for p in self.parameters())
        trainable = sum(p.numel() for p in self.parameters() if p.requires_grad)
        return {
            'total_parameters'    : total,
            'trainable_parameters': trainable,
            'architecture'        : getattr(self, 'backbone_type', 'Unknown'),
            'num_classes'         : self.num_classes,
            'num_timesteps'       : self.num_timesteps,
            'encoding_type'       : self.encoding_type,
            'is_spikingjelly'     : getattr(self, 'is_spikingjelly', False)
        }


def create_sew_resnet18(num_classes=7, num_timesteps=8, encoding_type="direct"):
    """
    Instantiate and return a SEWResNet18EmotionRecognizer.

    Args:
        num_classes   (int): Number of output classes. Default: 7.
        num_timesteps (int): Number of simulation timesteps. Default: 8.
        encoding_type (str): Encoding label for logging. Default: 'direct'.

    Returns:
        SEWResNet18EmotionRecognizer: Model ready for training or inference.
    """
    model = SEWResNet18EmotionRecognizer(num_classes, num_timesteps, encoding_type)
    info  = model.get_architecture_info()

    print(f"[create_sew_resnet18] Architecture  : {info['architecture']}")
    print(f"[create_sew_resnet18] Parameters    : {info['total_parameters']:,}")
    print(f"[create_sew_resnet18] Timesteps     : {info['num_timesteps']}")
    print(f"[create_sew_resnet18] SpikingJelly  : {info['is_spikingjelly']}")

    return model


def reset_snn_state(model):
    """Reset membrane potentials of all LIF neurons in the model."""
    if getattr(model, 'is_spikingjelly', False):
        try:
            functional.reset_net(model.backbone)
        except Exception:
            pass


def attach_spike_hooks(model):
    """
    Attach forward hooks to all LIF/IF layers.

    Returns:
        dict: {module: spike_tensor} populated after each forward pass.

    Note: Prefer the inline hook pattern (make_hook + register_forward_hook)
    for analysis pipelines, as reset_net() invalidates module-keyed dicts.
    """
    spike_dict = {}

    if not getattr(model, 'is_spikingjelly', False):
        return spike_dict

    def hook_fn(module, input, output):
        if isinstance(output, torch.Tensor):
            spike_dict[module] = output.detach()

    for _, module in model.backbone.named_modules():
        if isinstance(module, (neuron.LIFNode, neuron.IFNode)):
            module.register_forward_hook(hook_fn)

    return spike_dict


def test_sew_resnet(batch_size=2, timesteps=4, num_classes=7):
    """
    Validate SEW-ResNet18 forward pass shape and output consistency.

    Returns:
        bool: True if all tests pass.
    """
    print("Running SEW-ResNet18 validation tests...")

    model = create_sew_resnet18(num_classes, timesteps)
    x     = torch.randn(batch_size, 1, 48, 48)

    with torch.no_grad():
        out = model(x)

    expected = (batch_size, num_classes)
    passed   = out.shape == expected

    print(f"  Input  : {x.shape}")
    print(f"  Output : {out.shape}  —  {'PASS ✓' if passed else 'FAIL ✗'}")

    reset_snn_state(model)
    return passed


if __name__ == "__main__":
    success = test_sew_resnet()
    print("\nSEW-ResNet18 ready." if success else "\nTest failed.")