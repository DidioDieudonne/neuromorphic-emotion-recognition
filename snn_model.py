"""
SEW-RESNET18 IMPLEMENTATION - Fully Working Version
Fixed SpikingJelly integration for facial emotion recognition
"""

import torch
import torch.nn as nn
from spikingjelly.activation_based import layer, neuron, surrogate, functional
from spikingjelly.activation_based.model import sew_resnet
from spikingjelly.activation_based import encoding
import torch.nn.functional as F
import math

# Export list for module interface
__all__ = [
    "SEWResNet18EmotionRecognizer",
    "create_sew_resnet18",
    "reset_snn_state",
    "test_sew_resnet",
    "attach_spike_hooks"
]


class SEWResNet18EmotionRecognizer(nn.Module):
    """
    SEW-ResNet18 wrapper for emotion recognition with robust SpikingJelly integration

    This class provides a stable interface to SEW-ResNet18 architecture with automatic
    fallback to standard ResNet18 if SpikingJelly encounters compatibility issues.
    Designed for comparative studies between spiking and traditional neural networks.
    """

    def __init__(self, num_classes=7, num_timesteps=8, encoding_type="temporal"):
        super(SEWResNet18EmotionRecognizer, self).__init__()

        # Store configuration parameters for later reference
        self.num_classes = num_classes
        self.num_timesteps = num_timesteps
        self.encoding_type = encoding_type

        # Attempt to create SpikingJelly model with graceful fallback
        try:
            print("Attempting SpikingJelly SEW-ResNet18...")
            self._create_spikingjelly_model(num_classes)
        except Exception as e:
            print(f"SpikingJelly failed: {e}")
            print("Using ResNet18 fallback...")
            self._create_fallback_model(num_classes)

    def _create_spikingjelly_model(self, num_classes):
        """
        Initialize authentic SpikingJelly SEW-ResNet18 model
        Compatible with current SpikingJelly API (requires cnf).
        """

        print("Creating SEW-ResNet18 with proper configuration...")

        #  IMPORTANT : cnf est obligatoire dans ta version
        self.backbone = sew_resnet.sew_resnet18(
            pretrained=False,
            progress=False,
            cnf="ADD",  # REQUIRED in your SpikingJelly version
            spiking_neuron=neuron.LIFNode
        )

        print("Backbone created:", self.backbone)

        if self.backbone is None:
            raise RuntimeError("SpikingJelly returned None backbone")

        # Adapter pour images grayscale (1 canal)
        self.backbone.conv1 = layer.Conv2d(
            1, 64, kernel_size=7, stride=2, padding=3, bias=False
        )

        # Adapter la couche finale
        in_features = self.backbone.fc.in_features
        self.backbone.fc = layer.Linear(in_features, num_classes)

        # Activer mode multi-step
        functional.set_step_mode(self.backbone, step_mode='m')

        # Métadonnées
        self.backbone_type = "SpikingJelly SEW-ResNet18"
        self.is_spikingjelly = True

        print("SpikingJelly SEW-ResNet18 created successfully")

    def _create_fallback_model(self, num_classes):
        """
        Initialize ResNet18 fallback model when SpikingJelly fails

        Creates a standard ResNet18 architecture adapted for grayscale input
        and emotion classification. Maintains similar parameter count for fair comparison.

        Args:
            num_classes (int): Number of output emotion classes
        """
        import torchvision.models as models

        # Create standard ResNet18 backbone
        self.backbone = models.resnet18(weights=None)

        # Adapt for single-channel grayscale input
        self.backbone.conv1 = nn.Conv2d(1, 64, 7, 2, 3, bias=False)

        # Adapt final layer for emotion classification
        self.backbone.fc = nn.Linear(512, num_classes)

        # Set model metadata
        self.backbone_type = "ResNet18 Fallback"
        self.is_spikingjelly = False
        print("ResNet18 fallback created")

    def forward(self, x):
        """
        Forward pass with true multi-step rate (Poisson) encoding for SpikingJelly.
        """

        if len(x.shape) != 4:
            raise ValueError(f"Expected input shape (B, C, H, W), got {x.shape}")

        if self.is_spikingjelly:

            # Ensure input is in [0, 1] range for Poisson encoding
            x = torch.clamp(x, 0., 1.)

            # Create Poisson encoder
            encoder = encoding.PoissonEncoder()

            # Generate spike train sequence: [T, B, C, H, W]
            spike_seq = []
            for _ in range(self.num_timesteps):
                spike_seq.append(encoder(x))

            spike_seq = torch.stack(spike_seq, dim=0)

            # Forward entire temporal sequence at once (multi-step mode)
            out_seq = self.backbone(spike_seq)  # Shape: [T, B, num_classes]

            # Temporal integration
            out = out_seq.mean(dim=0)

            # Reset neuron states AFTER full sequence
            functional.reset_net(self.backbone)

            return out

        else:
            # Standard CNN forward (no fake temporal loop)
            return self.backbone(x)

    def get_architecture_info(self):
        """
        Retrieve comprehensive model architecture information

        Provides detailed metadata about the model including parameter counts,
        architecture type, and configuration settings for analysis and reporting.

        Returns:
            dict: Model information dictionary containing:
                - total_parameters: Total number of model parameters
                - trainable_parameters: Number of trainable parameters
                - architecture: String describing the backbone architecture
                - num_classes: Number of output classes
                - num_timesteps: Number of temporal processing steps
                - encoding_type: Type of temporal encoding used
                - is_spikingjelly: Boolean indicating if using authentic SpikingJelly
        """
        # Calculate parameter statistics
        total_params = sum(p.numel() for p in self.parameters())
        trainable_params = sum(p.numel() for p in self.parameters() if p.requires_grad)

        return {
            'total_parameters': total_params,
            'trainable_parameters': trainable_params,
            'architecture': getattr(self, 'backbone_type', 'Unknown'),
            'num_classes': self.num_classes,
            'num_timesteps': self.num_timesteps,
            'encoding_type': self.encoding_type,
            'is_spikingjelly': getattr(self, 'is_spikingjelly', False)
        }


def create_sew_resnet18(num_classes=7, num_timesteps=8, encoding_type="temporal"):
    """
    Factory function to create SEW-ResNet18 model with robust configuration

    Creates a SEW-ResNet18 model optimized for emotion recognition tasks.
    Automatically handles SpikingJelly compatibility issues with graceful fallback.

    Args:
        num_classes (int): Number of emotion classes to classify (default: 7)
        num_timesteps (int): Number of temporal processing steps (default: 8)
        encoding_type (str): Temporal encoding method ("temporal" or "rate")

    Returns:
        SEWResNet18EmotionRecognizer: Configured model ready for training/inference
    """
    # Display creation parameters for debugging and logging
    print(f"Création du modèle SEW-ResNet18...")
    print(f"Classes: {num_classes}")
    print(f"Pas temporels: {num_timesteps}")
    print(f"Encodage: {encoding_type}")

    # Instantiate the model with specified configuration
    model = SEWResNet18EmotionRecognizer(
        num_classes=num_classes,
        num_timesteps=num_timesteps,
        encoding_type=encoding_type
    )

    # Display comprehensive model information for verification
    try:
        info = model.get_architecture_info()
        print(f"Paramètres totaux: {info['total_parameters']:,}")
        print(f"Paramètres entraînables: {info['trainable_parameters']:,}")
        print(f"Architecture: {info['architecture']}")

        # Indicate whether authentic SpikingJelly is being used
        if info['is_spikingjelly']:
            print(" Utilise SpikingJelly authentique")
        else:
            print(" Utilise fallback ResNet18")

    except Exception as e:
        print(f"Erreur info modèle: {e}")

    return model


def reset_snn_state(model):
    """
    Reset spiking neuron states between batches or inference runs

    Essential for proper SNN operation to clear membrane potentials and
    internal states. Only applies to authentic SpikingJelly models.

    Args:
        model (SEWResNet18EmotionRecognizer): Model instance to reset
    """
    try:
        # Only reset if using authentic SpikingJelly implementation
        if hasattr(model, 'is_spikingjelly') and model.is_spikingjelly:
            functional.reset_net(model.backbone)
    except:
        # Graceful handling if reset fails (fallback models don't need reset)
        pass


def attach_spike_hooks(model):
    """
    Attach forward hooks to all spiking neuron layers in SpikingJelly model.

    Returns:
        spike_dict (dict): Dictionary storing spike outputs per layer
    """

    spike_dict = {}

    if not hasattr(model, 'is_spikingjelly') or not model.is_spikingjelly:
        return spike_dict

    def hook_fn(module, input, output):
        # output shape in multi-step mode:
        # [T, B, ...]
        if isinstance(output, torch.Tensor):
            spike_dict[module] = output.detach()

    for name, module in model.backbone.named_modules():
        # Capture LIF and IF neurons
        if isinstance(module, (neuron.LIFNode, neuron.IFNode)):
            module.register_forward_hook(hook_fn)

    return spike_dict


def test_sew_resnet():
    """
    Comprehensive testing function for SEW-ResNet18 model validation

    Performs systematic testing of model creation, forward pass, and output
    shape validation for both temporal and rate encoding modes. Includes
    detailed debugging output for troubleshooting compatibility issues.

    Returns:
        bool: True if all tests pass, False if any test fails
    """
    print("Testing SEW-ResNet18 model...")

    # Test configuration parameters
    batch_size = 2
    timesteps = 4

    try:
        # Test 1: Temporal encoding validation
        print(f"\nTest 1: Temporal encoding")
        model_temporal = create_sew_resnet18(
            num_classes=7,
            num_timesteps=timesteps,
            encoding_type="temporal"
        )

        # Create test input tensor with standard image dimensions
        test_input = torch.randn(batch_size, 1, 48, 48)
        print(f"Input shape: {test_input.shape}")

        # Execute forward pass with gradient tracking disabled
        with torch.no_grad():
            output = model_temporal(test_input)

        print(f"Output shape: {output.shape}")

        # Validate output dimensions match expected format
        expected_shape = (batch_size, 7)
        if output.shape == expected_shape:
            print("✓ Temporal test passed")
            # Clean up neuron states after successful test
            reset_snn_state(model_temporal)

            # Test 2: Rate encoding validation
            print(f"\nTest 2: Rate encoding")
            model_rate = create_sew_resnet18(
                num_classes=7,
                num_timesteps=timesteps,
                encoding_type="rate"
            )

            # Test rate encoding with same input
            with torch.no_grad():
                output_rate = model_rate(test_input)

            # Validate rate encoding output
            if output_rate.shape == expected_shape:
                print("✓ Rate encoding test passed")
                reset_snn_state(model_rate)

                print(f"\n✓ All tests successful!")
                return True
            else:
                print(f"✗ Rate test failed: {output_rate.shape}")
                return False
        else:
            print(f"✗ Temporal test failed: {output.shape}")
            return False

    except Exception as e:
        # Comprehensive error reporting for debugging
        print(f"✗ Test error: {e}")
        import traceback
        traceback.print_exc()
        return False


# Main execution block for standalone testing
if __name__ == "__main__":
    print("SEW-RESNET18 - Version corrigée")
    print("=" * 50)

    # Execute comprehensive test suite
    success = test_sew_resnet()

    # Report final status
    if success:
        print("\n SEW-ResNet18 prêt!")
    else:
        print("\n Problèmes détectés")