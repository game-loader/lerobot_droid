"""DINOv3 Q-Planning configuration; FastWAM itself stays frozen."""

from dataclasses import dataclass


@dataclass
class FastWAMQConfig:
    """Q-network, replay-update, and planning options independent of the BC configuration."""

    action_dim: int = 7
    chunk_size: int = 32
    gamma: float = 0.99
    camera_keys: tuple[str, ...] = ("observation.images.image", "observation.images.image2")
    image_size: tuple[int, int] = (224, 224)
    # Official Meta HF conversion; a local pretrained directory works too.
    dino_model: str = "facebook/dinov3-vitl16-pretrain-lvd1689m"
    freeze_dino: bool = False
    text_model: str = "google/t5-v1_1-base"
    text_max_length: int = 128
    use_text: bool = True
    dim_model: int = 1024
    n_heads: int = 16
    dim_feedforward: int = 4096
    n_decoder_layers: int = 18
    dropout: float = 0.1
    qk_norm: bool = False
    qk_norm_eps: float = 1e-6
    gradient_checkpointing: bool = True
    v_min: float = -0.01
    v_max: float = 1.01
    num_bins: int = 101
    hl_gauss_sigma: float = 0.0075
    target_tau: float = 0.005
    learning_rate: float = 3e-4
    dino_learning_rate: float = 9e-5
    weight_decay: float = 1e-4
    batch_size: int = 8
    gradient_accumulation_steps: int = 1
    amp_dtype: str = "bfloat16"
    grad_clip_norm: float = 10.0
    num_candidates: int = 64
    candidate_batch_size: int = 8
    q_temperature: float = 1.0
    # Paper averages all candidates; the released code also offers top-k (default there: 16).
    n_elites: int = 0
    fastwam_inference_steps: int = 3
    execution_steps: int = 10

    def __post_init__(self):
        """Restore tuple-valued fields when loading JSON configurations."""
        self.camera_keys = tuple(self.camera_keys)
        self.image_size = tuple(self.image_size)
