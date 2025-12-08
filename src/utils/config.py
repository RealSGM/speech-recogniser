class AudioConfig:
    def __init__(
        self,
        use_mel_filter_bank: bool = False,
        use_spectral_features: bool = False,
        include_noisy_data: bool = False,
        use_pre_emphasis: bool = False,
        use_delta_features: bool = False,
        frame_size: float = 0.025,
        frame_stride: float = 0.01,
        num_mel_filters: int = 40,
        num_ceps: int = 13,
    ):
        self.use_mel_filter_bank = use_mel_filter_bank
        self.use_spectral_features = use_spectral_features
        self.include_noisy_data = include_noisy_data
        self.use_pre_emphasis = use_pre_emphasis
        self.use_delta_features = use_delta_features
        self.frame_size = frame_size
        self.frame_stride = frame_stride
        self.num_mel_filters = num_mel_filters
        self.num_ceps = num_ceps


class VisualConfig:
    def __init__(
        self,
        roi_size: tuple = (96, 96),
    ):
        self.roi_size = roi_size


class TrainingConfig:
    def __init__(
        self,
        learning_rate: float = 0.0005,
        batch_size: int = 32,
        epochs: int = 25,

        conv_blocks: int = 3,
        base_filters: int = 64,
        kernel_size: tuple = (3, 3),
        pool_size: tuple = (2, 2),

        dense_units: int = 64,
        dropout_rate: float = 0.5,
        use_batch_norm: bool = False,
    ):
        self.learning_rate = learning_rate
        self.batch_size = batch_size
        self.epochs = epochs

        self.conv_blocks = conv_blocks
        self.base_filters = base_filters
        self.kernel_size = kernel_size
        self.pool_size = pool_size

        self.dense_units = dense_units
        self.dropout_rate = dropout_rate
        self.use_batch_norm = use_batch_norm


class IntegrationConfig:
    def __init__(
        self,
        integration_type: str = "early",
        audio_weight: float = 0.5,
        visual_weight: float = 0.5,
    ):
        self.integration_type = integration_type
        self.audio_weight = audio_weight
        self.visual_weight = visual_weight


class MainConfig:
    def __init__(
        self,
        test_name: str = "default_test",
        audio: 'AudioConfig' = AudioConfig(),
        visual: 'VisualConfig' = VisualConfig(),
        training: 'TrainingConfig' = TrainingConfig(),
        integration: 'IntegrationConfig' = IntegrationConfig(),
        start_time: float = 0.0,
        notes: str = "default",
    ):
        self.test_name = test_name
        self.audio = audio
        self.visual = visual
        self.training = training
        self.integration = integration
        self.start_time = start_time
        self.notes = notes


def get_configs() -> list[MainConfig]:
    return configs


late_integration = IntegrationConfig(
    integration_type="late",
    audio_weight=0.5,
    visual_weight=0.5,
)

audio_only = IntegrationConfig(
    integration_type="late",
    audio_weight=1.0,
    visual_weight=0.0,
)

visual_only = IntegrationConfig(
    integration_type="late",
    audio_weight=0.0,
    visual_weight=1.0,
)

configs = [
    # MainConfig(test_name="default"),
    # MainConfig(test_name="default_late_integration", integration=late_integration),
    # MainConfig(test_name="audio_only_test", integration=audio_only),
    MainConfig(test_name="visual_only_test", integration=visual_only),
]
