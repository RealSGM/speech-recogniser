class AudioConfig:
    def __init__(
        self,
        use_spectral_features: bool = False,
        include_noisy_data: bool = False,
        use_pre_emphasis: bool = False,
        use_delta_features: bool = False,
        frame_size: float = 0.025,
        frame_stride: float = 0.01,
        num_mel_filters: int = 40,
        num_ceps: int = 13,
    ):
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
		roi_size=(96,96),
		coeff_size=8,
		use_deltas=True,
		use_shape=True,
		use_motion_stats=True,
		use_lbp=False,
		frame_stride=1,
		pad_px=0,
		lip_landmarks=None,
	):
		self.roi_size = roi_size
		self.coeff_size = coeff_size
		self.use_deltas = use_deltas
		self.use_shape = use_shape
		self.use_motion_stats = use_motion_stats
		self.use_lbp = use_lbp
		self.frame_stride = max(1, frame_stride)
		self.pad_px = pad_px
		self.lip_landmarks = lip_landmarks or [
			61,146,91,181,84,17,314,405,321,375,
			291,308,324,318,402,317,14,87,178,88
		]



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

visual_simple = VisualConfig(
    use_deltas=False,
    use_shape=False,
    use_motion_stats=False,
    use_lbp=False,
)

visual_full = VisualConfig(
    use_deltas=True,
    use_shape=True,
    use_motion_stats=True,
    use_lbp=True,
    pad_px=8,
    )

configs = [
    # MainConfig(test_name="default"),
    # MainConfig(test_name="default_late_integration", integration=late_integration),
    # MainConfig(test_name="audio_only_test", integration=audio_only),
    MainConfig(test_name="visual_simple_test", visual=visual_simple, integration=visual_only),
    MainConfig(test_name="visual_full_test", visual=visual_full, integration=visual_only),
]
