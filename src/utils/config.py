class AudioConfig:
    def __init__(
        self,
        use_spectral_features: bool = False,
        include_noisy_data: bool = True,
        use_pre_emphasis: bool = True,
        pre_emphasis_coeff: float = 0.97,
        use_delta_features: bool = True,
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
        self.pre_emphasis_coeff = pre_emphasis_coeff


class VisualConfig:
	def __init__(
		self,
		roi_size=(96,96),
		coeff_size=8,
		use_deltas=False,
		use_motion_stats=False,
		use_lbp=False,
		frame_stride=1,
		pad_px=0,
		lip_landmarks=None,
	):
		self.roi_size = roi_size
		self.coeff_size = coeff_size
		self.use_deltas = use_deltas
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


# Training Configs ------------------------------


default_model = TrainingConfig()


optimum_model = TrainingConfig(
    batch_size=16,
    base_filters=64,
    dense_units=128,
    conv_blocks=2,
    dropout_rate=0.5
)


# Audio Configs ---------------------------------


optimum_audio = AudioConfig() # Already optimised


# Integration Configs ----------------------------


early_integration = IntegrationConfig()

audio_only = IntegrationConfig(
    integration_type="late",
    audio_weight=1.0,
    visual_weight=0.0,
)


late_integration = IntegrationConfig(
    integration_type="late",
    audio_weight=0.5,
    visual_weight=0.5,
)


visual_only = IntegrationConfig(
    integration_type="late",
    audio_weight=0.0,
    visual_weight=1.0,
)


# Visual Configs --------------------------------


## Hyperparameter Testing
use_delta = VisualConfig(use_deltas=True)
use_motion_stats = VisualConfig(use_motion_stats=True)
use_lbp = VisualConfig(use_lbp=True)
roi_small = VisualConfig(roi_size=(64,64))
roi_large = VisualConfig(roi_size=(128,128))
coeff_size_16 = VisualConfig(coeff_size=16)
coeff_size_32 = VisualConfig(coeff_size=32)
frame_stride_2 = VisualConfig(frame_stride=2)
frame_stride_3 = VisualConfig(frame_stride=3)
pad_px_2 = VisualConfig(pad_px=2)
pad_px_4 = VisualConfig(pad_px=4)
lip_landmarks_minimal = VisualConfig(lip_landmarks=[61,146,91,181,84,17,314,405])
lip_landmarks_extended = VisualConfig(lip_landmarks=[
        61,146,91,181,84,17,314,405,321,375,
        291,308,324,318,402,317,14,87,178,88,
        95,185,40,39,37,0,267,269,270,409,
        415,310,311,312,13,82,81,42,183,78
    ]
)


optimum_visual= VisualConfig(
    roi_size=(64,64),
    coeff_size=16,
    frame_stride=1,
    pad_px=2,
    lip_landmarks=[61,146,91,181,84,17,314,405]
)

statistically_best = VisualConfig(
    roi_size=(64,64),
    coeff_size=16,
    pad_px=2,
    lip_landmarks=[61,146,91,181,84,17,314,405],
    use_deltas=True,   
    use_motion_stats=False,
    use_lbp=True,
)


configs = [
    # MainConfig(test_name="roi_small", visual=roi_small, integration=visual_only, training=optimum_model),
    # MainConfig(test_name="roi_large", visual=roi_large, integration=visual_only, training=optimum_model),
    # MainConfig(test_name="coeff_size_16", visual=coeff_size_16, integration=visual_only, training=optimum_model),
    # MainConfig(test_name="coeff_size_32", visual=coeff_size_32, integration=visual_only, training=optimum_model),
    # MainConfig(test_name="frame_stride_2", visual=frame_stride_2, integration=visual_only, training=optimum_model),
    # MainConfig(test_name="frame_stride_3", visual=frame_stride_3, integration=visual_only, training=optimum_model),
    # MainConfig(test_name="pad_px_2", visual=pad_px_2, integration=visual_only, training=optimum_model),
    # MainConfig(test_name="pad_px_4", visual=pad_px_4, integration=visual_only, training=optimum_model),
    # MainConfig(test_name="lip_landmarks_minimal", visual=lip_landmarks_minimal, integration=visual_only, training=optimum_model),
    # MainConfig(test_name="lip_landmarks_extended", visual=lip_landmarks_extended, integration=visual_only, training=optimum_model),
    MainConfig(test_name="use_delta", visual=use_delta, integration=visual_only, training=optimum_model),
    MainConfig(test_name="use_motion_stats", visual=use_motion_stats, integration=visual_only, training=optimum_model),
    MainConfig(test_name="use_lbp", visual=use_lbp, integration=visual_only, training=optimum_model),
    
    MainConfig(test_name="baseline_visual", integration=visual_only,),
    MainConfig(test_name="baseline_audio", integration=audio_only),
    MainConfig(test_name="baseline_early_integration", integration=early_integration),
    MainConfig(test_name="baseline_late_integration", integration=late_integration),
    
    # MainConfig(test_name="complete_early_statistically_best_visual", visual=statistically_best, integration=early_integration, training=optimum_model),
    # MainConfig(test_name="complete_late_statistically_best_visual", visual=statistically_best, integration=late_integration, training=optimum_model),
]
