# MoCoLUS — Motion-Compensated Lung Ultrasound
# src package init

try:
    from .lung_us_generator import (
        LungUSArtifactSynthesizer,
        DualIMUReading,
        IMUPose,
        GridConfig,
        PipelineParams,
        PATHOLOGY_CLASSES,
        N_CLASSES,
    )
    from .poc_image_stack import (
        POCImageStackGenerator,
        ZoneResolver,
        TemporalStackGenerator,
        SimulatorExporter,
        LungZone,
        Pathology,
        SCENARIOS,
    )
    from .lung_us_dataset import (
        LungUSDatasetBuilder,
        LungUSZeaDataset,
        DatasetSpec,
        IMUSimulator,
    )
    from .simulator_bridge import (
        SimulatorSession,
        ChestWallMapping,
        bmode_to_rgba,
        gui_angles_to_imu_pose,
        imu_pose_to_gui_angles,
    )
except ImportError:
    pass  # keras not installed — legacy modules unavailable
from .clinical_frames import (
    ClinicalFrameGenerator,
    ClinicalTemporalStack,
    ClinicalPathology,
    CLINICAL_SCENARIOS,
)
