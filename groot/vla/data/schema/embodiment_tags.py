from enum import Enum


class EmbodimentTag(Enum):
    REAL_GR1_ARMS_ONLY = "real_gr1_arms_only"
    """
    The real GR1 robot embodiment with arms only.
    """

    REAL_GR1_ARMS_ONLY_ANNOTATED = "real_gr1_arms_only_annotated"
    """
    The real GR1 robot embodiment with arms only with annotations.
    """

    REAL_GR1_ARMS_WAIST = "real_gr1_arms_waist"
    """
    The real GR1 robot embodiment with arms and waist.
    """

    REAL_GR1_ARMS_WAIST_ANNOTATED = "real_gr1_arms_waist_annotated"
    """
    The real GR1 robot embodiment with arms and waist with annotations.
    """

    DEXMG_GR1_ARMS_ONLY_INSPIRE = "dexmg_gr1_arms_only_inspire"
    """
    The DEXMG GR1 dataset with arms only and inspire hand.
    """

    DEXMG_GR1_ARMS_ONLY_FOURIER = "dexmg_gr1_arms_only_fourier"
    """
    The DEXMG GR1 dataset with arms only and Fourier hand.
    """

    DEXMG_GR1_ARMS_WAIST_FOURIER = "dexmg_gr1_arms_waist_fourier"
    """
    The DEXMG GR1 dataset with arms and waist and Fourier hand.
    """

    ROBOCASA_SINGLE_ARM = "robocasa_single_arm"
    """
    The RoboCasa dataset with single arm.
    """

    ONE_X_EVE_GRIPPER = "onex_eve_gripper"
    """
    The OneX Eve Robot with gripper.
    """

    ROBOCASA_GR1_ARMS_ONLY_INSPIRE_HANDS = "robocasa_gr1_arms_only_inspire_hands"
    """
    The RoboCasa GR1 dataset with arms only and inspire hands.
    """

    ROBOCASA_GR1_ARMS_ONLY_FOURIER_HANDS = "robocasa_gr1_arms_only_fourier_hands"
    """
    The RoboCasa GR1 dataset with arms only and Fourier hands.
    """

    ROBOCASA_GR1_FIXED_LOWER_BODY_INSPIRE_HANDS = "robocasa_gr1_fixed_lower_body_inspire_hands"
    """
    The RoboCasa GR1 dataset with fixed lower body and inspire hands.
    """

    ROBOCASA_GR1_FIXED_LOWER_BODY_FOURIER_HANDS = "robocasa_gr1_fixed_lower_body_fourier_hands"
    """
    The RoboCasa GR1 dataset with fixed lower body and Fourier hands.
    """

    ROBOCASA_GR1_ARMS_WAIST_FOURIER_HANDS = "robocasa_gr1_arms_waist_fourier_hands"
    """
    The RoboCasa GR1 dataset with arms and waist and Fourier hands.
    """

    ROBOCASA_PANDA_OMRON = "robocasa_panda_omron"
    """
    The RoboCasa dataset with panda omron.
    """

    ROBOCASA_SINGLE_ARM_PANDA_OMRON = "robocasa_single_arm_panda_omron"
    """
    The RoboCasa dataset with single arm panda omron.
    """

    ROBOCASA_BIMANUAL_PANDA_PARALLEL_GRIPPER = "robocasa_bimanual_panda_parallel_gripper"
    """
    The dexmg bimanual panda dataset with parallel grippers.
    """

    ROBOCASA_BIMANUAL_PANDA_INSPIRE_HAND = "robocasa_bimanual_panda_inspire_hand"
    """
    The DEXMG bimanual panda dataset with inspire hands.
    """

    OXE_DROID = "oxe_droid"
    """
    The Open X-Embodiment droid dataset.
    """

    OXE_FRACTAL = "oxe_fractal"
    """
    The Open X-Embodiment fractal (RT-1) dataset.
    """

    OXE_LANGUAGE_TABLE = "oxe_language_table"
    """
    The Open X-Embodiment language table dataset.
    """

    OXE_BRIDGE = "oxe_bridge"
    """
    The Open X-Embodiment bridge dataset.
    """

    OXE_MUTEX = "oxe_mutex"
    """
    The Open X-Embodiment mutex dataset.
    """

    OXE_ROBOSET = "oxe_roboset"
    """
    The Open X-Embodiment Roboset dataset.
    """

    OXE_PLEX = "oxe_plex"
    """
    The Open X-Embodiment Plex RoboSuite dataset.
    """

    REAL_PANDA_SINGLE_ARM = "real_panda_single_arm"
    """
    The real single arm panda robot.
    """

    HOT3D_HANDS_ONLY = "hot3d_hands_only"
    """
    The HOT3D dataset with hands only.
    """

    GR1_UNIFIED = "gr1_unified"
    """
    The GR1 unified dataset.
    """

    GR1_ISAAC = "gr1_isaac"
    """
    The GR1 Isaac dataset (Shiwei).
    """

    LAPA = "lapa"
    """
    The datasets with LAPA actions.
    """
    AGIBOT = "agibot"

    YAM = "yam"

    DREAM = "dream"
    """
    The datasets with DREAM actions.
    """

    UNKNOWN = "unknown"

    GR1_UNIFIED_SEGMENTATION = "gr1_unified_segmentation"
    """
    The GR1 unified dataset with segmentation.
    """

    LANGUAGE_TABLE_SIM = "language_table_sim"
    """
    Simulated Language Table.
    """

    SIMPLER_ENV_GOOGLE = "simpler_env_google"
    """
    SimplerEnv Google.
    """

    SIMPLER_ENV_WIDOWX = "simpler_env_widowx"
    """
    SimplerEnv Widowx.
    """

    LIBERO_SIM = "libero_sim"
    """
    The Libero Sim dataset.
    """

    DROID_SIM = "droid_sim"
    """
    The Droid dataset in sim.
    """

    UNITREE_G1_UPPER_BODY = "unitree_g1_upper_body"
    """
    The Unitree G1 dataset.
    """

    UNITREE_G1_FULL_BODY = "unitree_g1_full_body"
    """
    The Unitree G1 dataset with full body.
    """

    UNITREE_G1_FULL_BODY_IN_SIM = "unitree_g1_full_body_in_sim"
    """
    The Unitree G1 dataset in sim, data is collected with 50Hz.
    """

    UNITREE_G1_FULL_BODY_WITH_HEIGHT = "unitree_g1_full_body_with_height"
    """
    The Unitree G1 dataset with full body and height command.
    """

    UNITREE_G1_FULL_BODY_WITH_HEIGHT_AND_EEF = "unitree_g1_full_body_with_height_and_eef"
    """
    The Unitree G1 dataset with full body and height command and eef command.
    """

    UNITREE_G1_FULL_BODY_WITH_HEIGHT_EEF_NAV_CMD = "unitree_g1_full_body_with_height_eef_nav_cmd"
    """
    The Unitree G1 dataset with full body and height command and eef command and navigate command.
    """

    UNITREE_G1_FULL_BODY_WITH_HEIGHT_NAV_CMD = "unitree_g1_full_body_with_height_nav_cmd"
    """
    The Unitree G1 dataset with full body and height command and navigate command.
    """

    UNITREE_G1_FULL_BODY_WITH_WAIST_HEIGHT_NAV_CMD = (
        "unitree_g1_full_body_with_waist_height_nav_cmd"
    )
    """
    The Unitree G1 dataset with full body and waist and height command and navigate command.
    """

    UNITREE_G1_FULL_BODY_WITH_WAIST_HEIGHT_NAV_CMD_AND_TASK_PROG = (
        "unitree_g1_full_body_with_waist_height_nav_cmd_and_task_prog"
    )
    """
    The Unitree G1 dataset with full body and waist and height command and navigate command and task progress.
    """

    UNITREE_G1_FULL_BODY_WITH_HEIGHT_NAV_CMD_IN_SIM = (
        "unitree_g1_full_body_with_height_nav_cmd_in_sim"
    )
    """
    The Unitree G1 dataset with full body and height command and navigate command in sim, data is collected with 50Hz.
    """

    G1_FIX_LOWER_RIGHT_HAND = "g1_fix_lower_right_hand"
    """
    The G1 robot with fixed lower body and right hand.
    """

    GR1_UNIFIED_OFFLINE_RL = "gr1_unified_offline_rl"
    """
    The GR1 robot with reward information for oflfine RL.
    """

    UNITREE_G1_FULL_BODY_WITH_HEIGHT_NAV_CMD_OAK_STEREO = (
        "unitree_g1_full_body_with_height_nav_cmd_oak_stereo"
    )
    """
    The Unitree G1 dataset with full body and height command and navigate command and oak stereo.
    """

    XDOF = "xdof"
    """
    The XDOF robot.
    """

    XDOF_H16 = "xdof_h16"
    """
    The XDOF robot with action horizon 16.
    """

    XDOF_OSS_DATA = "xdof_oss_data"
    """
    The XDOF data with conversions via jimmywu's processing pipeline.
    """

    SIM_BEHAVIOR_R1_PRO = "sim_behavior_r1_pro"
    """
    The sim BEHAVIOR Galaxea R1 Pro robot with grippers.
    """

    # ------------- Deprecated G1 Embodiments -------------
    # below are deprecated G1 embodiments. Why deprecated?
    # 1. The neck wasn't locked. So the the camera angles differe slightly between episodes and robots
    # 2. Video modality was called `rs_view`. It is now called `ego_view`
    # TODO: In future if we want to unify these embodiment tags, we can do so by renaming
    # video.rs_view -> video.ego_view. After that we can remove the below deprecated embodiment tags.
    DEPRECATED_UNITREE_G1_UPPER_BODY = "deprecated_unitree_g1_upper_body"
    """
    The Unitree G1 dataset.
    """

    DEPRECATED_UNITREE_G1_FULL_BODY = "deprecated_unitree_g1_full_body"
    """
    The Unitree G1 dataset with full body.
    """

    DEPRECATED_UNITREE_G1_FULL_BODY_WITH_HEIGHT = "deprecated_unitree_g1_full_body_with_height"
    """
    The Unitree G1 dataset with full body and height command.
    """

    DEPRECATED_UNITREE_G1_FULL_BODY_WITH_HEIGHT_AND_EEF = (
        "deprecated_unitree_g1_full_body_with_height_and_eef"
    )
    """
    The Unitree G1 dataset with full body and height command and eef command.
    """

    DEPRECATED_UNITREE_G1_FULL_BODY_WITH_HEIGHT_EEF_NAV_CMD = (
        "deprecated_unitree_g1_full_body_with_height_eef_nav_cmd"
    )
    """
    The Unitree G1 dataset with full body and height command and eef command and navigate command.
    """

    DEPRECATED_UNITREE_G1_FULL_BODY_WITH_HEIGHT_NAV_CMD = (
        "deprecated_unitree_g1_full_body_with_height_nav_cmd"
    )
    """
    The Unitree G1 dataset with full body and height command and navigate command.
    """

    GR1_UNIFIED_512 = "gr1_unified_512"
    """
    The GR1 unified dataset with 512*320 resolution without cropping.
    """

    R1_PRO = "r1_pro"
    """
    Real Galaxea R1 Pro.
    """

    R1_PRO_SINGLE_VIEW = "r1_pro_single-view"
    """
    Real Galaxea R1 Pro with single camera view.
    """

    MECKA_HANDS = "mecka_hands"
    """
    The Mecka robot with hands.
    """
