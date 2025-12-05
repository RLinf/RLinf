import numpy as np

from robosuite.models.robots.manipulators.manipulator_model import ManipulatorModel
from robosuite.utils.mjcf_utils import xml_path_completion


class MountedPanda(ManipulatorModel):
    """
    Panda is a sensitive single-arm robot designed by Franka.
    Args:
        idn (int or str): Number or some other unique identification string for this robot instance
    """

    def __init__(self, idn=0):
        super().__init__(xml_path_completion("robots/panda/robot.xml"), idn=idn)

        # Set joint damping
        self.set_joint_attribute(
            attrib="damping", values=np.array((0.1, 0.1, 0.1, 0.1, 0.1, 0.01, 0.01))
        )

    @property
    def default_mount(self):
        return "RethinkMount"

    @property
    def default_gripper(self):
        return "PandaGripper"

    @property
    def default_controller_config(self):
        return "default_panda"

    @property
    def init_qpos(self):
        return np.array(
            [0, -1.61037389e-01, 0.00, -2.44459747e00, 0.00, 2.22675220e00, np.pi / 4]
        )
        # return np.array(
        #     [0.5, -1.61037389e-01, 0.00, -2.44459747e00, 0.00, 2.22675220e00, np.pi / 4]
        # )

    @property
    def base_xpos_offset(self):
        return {
            "bins": (-0.5, -0.1, 0),
            "empty": (-0.6, 0, 0),
            "table": lambda table_length: (-0.16 - table_length / 2, 0, 0),
            "study_table": lambda table_length: (-0.25 - table_length / 2, 0, 0),
            "kitchen_table": lambda table_length: (-0.16 - table_length / 2, 0, 0),
        }

    @property
    def top_offset(self):
        return np.array((0, 0, 1.0))

    @property
    def _horizontal_radius(self):
        return 0.5

    @property
    def arm_type(self):
        return "single"

# class MountedPanda1(ManipulatorModel):
#     """
#     Panda is a sensitive single-arm robot designed by Franka.
#     Args:
#         idn (int or str): Number or some other unique identification string for this robot instance
#     """

#     def __init__(self, idn=0):
#         super().__init__(xml_path_completion("robots/panda/robot.xml"), idn=idn)

#         # Set joint damping
#         self.set_joint_attribute(
#             attrib="damping", values=np.array((0.1, 0.1, 0.1, 0.1, 0.1, 0.01, 0.01))
#         )

#     @property
#     def default_mount(self):
#         return "RethinkMount"

#     @property
#     def default_gripper(self):
#         return "PandaGripper"

#     @property
#     def default_controller_config(self):
#         return "default_panda"

#     @property
#     def init_qpos(self):
#         return np.array(
#             [0.5, -1.61037389e-01, 0.00, -2.44459747e00, 0.00, 2.22675220e00, np.pi / 4]
#         )

#     @property
#     def base_xpos_offset(self):
#         return {
#             "bins": (-0.5, -0.1, 0),
#             "empty": (-0.6, 0, 0),
#             "table": lambda table_length: (-0.16 - table_length / 2, 0, 0),
#             "study_table": lambda table_length: (-0.25 - table_length / 2, 0, 0),
#             "kitchen_table": lambda table_length: (-0.16 - table_length / 2, 0, 0),
#         }

#     @property
#     def top_offset(self):
#         return np.array((0, 0, 1.0))

#     @property
#     def _horizontal_radius(self):
#         return 0.5

#     @property
#     def arm_type(self):
#         return "single"


class MountedPanda1(MountedPanda):
    """
    Panda Robot New Init State 1
    """

    @property
    def init_qpos(self):
        return np.array([ 0.02097405, -0.16687568,  0.02734903, -2.38028663, -0.00988726,
  2.21686563,  0.85208136])


class MountedPanda2(MountedPanda):
    """
    Panda Robot New Init State 2
    """

    @property
    def init_qpos(self):
        return np.array([ 0.03347899, -0.181518  ,  0.02366894, -2.46481386, -0.02031725,
  2.2373077 ,  0.70193217])


class MountedPanda3(MountedPanda):
    """
    Panda Robot New Init State 3
    """

    @property
    def init_qpos(self):
        return np.array([-0.05631059, -0.17939348, -0.03306425, -2.43433875, -0.02964279,
  2.18064702,  0.83324482])


class MountedPanda4(MountedPanda):
    """
    Panda Robot New Init State 4
    """

    @property
    def init_qpos(self):
        return np.array([-0.01149166, -0.15760031, -0.07251745, -2.4723057 ,  0.00564579,
  2.16816843,  0.8045206 ])


class MountedPanda5(MountedPanda):
    """
    Panda Robot New Init State 5
    """

    @property
    def init_qpos(self):
        return np.array([-0.02445108, -0.17291179, -0.02449455, -2.36919408, -0.00054945,
  2.18369442,  0.8188827 ])


class MountedPanda6(MountedPanda):
    """
    Panda Robot New Init State 6
    """

    @property
    def init_qpos(self):
        return np.array([-0.04384856, -0.15353572, -0.0703847 , -2.4923014 ,  0.00707059,
  2.25327541,  0.79155313])


class MountedPanda7(MountedPanda):
    """
    Panda Robot New Init State 7
    """

    @property
    def init_qpos(self):
        return np.array([-0.0056058 , -0.17563275, -0.0716682 , -2.47949038, -0.02232848,
  2.27799394,  0.80205433])


class MountedPanda8(MountedPanda):
    """
    Panda Robot New Init State 8
    """

    @property
    def init_qpos(self):
        return np.array([-0.07124198, -0.14794161, -0.01556064, -2.47195094,  0.02471698,
  2.26841345,  0.82302989])


class MountedPanda9(MountedPanda):
    """
    Panda Robot New Init State 9
    """

    @property
    def init_qpos(self):
        return np.array([-0.04585562, -0.17793304,  0.01810054, -2.39129278, -0.02618252,
  2.21660762,  0.724947  ])


class MountedPanda10(MountedPanda):
    """
    Panda Robot New Init State 10
    """

    @property
    def init_qpos(self):
        return np.array([-0.05105416, -0.12635875,  0.05788439, -2.44767087,  0.04283084,
  2.24218684,  0.75786442])


class MountedPanda11(MountedPanda):
    """
    Panda Robot New Init State 11
    """

    @property
    def init_qpos(self):
        return np.array([ 0.0102252 , -0.11752073, -0.00101365, -2.400328  , -0.07412213,
  2.25000681,  0.78786104])


class MountedPanda12(MountedPanda):
    """
    Panda Robot New Init State 12
    """

    @property
    def init_qpos(self):
        return np.array([-0.0115717 , -0.15748621, -0.07691969, -2.45309886,  0.01382039,
  2.28394727,  0.76534091])


class MountedPanda13(MountedPanda):
    """
    Panda Robot New Init State 13
    """

    @property
    def init_qpos(self):
        return np.array([-0.05213404, -0.19339216,  0.05902782, -2.42339863, -0.0341605 ,
  2.25984919,  0.79165801])


class MountedPanda14(MountedPanda):
    """
    Panda Robot New Init State 14
    """

    @property
    def init_qpos(self):
        return np.array([ 0.04849305, -0.19618411, -0.01640367, -2.46422749, -0.07326761,
  2.2415768 ,  0.79846731])


class MountedPanda15(MountedPanda):
    """
    Panda Robot New Init State 15
    """

    @property
    def init_qpos(self):
        return np.array([ 0.00029413, -0.17453103, -0.08141324, -2.46879332, -0.01971321,
  2.18060458,  0.77612088])


class MountedPanda16(MountedPanda):
    """
    Panda Robot New Init State 16
    """

    @property
    def init_qpos(self):
        return np.array([ 0.01474826, -0.09218974,  0.00637226, -2.43519663, -0.00271735,
  2.15671515,  0.78443038])


class MountedPanda17(MountedPanda):
    """
    Panda Robot New Init State 17
    """

    @property
    def init_qpos(self):
        return np.array([ 0.00202208, -0.07833998, -0.00645806, -2.43447375, -0.00116536,
  2.18751665,  0.82376568])


class MountedPanda18(MountedPanda):
    """
    Panda Robot New Init State 18
    """

    @property
    def init_qpos(self):
        return np.array([ 0.02257575, -0.13728775, -0.0273031 , -2.40248051, -0.04208864,
  2.24437177,  0.85116357])


class MountedPanda19(MountedPanda):
    """
    Panda Robot New Init State 19
    """

    @property
    def init_qpos(self):
        return np.array([-0.0438487 , -0.18610605,  0.00441133, -2.46688515, -0.06864421,
  2.22978732,  0.73837249])


class MountedPanda20(MountedPanda):
    """
    Panda Robot New Init State 20
    """

    @property
    def init_qpos(self):
        return np.array([ 0.01877035, -0.19747782,  0.06143005, -2.47564091, -0.01276458,
  2.25899512,  0.73661412])


class MountedPanda21(MountedPanda):
    """
    Panda Robot New Init State 21
    """

    @property
    def init_qpos(self):
        return np.array([ 0.00886241, -0.11010785, -0.06263156, -2.43740367,  0.01012568,
  2.25721397,  0.73720348])


class MountedPanda22(MountedPanda):
    """
    Panda Robot New Init State 22
    """

    @property
    def init_qpos(self):
        return np.array([-0.07886483, -0.12986421,  0.01773754, -2.42963668,  0.02069177,
  2.18613743,  0.79926961])


class MountedPanda23(MountedPanda):
    """
    Panda Robot New Init State 23
    """

    @property
    def init_qpos(self):
        return np.array([ 0.01099719, -0.18784257,  0.07001095, -2.42681746, -0.04470224,
  2.25138859,  0.7488244 ])


class MountedPanda24(MountedPanda):
    """
    Panda Robot New Init State 24
    """

    @property
    def init_qpos(self):
        return np.array([ 0.02781469, -0.12009392, -0.02900199, -2.41055284,  0.01458722,
  2.25580289,  0.8524287 ])


class MountedPanda25(MountedPanda):
    """
    Panda Robot New Init State 25
    """

    @property
    def init_qpos(self):
        return np.array([-0.01623698, -0.21091102, -0.05885789, -2.49857846, -0.00510171,
  2.24932573,  0.80370639])


class MountedPanda26(MountedPanda):
    """
    Panda Robot New Init State 26
    """

    @property
    def init_qpos(self):
        return np.array([ 0.0245069 , -0.16065218,  0.04306375, -2.45243844,  0.08059025,
  2.2452888 ,  0.76000322])


class MountedPanda27(MountedPanda):
    """
    Panda Robot New Init State 27
    """

    @property
    def init_qpos(self):
        return np.array([-0.06304125, -0.13263523, -0.0131548 , -2.40256572,  0.02785853,
  2.22246491,  0.73554915])


class MountedPanda28(MountedPanda):
    """
    Panda Robot New Init State 28
    """

    @property
    def init_qpos(self):
        return np.array([-0.06772085, -0.18099872,  0.03828508, -2.43502647, -0.05569042,
  2.23449421,  0.80262368])


class MountedPanda29(MountedPanda):
    """
    Panda Robot New Init State 29
    """

    @property
    def init_qpos(self):
        return np.array([-0.04576382, -0.15307791,  0.00301389, -2.50377747,  0.01852529,
  2.25578815,  0.84147571])


class MountedPanda30(MountedPanda):
    """
    Panda Robot New Init State 30
    """

    @property
    def init_qpos(self):
        return np.array([ 0.023848  , -0.19221465, -0.02122339, -2.432942  ,  0.0116272 ,
  2.23840795,  0.87258716])


class MountedPanda31(MountedPanda):
    """
    Panda Robot New Init State 31
    """

    @property
    def init_qpos(self):
        return np.array([ 0.02778215, -0.10577557,  0.04642609, -2.41289779, -0.01534244,
  2.26368712,  0.74778895])


class MountedPanda32(MountedPanda):
    """
    Panda Robot New Init State 32
    """

    @property
    def init_qpos(self):
        return np.array([-0.00677607, -0.17492506,  0.00234266, -2.37836832, -0.05342792,
  2.24638811,  0.73925364])


class MountedPanda33(MountedPanda):
    """
    Panda Robot New Init State 33
    """

    @property
    def init_qpos(self):
        return np.array([-0.02336083, -0.10713388,  0.00318189, -2.49794629, -0.03540783,
  2.26039257,  0.74924471])


class MountedPanda34(MountedPanda):
    """
    Panda Robot New Init State 34
    """

    @property
    def init_qpos(self):
        return np.array([ 0.00698332, -0.15956716, -0.02102174, -2.37543018,  0.02045131,
  2.16141765,  0.7914135 ])


class MountedPanda35(MountedPanda):
    """
    Panda Robot New Init State 35
    """

    @property
    def init_qpos(self):
        return np.array([-0.03209716, -0.11969371, -0.03843787, -2.45016228,  0.02449227,
  2.268742  ,  0.72718286])


class MountedPanda36(MountedPanda):
    """
    Panda Robot New Init State 36
    """

    @property
    def init_qpos(self):
        return np.array([-0.01314304, -0.17969869, -0.02567027, -2.37523018,  0.01591233,
  2.17721022,  0.82146231])


class MountedPanda37(MountedPanda):
    """
    Panda Robot New Init State 37
    """

    @property
    def init_qpos(self):
        return np.array([ 0.06575681, -0.12904558, -0.04707897, -2.45960187,  0.03925632,
  2.20482446,  0.79915028])


class MountedPanda38(MountedPanda):
    """
    Panda Robot New Init State 38
    """

    @property
    def init_qpos(self):
        return np.array([ 0.02024626, -0.18526415, -0.00155579, -2.529313  , -0.02677395,
  2.22015094,  0.75278542])


class MountedPanda39(MountedPanda):
    """
    Panda Robot New Init State 39
    """

    @property
    def init_qpos(self):
        return np.array([ 0.05058799, -0.20535709, -0.01363686, -2.44054585,  0.04466468,
  2.18225521,  0.8214443 ])


class MountedPanda40(MountedPanda):
    """
    Panda Robot New Init State 40
    """

    @property
    def init_qpos(self):
        return np.array([ 0.00077795, -0.2356552 ,  0.03513077, -2.42946424, -0.04563064,
  2.23205881,  0.75610524])


class MountedPanda41(MountedPanda):
    """
    Panda Robot New Init State 41
    """

    @property
    def init_qpos(self):
        return np.array([ 0.00316274, -0.1425896 ,  0.04418842, -2.47908456,  0.05942898,
  2.17236459,  0.78116924])


class MountedPanda42(MountedPanda):
    """
    Panda Robot New Init State 42
    """

    @property
    def init_qpos(self):
        return np.array([ 0.0399526 , -0.14195524, -0.04228751, -2.45873105, -0.03347968,
  2.18672846,  0.84309462])


class MountedPanda43(MountedPanda):
    """
    Panda Robot New Init State 43
    """

    @property
    def init_qpos(self):
        return np.array([ 0.01978629, -0.1994394 ,  0.04985707, -2.42756651,  0.04504994,
  2.2616471 ,  0.73945411])


class MountedPanda44(MountedPanda):
    """
    Panda Robot New Init State 44
    """

    @property
    def init_qpos(self):
        return np.array([-0.0311145 , -0.11952998,  0.03390219, -2.44575842,  0.00651679,
  2.29771837,  0.75254013])


class MountedPanda45(MountedPanda):
    """
    Panda Robot New Init State 45
    """

    @property
    def init_qpos(self):
        return np.array([ 0.02539213, -0.17042164, -0.01010312, -2.39360055,  0.03830959,
  2.26450917,  0.84598862])


class MountedPanda46(MountedPanda):
    """
    Panda Robot New Init State 46
    """

    @property
    def init_qpos(self):
        return np.array([ 0.0020523 , -0.09440329, -0.03031638, -2.41292295, -0.01271637,
  2.23622974,  0.84355137])


class MountedPanda47(MountedPanda):
    """
    Panda Robot New Init State 47
    """

    @property
    def init_qpos(self):
        return np.array([-0.0260161 , -0.09450795, -0.03198728, -2.48320375,  0.03682323,
  2.25192386,  0.80524264])


class MountedPanda48(MountedPanda):
    """
    Panda Robot New Init State 48
    """

    @property
    def init_qpos(self):
        return np.array([ 0.03869536, -0.16179158, -0.05525556, -2.4399292 , -0.04170161,
  2.28680294,  0.77634194])


class MountedPanda49(MountedPanda):
    """
    Panda Robot New Init State 49
    """

    @property
    def init_qpos(self):
        return np.array([-0.05739723, -0.18338351,  0.02871133, -2.48379352, -0.05716939,
  2.24369589,  0.80243081])


class MountedPanda50(MountedPanda):
    """
    Panda Robot New Init State 50
    """

    @property
    def init_qpos(self):
        return np.array([-0.02229029, -0.18174894,  0.01020324, -2.50826974, -0.06188618,
  2.19516221,  0.77601289])


class MountedPanda51(MountedPanda):
    """
    Panda Robot New Init State 51
    """

    @property
    def init_qpos(self):
        return np.array([ 0.01546771, -0.08763812,  0.04266874, -2.45255444, -0.00094606,
  2.17687616,  0.78447713])


class MountedPanda52(MountedPanda):
    """
    Panda Robot New Init State 52
    """

    @property
    def init_qpos(self):
        return np.array([-0.01508021, -0.14417781, -0.04321649, -2.41746559,  0.08007388,
  2.22107031,  0.80638453])


class MountedPanda53(MountedPanda):
    """
    Panda Robot New Init State 53
    """

    @property
    def init_qpos(self):
        return np.array([ 0.06063846, -0.19629002,  0.01968955, -2.44349106,  0.00858216,
  2.15883287,  0.78755171])


class MountedPanda54(MountedPanda):
    """
    Panda Robot New Init State 54
    """

    @property
    def init_qpos(self):
        return np.array([ 0.01633563, -0.11343612,  0.03146657, -2.37396751, -0.02517099,
  2.25536658,  0.79141226])


class MountedPanda55(MountedPanda):
    """
    Panda Robot New Init State 55
    """

    @property
    def init_qpos(self):
        return np.array([ 0.06354166, -0.18449184, -0.02436627, -2.46199009, -0.06162923,
  2.21149633,  0.76337036])


class MountedPanda56(MountedPanda):
    """
    Panda Robot New Init State 56
    """

    @property
    def init_qpos(self):
        return np.array([ 0.0061683 , -0.1470205 ,  0.07694986, -2.40561649, -0.02366131,
  2.18990434,  0.80557389])


class MountedPanda57(MountedPanda):
    """
    Panda Robot New Init State 57
    """

    @property
    def init_qpos(self):
        return np.array([-0.03894954, -0.10700567,  0.03479586, -2.4584391 , -0.05054092,
  2.26669416,  0.78201901])


class MountedPanda58(MountedPanda):
    """
    Panda Robot New Init State 58
    """

    @property
    def init_qpos(self):
        return np.array([ 0.0552114 , -0.23215504, -0.02673445, -2.44436358,  0.00209552,
  2.20667754,  0.81317968])


class MountedPanda59(MountedPanda):
    """
    Panda Robot New Init State 59
    """

    @property
    def init_qpos(self):
        return np.array([-0.04526122, -0.16707349,  0.00509987, -2.4227881 ,  0.03016855,
  2.17907358,  0.72036019])


class MountedPanda60(MountedPanda):
    """
    Panda Robot New Init State 60
    """

    @property
    def init_qpos(self):
        return np.array([ 0.05165599, -0.14760206, -0.03026103, -2.38188499,  0.00467668,
  2.27443074,  0.78812791])


class MountedPanda61(MountedPanda):
    """
    Panda Robot New Init State 61
    """

    @property
    def init_qpos(self):
        return np.array([ 0.06062446, -0.1093976 , -0.00732419, -2.41601515,  0.0189861 ,
  2.26701552,  0.7570114 ])


class MountedPanda62(MountedPanda):
    """
    Panda Robot New Init State 62
    """

    @property
    def init_qpos(self):
        return np.array([ 0.02084684, -0.12887536, -0.05344229, -2.4805528 , -0.06196554,
  2.21856581,  0.80720191])


class MountedPanda63(MountedPanda):
    """
    Panda Robot New Init State 63
    """

    @property
    def init_qpos(self):
        return np.array([ 0.04781634, -0.15867913,  0.05183484, -2.48852271, -0.05421448,
  2.22498425,  0.79762202])


class MountedPanda64(MountedPanda):
    """
    Panda Robot New Init State 64
    """

    @property
    def init_qpos(self):
        return np.array([-0.00119775, -0.23677652, -0.00326484, -2.49238569,  0.02453293,
  2.24018224,  0.7509664 ])


class MountedPanda65(MountedPanda):
    """
    Panda Robot New Init State 65
    """

    @property
    def init_qpos(self):
        return np.array([-0.02632928, -0.21530888, -0.00321152, -2.39565833, -0.05050618,
  2.2525783 ,  0.75822907])


class MountedPanda66(MountedPanda):
    """
    Panda Robot New Init State 66
    """

    @property
    def init_qpos(self):
        return np.array([-0.02931612, -0.16499479, -0.03827763, -2.46506841, -0.04429101,
  2.29939705,  0.78670202])


class MountedPanda67(MountedPanda):
    """
    Panda Robot New Init State 67
    """

    @property
    def init_qpos(self):
        return np.array([-0.05173512, -0.14521651, -0.00830512, -2.46093514,  0.04540921,
  2.28275952,  0.74617487])


class MountedPanda68(MountedPanda):
    """
    Panda Robot New Init State 68
    """

    @property
    def init_qpos(self):
        return np.array([-0.01619734, -0.16877439, -0.06475133, -2.48721867,  0.03844916,
  2.27302392,  0.77839296])


class MountedPanda69(MountedPanda):
    """
    Panda Robot New Init State 69
    """

    @property
    def init_qpos(self):
        return np.array([ 0.01505194, -0.15291171,  0.08037911, -2.41536918, -0.00333949,
  2.20180629,  0.74345931])


class MountedPanda70(MountedPanda):
    """
    Panda Robot New Init State 70
    """

    @property
    def init_qpos(self):
        return np.array([ 0.00726294, -0.18803648, -0.0507695 , -2.46767787, -0.0386075 ,
  2.28697728,  0.81686963])


class MountedPanda71(MountedPanda):
    """
    Panda Robot New Init State 71
    """

    @property
    def init_qpos(self):
        return np.array([-0.00030976, -0.10353642,  0.00300603, -2.47806135,  0.05917866,
  2.24769073,  0.74509755])


class MountedPanda72(MountedPanda):
    """
    Panda Robot New Init State 72
    """

    @property
    def init_qpos(self):
        return np.array([-0.00617811, -0.18945865, -0.04488362, -2.41453513,  0.06197682,
  2.18135678,  0.80367131])


class MountedPanda73(MountedPanda):
    """
    Panda Robot New Init State 73
    """

    @property
    def init_qpos(self):
        return np.array([-0.040954  , -0.19169898, -0.0372876 , -2.49898045,  0.00305414,
  2.17444893,  0.80242178])


class MountedPanda74(MountedPanda):
    """
    Panda Robot New Init State 74
    """

    @property
    def init_qpos(self):
        return np.array([-0.00290384, -0.17484894, -0.05245853, -2.47793571,  0.04366274,
  2.25570596,  0.72889402])


class MountedPanda75(MountedPanda):
    """
    Panda Robot New Init State 75
    """

    @property
    def init_qpos(self):
        return np.array([ 0.00444058, -0.12744717, -0.07462956, -2.42030694, -0.02962212,
  2.2522604 ,  0.75127721])


class MountedPanda76(MountedPanda):
    """
    Panda Robot New Init State 76
    """

    @property
    def init_qpos(self):
        return np.array([-0.0571827 , -0.21260157,  0.00152344, -2.43636888, -0.02865077,
  2.24698424,  0.73275749])


class MountedPanda77(MountedPanda):
    """
    Panda Robot New Init State 77
    """

    @property
    def init_qpos(self):
        return np.array([-0.00339032, -0.22317034, -0.0334434 , -2.44216561, -0.04414476,
  2.20702202,  0.83702748])


class MountedPanda78(MountedPanda):
    """
    Panda Robot New Init State 78
    """

    @property
    def init_qpos(self):
        return np.array([-0.01707835, -0.1362975 , -0.03344392, -2.42891311,  0.04267629,
  2.15358147,  0.76180682])


class MountedPanda79(MountedPanda):
    """
    Panda Robot New Init State 79
    """

    @property
    def init_qpos(self):
        return np.array([ 0.03828278, -0.17450735,  0.0246217 , -2.48466565,  0.00574434,
  2.21642462,  0.86286845])


class MountedPanda80(MountedPanda):
    """
    Panda Robot New Init State 80
    """

    @property
    def init_qpos(self):
        return np.array([ 0.0241918 , -0.12893618, -0.03916364, -2.49096189, -0.04113013,
  2.26425899,  0.74536853])


class MountedPanda81(MountedPanda):
    """
    Panda Robot New Init State 81
    """

    @property
    def init_qpos(self):
        return np.array([ 0.00872919, -0.09851794,  0.02624179, -2.45441861,  0.03618541,
  2.21445932,  0.72400163])


class MountedPanda82(MountedPanda):
    """
    Panda Robot New Init State 82
    """

    @property
    def init_qpos(self):
        return np.array([-0.03655432, -0.22887436, -0.01274627, -2.4439296 ,  0.06078946,
  2.23860695,  0.77745334])


class MountedPanda83(MountedPanda):
    """
    Panda Robot New Init State 83
    """

    @property
    def init_qpos(self):
        return np.array([ 0.0264462 , -0.23154103,  0.00751274, -2.42001788, -0.04714579,
  2.26322162,  0.79619137])


class MountedPanda84(MountedPanda):
    """
    Panda Robot New Init State 84
    """

    @property
    def init_qpos(self):
        return np.array([-0.01597632, -0.13669397,  0.08735463, -2.43760099,  0.00954916,
  2.20908037,  0.75270425])


class MountedPanda85(MountedPanda):
    """
    Panda Robot New Init State 85
    """

    @property
    def init_qpos(self):
        return np.array([ 0.04736774, -0.20987396,  0.0040826 , -2.47184615,  0.02732411,
  2.24578645,  0.84458616])


class MountedPanda86(MountedPanda):
    """
    Panda Robot New Init State 86
    """

    @property
    def init_qpos(self):
        return np.array([-0.02918384, -0.17648001, -0.05600621, -2.47002054,  0.02158966,
  2.27006813,  0.73263059])


class MountedPanda87(MountedPanda):
    """
    Panda Robot New Init State 87
    """

    @property
    def init_qpos(self):
        return np.array([ 0.02553359, -0.12123281,  0.01213938, -2.38949052, -0.02272019,
  2.19020634,  0.73317093])


class MountedPanda88(MountedPanda):
    """
    Panda Robot New Init State 88
    """

    @property
    def init_qpos(self):
        return np.array([ 0.04727849, -0.14035791, -0.00175661, -2.43574981, -0.03556808,
  2.30404367,  0.78948185])


class MountedPanda89(MountedPanda):
    """
    Panda Robot New Init State 89
    """

    @property
    def init_qpos(self):
        return np.array([ 0.00479297, -0.12923898,  0.02107472, -2.43478832, -0.03463349,
  2.2474089 ,  0.86785632])


class MountedPanda90(MountedPanda):
    """
    Panda Robot New Init State 90
    """

    @property
    def init_qpos(self):
        return np.array([ 0.05157463, -0.099965  , -0.01959668, -2.48253247, -0.00482185,
  2.22888833,  0.82734233])


class MountedPanda91(MountedPanda):
    """
    Panda Robot New Init State 91
    """

    @property
    def init_qpos(self):
        return np.array([-0.05392184, -0.112306  , -0.00503412, -2.45819788, -0.03224559,
  2.17402855,  0.81162434])


class MountedPanda92(MountedPanda):
    """
    Panda Robot New Init State 92
    """

    @property
    def init_qpos(self):
        return np.array([ 0.00250423, -0.20509695, -0.04423437, -2.45606644,  0.05700666,
  2.21788567,  0.73405721])


class MountedPanda93(MountedPanda):
    """
    Panda Robot New Init State 93
    """

    @property
    def init_qpos(self):
        return np.array([-0.00728865, -0.16912627, -0.07998869, -2.44620784, -0.00684944,
  2.24740143,  0.84023754])


class MountedPanda94(MountedPanda):
    """
    Panda Robot New Init State 94
    """

    @property
    def init_qpos(self):
        return np.array([ 0.03715728, -0.16990609, -0.03649633, -2.35972084,  0.00195319,
  2.22721163,  0.78460245])


class MountedPanda95(MountedPanda):
    """
    Panda Robot New Init State 95
    """

    @property
    def init_qpos(self):
        return np.array([ 0.01621277, -0.17285295, -0.04695287, -2.48935657, -0.00268078,
  2.18227417,  0.72705343])


class MountedPanda96(MountedPanda):
    """
    Panda Robot New Init State 96
    """

    @property
    def init_qpos(self):
        return np.array([ 0.00262738, -0.16733187,  0.03712823, -2.51004047,  0.02694542,
  2.2575136 ,  0.73421354])


class MountedPanda97(MountedPanda):
    """
    Panda Robot New Init State 97
    """

    @property
    def init_qpos(self):
        return np.array([-0.0121001 , -0.17415276, -0.04969844, -2.47206174, -0.0392138 ,
  2.28862387,  0.81843644])


class MountedPanda98(MountedPanda):
    """
    Panda Robot New Init State 98
    """

    @property
    def init_qpos(self):
        return np.array([ 0.05222094, -0.13139935, -0.04636854, -2.46613876,  0.02009791,
  2.17656116,  0.81467999])


class MountedPanda99(MountedPanda):
    """
    Panda Robot New Init State 99
    """

    @property
    def init_qpos(self):
        return np.array([-0.01276572, -0.1809473 ,  0.03776512, -2.42099887, -0.01917398,
  2.28833405,  0.72797371])


class MountedPanda100(MountedPanda):
    """
    Panda Robot New Init State 100
    """

    @property
    def init_qpos(self):
        return np.array([ 0.03346968, -0.12880851, -0.01682062, -2.42687554, -0.06798499,
  2.27696344,  0.77535066])


class MountedPanda101(MountedPanda):
    """
    Panda Robot New Init State 101
    """

    @property
    def init_qpos(self):
        return np.array([-0.03632438, -0.08814102, -0.04894533, -2.54247241, -0.10817123,
  2.26886425,  0.69642025])


class MountedPanda102(MountedPanda):
    """
    Panda Robot New Init State 102
    """

    @property
    def init_qpos(self):
        return np.array([ 0.10678435, -0.28772887,  0.10323432, -2.43175644, -0.00588528,
  2.19359228,  0.80968676])


class MountedPanda103(MountedPanda):
    """
    Panda Robot New Init State 103
    """

    @property
    def init_qpos(self):
        return np.array([-0.00617766,  0.02006746,  0.01875025, -2.41992573, -0.05968623,
  2.21740469,  0.83592322])


class MountedPanda104(MountedPanda):
    """
    Panda Robot New Init State 104
    """

    @property
    def init_qpos(self):
        return np.array([-0.14612396, -0.276232  ,  0.0635076 , -2.42999801, -0.01572031,
  2.22832727,  0.815097  ])


class MountedPanda105(MountedPanda):
    """
    Panda Robot New Init State 105
    """

    @property
    def init_qpos(self):
        return np.array([-0.04400783, -0.22449434,  0.01596771, -2.52436642,  0.03328577,
  2.0879367 ,  0.86930756])


class MountedPanda106(MountedPanda):
    """
    Panda Robot New Init State 106
    """

    @property
    def init_qpos(self):
        return np.array([ 0.02976572, -0.14491181,  0.06189307, -2.33970049,  0.06388832,
  2.11080797,  0.70480625])


class MountedPanda107(MountedPanda):
    """
    Panda Robot New Init State 107
    """

    @property
    def init_qpos(self):
        return np.array([-0.08492501, -0.15749111,  0.07035994, -2.54324018,  0.02538524,
  2.12408095,  0.70228099])


class MountedPanda108(MountedPanda):
    """
    Panda Robot New Init State 108
    """

    @property
    def init_qpos(self):
        return np.array([-0.07375089, -0.20944227, -0.07086848, -2.49576226,  0.05524217,
  2.17697546,  0.92341329])


class MountedPanda109(MountedPanda):
    """
    Panda Robot New Init State 109
    """

    @property
    def init_qpos(self):
        return np.array([ 0.03403201, -0.14828629, -0.05921463, -2.39628592, -0.03971093,
  2.23516916,  0.96200805])


class MountedPanda110(MountedPanda):
    """
    Panda Robot New Init State 110
    """

    @property
    def init_qpos(self):
        return np.array([-0.00653623, -0.08283712, -0.0478464 , -2.4469782 ,  0.120491  ,
  2.18409134,  0.90872302])


class MountedPanda111(MountedPanda):
    """
    Panda Robot New Init State 111
    """

    @property
    def init_qpos(self):
        return np.array([ 0.06011694, -0.20881371,  0.05371716, -2.36198801,  0.05281697,
  2.09337621,  0.72363462])


class MountedPanda112(MountedPanda):
    """
    Panda Robot New Init State 112
    """

    @property
    def init_qpos(self):
        return np.array([-0.02966835, -0.16995921,  0.07439567, -2.42329768, -0.1600585 ,
  2.27232389,  0.85858486])


class MountedPanda113(MountedPanda):
    """
    Panda Robot New Init State 113
    """

    @property
    def init_qpos(self):
        return np.array([ 0.04846711, -0.06746252,  0.07220165, -2.40484129, -0.006075  ,
  2.08294486,  0.82259486])


class MountedPanda114(MountedPanda):
    """
    Panda Robot New Init State 114
    """

    @property
    def init_qpos(self):
        return np.array([ 0.01960103, -0.13540647, -0.12049627, -2.54662483,  0.09939388,
  2.22301909,  0.84971646])


class MountedPanda115(MountedPanda):
    """
    Panda Robot New Init State 115
    """

    @property
    def init_qpos(self):
        return np.array([ 0.00453444, -0.15627273,  0.15024129, -2.52722836,  0.0153912 ,
  2.15273106,  0.71582511])


class MountedPanda116(MountedPanda):
    """
    Panda Robot New Init State 116
    """

    @property
    def init_qpos(self):
        return np.array([-0.03582176, -0.1353002 , -0.05546948, -2.2991012 , -0.10365231,
  2.2051006 ,  0.73444939])


class MountedPanda117(MountedPanda):
    """
    Panda Robot New Init State 117
    """

    @property
    def init_qpos(self):
        return np.array([ 0.11554121, -0.14534249,  0.0823928 , -2.56321949,  0.02132397,
  2.29778923,  0.79196854])


class MountedPanda118(MountedPanda):
    """
    Panda Robot New Init State 118
    """

    @property
    def init_qpos(self):
        return np.array([ 0.06391338, -0.1920672 ,  0.08454043, -2.30669689, -0.02176506,
  2.2000285 ,  0.87258017])


class MountedPanda119(MountedPanda):
    """
    Panda Robot New Init State 119
    """

    @property
    def init_qpos(self):
        return np.array([ 0.12219294, -0.20148493, -0.03250492, -2.46639584, -0.10400434,
  2.15568689,  0.70771959])


class MountedPanda120(MountedPanda):
    """
    Panda Robot New Init State 120
    """

    @property
    def init_qpos(self):
        return np.array([-0.06825867, -0.16412094,  0.02082214, -2.30675501, -0.08875558,
  2.31426034,  0.76637415])


class MountedPanda121(MountedPanda):
    """
    Panda Robot New Init State 121
    """

    @property
    def init_qpos(self):
        return np.array([-0.00664196, -0.07042295, -0.15075857, -2.39324767,  0.02235112,
  2.29287832,  0.82422759])


class MountedPanda122(MountedPanda):
    """
    Panda Robot New Init State 122
    """

    @property
    def init_qpos(self):
        return np.array([ 0.16159892, -0.20301111, -0.03494829, -2.48561031, -0.03655948,
  2.1848017 ,  0.86365491])


class MountedPanda123(MountedPanda):
    """
    Panda Robot New Init State 123
    """

    @property
    def init_qpos(self):
        return np.array([ 0.1419553 , -0.2180738 , -0.08317982, -2.39748748, -0.05518532,
  2.290003  ,  0.80567688])


class MountedPanda124(MountedPanda):
    """
    Panda Robot New Init State 124
    """

    @property
    def init_qpos(self):
        return np.array([-0.10315224, -0.05572369,  0.12221641, -2.48630011, -0.02638459,
  2.24620644,  0.80815924])


class MountedPanda125(MountedPanda):
    """
    Panda Robot New Init State 125
    """

    @property
    def init_qpos(self):
        return np.array([ 0.04783544, -0.01501985, -0.01285312, -2.5025843 , -0.10019119,
  2.1736588 ,  0.78299188])


class MountedPanda126(MountedPanda):
    """
    Panda Robot New Init State 126
    """

    @property
    def init_qpos(self):
        return np.array([ 0.1052975 , -0.19140876,  0.01313099, -2.4455611 ,  0.06973018,
  2.37502253,  0.7542489 ])


class MountedPanda127(MountedPanda):
    """
    Panda Robot New Init State 127
    """

    @property
    def init_qpos(self):
        return np.array([-0.03988267, -0.07595246,  0.0555649 , -2.2941159 ,  0.04758222,
  2.19747477,  0.83352851])


class MountedPanda128(MountedPanda):
    """
    Panda Robot New Init State 128
    """

    @property
    def init_qpos(self):
        return np.array([ 0.08383421, -0.09899693,  0.03835734, -2.36394126,  0.08841585,
  2.33126364,  0.83445012])


class MountedPanda129(MountedPanda):
    """
    Panda Robot New Init State 129
    """

    @property
    def init_qpos(self):
        return np.array([-0.02129298, -0.14234419,  0.15372464, -2.5486855 ,  0.04697368,
  2.17663582,  0.78906062])


class MountedPanda130(MountedPanda):
    """
    Panda Robot New Init State 130
    """

    @property
    def init_qpos(self):
        return np.array([ 0.08441091, -0.1484199 ,  0.00306601, -2.53438317,  0.04927205,
  2.26937086,  0.92822897])


class MountedPanda131(MountedPanda):
    """
    Panda Robot New Init State 131
    """

    @property
    def init_qpos(self):
        return np.array([-0.0346839 , -0.13634107,  0.02810335, -2.26683235, -0.01073897,
  2.25819547,  0.85390276])


class MountedPanda132(MountedPanda):
    """
    Panda Robot New Init State 132
    """

    @property
    def init_qpos(self):
        return np.array([ 0.02159798, -0.21270724,  0.02246381, -2.32033624, -0.11880801,
  2.242142  ,  0.70436677])


class MountedPanda133(MountedPanda):
    """
    Panda Robot New Init State 133
    """

    @property
    def init_qpos(self):
        return np.array([ 0.09055494, -0.27645723, -0.04235243, -2.41601415,  0.11862795,
  2.22176996,  0.74332779])


class MountedPanda134(MountedPanda):
    """
    Panda Robot New Init State 134
    """

    @property
    def init_qpos(self):
        return np.array([ 0.10658242, -0.24307882, -0.12457974, -2.41966717, -0.02844534,
  2.16889129,  0.82553216])


class MountedPanda135(MountedPanda):
    """
    Panda Robot New Init State 135
    """

    @property
    def init_qpos(self):
        return np.array([ 0.01895024, -0.20488231, -0.099516  , -2.37678277,  0.05053909,
  2.21904342,  0.92893428])


class MountedPanda136(MountedPanda):
    """
    Panda Robot New Init State 136
    """

    @property
    def init_qpos(self):
        return np.array([-0.10449833, -0.31001142, -0.06756777, -2.44904914,  0.02376313,
  2.20319447,  0.81977787])


class MountedPanda137(MountedPanda):
    """
    Panda Robot New Init State 137
    """

    @property
    def init_qpos(self):
        return np.array([-0.10618642, -0.03854169, -0.00697009, -2.3498009 ,  0.02907841,
  2.26550527,  0.83373987])


class MountedPanda138(MountedPanda):
    """
    Panda Robot New Init State 138
    """

    @property
    def init_qpos(self):
        return np.array([ 0.04003035, -0.10357051,  0.11884168, -2.42702619,  0.06339317,
  2.21872879,  0.91416137])


class MountedPanda139(MountedPanda):
    """
    Panda Robot New Init State 139
    """

    @property
    def init_qpos(self):
        return np.array([-0.05136378, -0.02427783, -0.00304951, -2.55324747,  0.00972797,
  2.1750346 ,  0.84923483])


class MountedPanda140(MountedPanda):
    """
    Panda Robot New Init State 140
    """

    @property
    def init_qpos(self):
        return np.array([-0.03570717, -0.18544953, -0.1033829 , -2.46934461, -0.13261829,
  2.14009174,  0.82700291])


class MountedPanda141(MountedPanda):
    """
    Panda Robot New Init State 141
    """

    @property
    def init_qpos(self):
        return np.array([ 0.06802353, -0.12420725, -0.0837072 , -2.44872765, -0.00031186,
  2.12647726,  0.91554126])


class MountedPanda142(MountedPanda):
    """
    Panda Robot New Init State 142
    """

    @property
    def init_qpos(self):
        return np.array([ 0.07443285, -0.17978334,  0.00228092, -2.42691888, -0.17321482,
  2.20578239,  0.72754061])


class MountedPanda143(MountedPanda):
    """
    Panda Robot New Init State 143
    """

    @property
    def init_qpos(self):
        return np.array([-0.07395651, -0.18179296,  0.1327356 , -2.39727962, -0.04217408,
  2.26902994,  0.88872223])


class MountedPanda144(MountedPanda):
    """
    Panda Robot New Init State 144
    """

    @property
    def init_qpos(self):
        return np.array([ 0.1046239 , -0.15429011, -0.07320202, -2.36559229,  0.04452355,
  2.32804486,  0.85726895])


class MountedPanda145(MountedPanda):
    """
    Panda Robot New Init State 145
    """

    @property
    def init_qpos(self):
        return np.array([ 0.06341197, -0.19337528,  0.07959443, -2.43265888,  0.12538329,
  2.18511279,  0.89028177])


class MountedPanda146(MountedPanda):
    """
    Panda Robot New Init State 146
    """

    @property
    def init_qpos(self):
        return np.array([ 0.03146313, -0.26459742, -0.07692639, -2.4955251 ,  0.06743234,
  2.30987066,  0.69419345])


class MountedPanda147(MountedPanda):
    """
    Panda Robot New Init State 147
    """

    @property
    def init_qpos(self):
        return np.array([-0.00170548, -0.01102047,  0.12097479, -2.414043  ,  0.00266127,
  2.23515764,  0.82836103])


class MountedPanda148(MountedPanda):
    """
    Panda Robot New Init State 148
    """

    @property
    def init_qpos(self):
        return np.array([-0.07631046, -0.18024023, -0.1244929 , -2.41481147,  0.04828723,
  2.1907017 ,  0.90283317])


class MountedPanda149(MountedPanda):
    """
    Panda Robot New Init State 149
    """

    @property
    def init_qpos(self):
        return np.array([-0.08170215, -0.25864379,  0.01496061, -2.37480421,  0.11224047,
  2.19616574,  0.85729653])


class MountedPanda150(MountedPanda):
    """
    Panda Robot New Init State 150
    """

    @property
    def init_qpos(self):
        return np.array([-0.00275632, -0.17339355,  0.06324981, -2.3979061 , -0.11283356,
  2.33243866,  0.88418108])


class MountedPanda151(MountedPanda):
    """
    Panda Robot New Init State 151
    """

    @property
    def init_qpos(self):
        return np.array([-0.09846277, -0.09873842,  0.07775972, -2.40356775, -0.08661733,
  2.12103954,  0.78137599])


class MountedPanda152(MountedPanda):
    """
    Panda Robot New Init State 152
    """

    @property
    def init_qpos(self):
        return np.array([ 0.06083562, -0.13283799, -0.01922571, -2.4045654 , -0.14776888,
  2.28634471,  0.69515007])


class MountedPanda153(MountedPanda):
    """
    Panda Robot New Init State 153
    """

    @property
    def init_qpos(self):
        return np.array([-0.03326072, -0.26378801, -0.1188112 , -2.33806994, -0.04292032,
  2.2585504 ,  0.78109232])


class MountedPanda154(MountedPanda):
    """
    Panda Robot New Init State 154
    """

    @property
    def init_qpos(self):
        return np.array([ 0.0398848 , -0.15461441, -0.1072694 , -2.36130083, -0.04128238,
  2.0966083 ,  0.74960398])


class MountedPanda155(MountedPanda):
    """
    Panda Robot New Init State 155
    """

    @property
    def init_qpos(self):
        return np.array([ 0.10760345, -0.10007707, -0.02499812, -2.46963901, -0.02306101,
  2.37565361,  0.81278267])


class MountedPanda156(MountedPanda):
    """
    Panda Robot New Init State 156
    """

    @property
    def init_qpos(self):
        return np.array([ 0.07228846,  0.01214956,  0.0401396 , -2.48814152, -0.0330057 ,
  2.21471629,  0.77914124])


class MountedPanda157(MountedPanda):
    """
    Panda Robot New Init State 157
    """

    @property
    def init_qpos(self):
        return np.array([ 0.05984078, -0.15676532,  0.06025285, -2.45123566,  0.00646701,
  2.06241873,  0.86075761])


class MountedPanda158(MountedPanda):
    """
    Panda Robot New Init State 158
    """

    @property
    def init_qpos(self):
        return np.array([ 0.01722   , -0.11143766, -0.1439399 , -2.34080813, -0.00693741,
  2.28182736,  0.73371627])


class MountedPanda159(MountedPanda):
    """
    Panda Robot New Init State 159
    """

    @property
    def init_qpos(self):
        return np.array([ 0.04852586, -0.24445787, -0.04939656, -2.29302442, -0.01510023,
  2.24397081,  0.85429924])


class MountedPanda160(MountedPanda):
    """
    Panda Robot New Init State 160
    """

    @property
    def init_qpos(self):
        return np.array([ 0.03662794, -0.14992233,  0.02696844, -2.26699937, -0.00425769,
  2.24161222,  0.86303527])


class MountedPanda161(MountedPanda):
    """
    Panda Robot New Init State 161
    """

    @property
    def init_qpos(self):
        return np.array([ 0.07620032, -0.07921923,  0.04402561, -2.5233811 ,  0.11258715,
  2.14773832,  0.80625785])


class MountedPanda162(MountedPanda):
    """
    Panda Robot New Init State 162
    """

    @property
    def init_qpos(self):
        return np.array([-0.07767196, -0.16764207,  0.03385448, -2.41150554,  0.04344751,
  2.39292498,  0.8321011 ])


class MountedPanda163(MountedPanda):
    """
    Panda Robot New Init State 163
    """

    @property
    def init_qpos(self):
        return np.array([-0.02217771, -0.07346555,  0.1080443 , -2.55610593,  0.0542642 ,
  2.29044249,  0.7583693 ])


class MountedPanda164(MountedPanda):
    """
    Panda Robot New Init State 164
    """

    @property
    def init_qpos(self):
        return np.array([ 0.12023487, -0.17415204,  0.01097521, -2.45972371,  0.00136159,
  2.13093927,  0.65953971])


class MountedPanda165(MountedPanda):
    """
    Panda Robot New Init State 165
    """

    @property
    def init_qpos(self):
        return np.array([ 0.08604881, -0.20674437, -0.05350133, -2.56080699, -0.03448211,
  2.15535054,  0.8740109 ])


class MountedPanda166(MountedPanda):
    """
    Panda Robot New Init State 166
    """

    @property
    def init_qpos(self):
        return np.array([ 0.04617399, -0.19250313,  0.10299285, -2.39970615, -0.01485295,
  2.11246663,  0.89015174])


class MountedPanda167(MountedPanda):
    """
    Panda Robot New Init State 167
    """

    @property
    def init_qpos(self):
        return np.array([-0.07368801, -0.24829168,  0.05422616, -2.31521918,  0.07340698,
  2.25665589,  0.81693755])


class MountedPanda168(MountedPanda):
    """
    Panda Robot New Init State 168
    """

    @property
    def init_qpos(self):
        return np.array([ 0.11934492, -0.05490589,  0.03932289, -2.43002559, -0.00875326,
  2.12131488,  0.74614748])


class MountedPanda169(MountedPanda):
    """
    Panda Robot New Init State 169
    """

    @property
    def init_qpos(self):
        return np.array([-0.12058503, -0.16804187, -0.07041484, -2.52320335,  0.01281341,
  2.325901  ,  0.85080753])


class MountedPanda170(MountedPanda):
    """
    Panda Robot New Init State 170
    """

    @property
    def init_qpos(self):
        return np.array([-0.12238504, -0.23815655,  0.07331186, -2.52116017, -0.01750508,
  2.26961634,  0.70993759])


class MountedPanda171(MountedPanda):
    """
    Panda Robot New Init State 171
    """

    @property
    def init_qpos(self):
        return np.array([ 0.00943767, -0.2805157 , -0.05385983, -2.41595708, -0.14267208,
  2.2662021 ,  0.78363936])


class MountedPanda172(MountedPanda):
    """
    Panda Robot New Init State 172
    """

    @property
    def init_qpos(self):
        return np.array([ 0.06691981, -0.13391601,  0.16523019, -2.42942977, -0.05201133,
  2.24156537,  0.85120461])


class MountedPanda173(MountedPanda):
    """
    Panda Robot New Init State 173
    """

    @property
    def init_qpos(self):
        return np.array([ 0.00356544, -0.15807532, -0.05122601, -2.49297093, -0.10235145,
  2.35442299,  0.6946305 ])


class MountedPanda174(MountedPanda):
    """
    Panda Robot New Init State 174
    """

    @property
    def init_qpos(self):
        return np.array([-0.06953082, -0.23312726,  0.0898028 , -2.47419473,  0.12520693,
  2.15916957,  0.81339434])


class MountedPanda175(MountedPanda):
    """
    Panda Robot New Init State 175
    """

    @property
    def init_qpos(self):
        return np.array([-0.06136172, -0.00346212, -0.08723689, -2.42183911,  0.04655507,
  2.25714846,  0.77178757])


class MountedPanda176(MountedPanda):
    """
    Panda Robot New Init State 176
    """

    @property
    def init_qpos(self):
        return np.array([-0.00915513, -0.15794775, -0.01036678, -2.37662342,  0.15588905,
  2.18743713,  0.68879782])


class MountedPanda177(MountedPanda):
    """
    Panda Robot New Init State 177
    """

    @property
    def init_qpos(self):
        return np.array([-0.00481615, -0.02030361, -0.04384376, -2.35517689,  0.0915255 ,
  2.19655168,  0.81643854])


class MountedPanda178(MountedPanda):
    """
    Panda Robot New Init State 178
    """

    @property
    def init_qpos(self):
        return np.array([ 0.13159933, -0.1916767 , -0.0164524 , -2.49160618, -0.08200785,
  2.17428065,  0.68648509])


class MountedPanda179(MountedPanda):
    """
    Panda Robot New Init State 179
    """

    @property
    def init_qpos(self):
        return np.array([ 0.00243318, -0.21600491,  0.01668971, -2.55567111,  0.02362117,
  2.28625689,  0.64306762])


class MountedPanda180(MountedPanda):
    """
    Panda Robot New Init State 180
    """

    @property
    def init_qpos(self):
        return np.array([ 0.02915663, -0.06534415, -0.09428802, -2.31422529,  0.03266132,
  2.17179857,  0.78105113])


class MountedPanda181(MountedPanda):
    """
    Panda Robot New Init State 181
    """

    @property
    def init_qpos(self):
        return np.array([ 0.05972994, -0.15290632,  0.05763624, -2.54309134,  0.01811826,
  2.0755127 ,  0.77349917])


class MountedPanda182(MountedPanda):
    """
    Panda Robot New Init State 182
    """

    @property
    def init_qpos(self):
        return np.array([-0.07544855, -0.22243157,  0.10317585, -2.35350656,  0.05084434,
  2.1355776 ,  0.75898777])


class MountedPanda183(MountedPanda):
    """
    Panda Robot New Init State 183
    """

    @property
    def init_qpos(self):
        return np.array([ 0.09057037, -0.15045675,  0.1731621 , -2.4152322 ,  0.01433847,
  2.20367344,  0.79536838])


class MountedPanda184(MountedPanda):
    """
    Panda Robot New Init State 184
    """

    @property
    def init_qpos(self):
        return np.array([-0.01116493, -0.10918463,  0.07005655, -2.50215321, -0.09748235,
  2.09229218,  0.8225969 ])


class MountedPanda185(MountedPanda):
    """
    Panda Robot New Init State 185
    """

    @property
    def init_qpos(self):
        return np.array([-0.0613735 , -0.28078945,  0.02161424, -2.30592724, -0.00033769,
  2.27339228,  0.78994983])


class MountedPanda186(MountedPanda):
    """
    Panda Robot New Init State 186
    """

    @property
    def init_qpos(self):
        return np.array([-0.01239948, -0.04579713, -0.03639666, -2.41106995,  0.04033673,
  2.14298233,  0.9097874 ])


class MountedPanda187(MountedPanda):
    """
    Panda Robot New Init State 187
    """

    @property
    def init_qpos(self):
        return np.array([-0.02063033, -0.25015085,  0.0632624 , -2.55054005,  0.00332297,
  2.22567694,  0.91344121])


class MountedPanda188(MountedPanda):
    """
    Panda Robot New Init State 188
    """

    @property
    def init_qpos(self):
        return np.array([ 0.04306746, -0.16330798,  0.07419264, -2.31846954,  0.05060788,
  2.22769145,  0.66638054])


class MountedPanda189(MountedPanda):
    """
    Panda Robot New Init State 189
    """

    @property
    def init_qpos(self):
        return np.array([-0.15362209, -0.20506016, -0.08791037, -2.47156447,  0.00817066,
  2.30315016,  0.77523008])


class MountedPanda190(MountedPanda):
    """
    Panda Robot New Init State 190
    """

    @property
    def init_qpos(self):
        return np.array([ 0.03712321, -0.15611758, -0.15074083, -2.51627481, -0.01099481,
  2.1344447 ,  0.83117659])


class MountedPanda191(MountedPanda):
    """
    Panda Robot New Init State 191
    """

    @property
    def init_qpos(self):
        return np.array([ 0.11909309, -0.06621718, -0.01660595, -2.32861831,  0.01156639,
  2.20052676,  0.73767516])


class MountedPanda192(MountedPanda):
    """
    Panda Robot New Init State 192
    """

    @property
    def init_qpos(self):
        return np.array([-0.04382287, -0.21727781,  0.0246905 , -2.42133295,  0.00044132,
  2.29005733,  0.95790317])


class MountedPanda193(MountedPanda):
    """
    Panda Robot New Init State 193
    """

    @property
    def init_qpos(self):
        return np.array([ 0.05114065, -0.24101035, -0.13754792, -2.39427729, -0.07361673,
  2.21464689,  0.72237875])


class MountedPanda194(MountedPanda):
    """
    Panda Robot New Init State 194
    """

    @property
    def init_qpos(self):
        return np.array([-0.10575036, -0.12926131,  0.0445515 , -2.47843035, -0.15205696,
  2.1946955 ,  0.80839143])


class MountedPanda195(MountedPanda):
    """
    Panda Robot New Init State 195
    """

    @property
    def init_qpos(self):
        return np.array([-0.13295018, -0.14455382, -0.00137631, -2.39252072,  0.01074995,
  2.13927591,  0.89296673])


class MountedPanda196(MountedPanda):
    """
    Panda Robot New Init State 196
    """

    @property
    def init_qpos(self):
        return np.array([-0.01441759, -0.16352062, -0.08487677, -2.48491211, -0.08046906,
  2.21102349,  0.9410708 ])


class MountedPanda197(MountedPanda):
    """
    Panda Robot New Init State 197
    """

    @property
    def init_qpos(self):
        return np.array([-0.09361906, -0.27114744,  0.10039323, -2.4588818 , -0.04565778,
  2.29771569,  0.74407073])


class MountedPanda198(MountedPanda):
    """
    Panda Robot New Init State 198
    """

    @property
    def init_qpos(self):
        return np.array([ 0.11730098, -0.11748813, -0.03134293, -2.30601421, -0.0389105 ,
  2.27442364,  0.80462441])


class MountedPanda199(MountedPanda):
    """
    Panda Robot New Init State 199
    """

    @property
    def init_qpos(self):
        return np.array([ 0.08479854, -0.05931681,  0.00208483, -2.49367691,  0.02996369,
  2.1826042 ,  0.91655503])


class MountedPanda200(MountedPanda):
    """
    Panda Robot New Init State 200
    """

    @property
    def init_qpos(self):
        return np.array([ 0.01244126, -0.19432592,  0.01682829, -2.56734921, -0.08853462,
  2.33613533,  0.72554118])


class MountedPanda201(MountedPanda):
    """
    Panda Robot New Init State 201
    """

    @property
    def init_qpos(self):
        return np.array([-0.1196246 , -0.09971711,  0.13548371, -2.36244674,  0.11381274,
  2.1402775 ,  0.6229359 ])


class MountedPanda202(MountedPanda):
    """
    Panda Robot New Init State 202
    """

    @property
    def init_qpos(self):
        return np.array([ 0.14698563, -0.19262883, -0.0245574 , -2.31456278,  0.05445679,
  2.27138515,  0.99735919])


class MountedPanda203(MountedPanda):
    """
    Panda Robot New Init State 203
    """

    @property
    def init_qpos(self):
        return np.array([-0.09381943, -0.02576627,  0.00831202, -2.40543902,  0.22283757,
  2.30079029,  0.86387711])


class MountedPanda204(MountedPanda):
    """
    Panda Robot New Init State 204
    """

    @property
    def init_qpos(self):
        return np.array([ 0.09140475, -0.19214059, -0.07152189, -2.53364164, -0.16758331,
  2.40199305,  0.69137691])


class MountedPanda205(MountedPanda):
    """
    Panda Robot New Init State 205
    """

    @property
    def init_qpos(self):
        return np.array([-0.04700756, -0.21984917,  0.06531165, -2.27128587,  0.18281628,
  2.10287151,  0.74989147])


class MountedPanda206(MountedPanda):
    """
    Panda Robot New Init State 206
    """

    @property
    def init_qpos(self):
        return np.array([ 0.03674447, -0.19337522,  0.20900656, -2.38075495,  0.09555772,
  2.31294806,  0.93798746])


class MountedPanda207(MountedPanda):
    """
    Panda Robot New Init State 207
    """

    @property
    def init_qpos(self):
        return np.array([ 0.02572882, -0.34535772,  0.05680814, -2.53717563, -0.07513294,
  2.31009284,  0.96139687])


class MountedPanda208(MountedPanda):
    """
    Panda Robot New Init State 208
    """

    @property
    def init_qpos(self):
        return np.array([ 0.00175378, -0.13557506,  0.14034239, -2.42474811,  0.21450078,
  2.27337673,  0.64022868])


class MountedPanda209(MountedPanda):
    """
    Panda Robot New Init State 209
    """

    @property
    def init_qpos(self):
        return np.array([ 0.10785027, -0.1064698 ,  0.18464423, -2.34822315, -0.0198217 ,
  2.0514786 ,  0.75547387])


class MountedPanda210(MountedPanda):
    """
    Panda Robot New Init State 210
    """

    @property
    def init_qpos(self):
        return np.array([-0.10435008, -0.23226162,  0.07221249, -2.23971743,  0.04842212,
  2.0800297 ,  0.83994632])


class MountedPanda211(MountedPanda):
    """
    Panda Robot New Init State 211
    """

    @property
    def init_qpos(self):
        return np.array([ 0.2117978 , -0.26896592, -0.02372037, -2.44199773, -0.13892308,
  2.3414946 ,  0.76399094])


class MountedPanda212(MountedPanda):
    """
    Panda Robot New Init State 212
    """

    @property
    def init_qpos(self):
        return np.array([ 0.05652424, -0.28061056, -0.05893942, -2.34726993, -0.05023238,
  2.45942867,  0.73155457])


class MountedPanda213(MountedPanda):
    """
    Panda Robot New Init State 213
    """

    @property
    def init_qpos(self):
        return np.array([ 0.02104497, -0.07368677, -0.13285047, -2.60952268, -0.14289978,
  2.34501146,  0.73370867])


class MountedPanda214(MountedPanda):
    """
    Panda Robot New Init State 214
    """

    @property
    def init_qpos(self):
        return np.array([ 0.17602439, -0.1998621 ,  0.01269859, -2.33848808,  0.13803509,
  2.36859999,  0.86854334])


class MountedPanda215(MountedPanda):
    """
    Panda Robot New Init State 215
    """

    @property
    def init_qpos(self):
        return np.array([ 0.17714071, -0.05854302,  0.13463966, -2.5399407 , -0.14153074,
  2.22616849,  0.7559599 ])


class MountedPanda216(MountedPanda):
    """
    Panda Robot New Init State 216
    """

    @property
    def init_qpos(self):
        return np.array([-0.06000318, -0.06884217,  0.12647352, -2.43289322,  0.19561123,
  2.07560319,  0.75975945])


class MountedPanda217(MountedPanda):
    """
    Panda Robot New Init State 217
    """

    @property
    def init_qpos(self):
        return np.array([-0.07721287, -0.3620923 , -0.00890599, -2.45771156,  0.16302198,
  2.29470039,  0.67507738])


class MountedPanda218(MountedPanda):
    """
    Panda Robot New Init State 218
    """

    @property
    def init_qpos(self):
        return np.array([ 0.20225595, -0.02783944,  0.06349884, -2.46930376, -0.1046616 ,
  2.18614957,  0.90416563])


class MountedPanda219(MountedPanda):
    """
    Panda Robot New Init State 219
    """

    @property
    def init_qpos(self):
        return np.array([ 0.17302289, -0.019353  , -0.04488101, -2.54738897,  0.01293469,
  2.0643927 ,  0.81506802])


class MountedPanda220(MountedPanda):
    """
    Panda Robot New Init State 220
    """

    @property
    def init_qpos(self):
        return np.array([-0.01230582, -0.19989145, -0.13295244, -2.40177797, -0.04441195,
  2.12921474,  0.54593399])


class MountedPanda221(MountedPanda):
    """
    Panda Robot New Init State 221
    """

    @property
    def init_qpos(self):
        return np.array([-0.00297673,  0.0306955 ,  0.17972506, -2.49404891, -0.06515094,
  2.27746703,  0.67738679])


class MountedPanda222(MountedPanda):
    """
    Panda Robot New Init State 222
    """

    @property
    def init_qpos(self):
        return np.array([ 0.04713568, -0.04138636,  0.19162982, -2.46024014, -0.03181099,
  2.0896236 ,  0.65625101])


class MountedPanda223(MountedPanda):
    """
    Panda Robot New Init State 223
    """

    @property
    def init_qpos(self):
        return np.array([-0.03410825, -0.10571128,  0.13758799, -2.56555811,  0.22480241,
  2.18592035,  0.7821802 ])


class MountedPanda224(MountedPanda):
    """
    Panda Robot New Init State 224
    """

    @property
    def init_qpos(self):
        return np.array([-0.07871655, -0.41642397,  0.09313158, -2.36697313, -0.02963703,
  2.23380947,  0.83975084])


class MountedPanda225(MountedPanda):
    """
    Panda Robot New Init State 225
    """

    @property
    def init_qpos(self):
        return np.array([-0.17630184, -0.22073033,  0.08961678, -2.58611878,  0.0331221 ,
  2.07368925,  0.8380254 ])


class MountedPanda226(MountedPanda):
    """
    Panda Robot New Init State 226
    """

    @property
    def init_qpos(self):
        return np.array([-0.00418676, -0.35076472,  0.13682034, -2.53088894, -0.09517925,
  2.25031279,  0.92032702])


class MountedPanda227(MountedPanda):
    """
    Panda Robot New Init State 227
    """

    @property
    def init_qpos(self):
        return np.array([-0.1908126 , -0.14945141,  0.08054916, -2.31440408,  0.03314289,
  2.15780946,  0.6299431 ])


class MountedPanda228(MountedPanda):
    """
    Panda Robot New Init State 228
    """

    @property
    def init_qpos(self):
        return np.array([ 0.00621297, -0.23836342, -0.21069696, -2.41311141,  0.06519462,
  2.0563301 ,  0.71257322])


class MountedPanda229(MountedPanda):
    """
    Panda Robot New Init State 229
    """

    @property
    def init_qpos(self):
        return np.array([ 0.06787898,  0.06452948,  0.00227965, -2.52440972, -0.02005126,
  2.37748233,  0.71457997])


class MountedPanda230(MountedPanda):
    """
    Panda Robot New Init State 230
    """

    @property
    def init_qpos(self):
        return np.array([-0.11174778, -0.22853746, -0.13330008, -2.42744498,  0.22717221,
  2.27192707,  0.75011259])


class MountedPanda231(MountedPanda):
    """
    Panda Robot New Init State 231
    """

    @property
    def init_qpos(self):
        return np.array([-0.02371029, -0.28804996,  0.07175455, -2.45092191, -0.01466438,
  2.48619897,  0.80967247])


class MountedPanda232(MountedPanda):
    """
    Panda Robot New Init State 232
    """

    @property
    def init_qpos(self):
        return np.array([-0.09670926, -0.21588338,  0.14773431, -2.43002654, -0.18501789,
  2.34500041,  0.69944912])


class MountedPanda233(MountedPanda):
    """
    Panda Robot New Init State 233
    """

    @property
    def init_qpos(self):
        return np.array([ 0.20543365, -0.04259481, -0.04905519, -2.30411587,  0.00744132,
  2.32027092,  0.83855599])


class MountedPanda234(MountedPanda):
    """
    Panda Robot New Init State 234
    """

    @property
    def init_qpos(self):
        return np.array([-0.13460831, -0.2633904 ,  0.16380953, -2.49784162, -0.06419804,
  2.07484528,  0.85276923])


class MountedPanda235(MountedPanda):
    """
    Panda Robot New Init State 235
    """

    @property
    def init_qpos(self):
        return np.array([ 0.20943865, -0.26943979,  0.00245936, -2.30119384, -0.00808717,
  2.2726882 ,  0.67751764])


class MountedPanda236(MountedPanda):
    """
    Panda Robot New Init State 236
    """

    @property
    def init_qpos(self):
        return np.array([ 0.04115468, -0.17901949,  0.0947157 , -2.33050527,  0.24882304,
  2.28244733,  0.81670291])


class MountedPanda237(MountedPanda):
    """
    Panda Robot New Init State 237
    """

    @property
    def init_qpos(self):
        return np.array([ 0.03289919, -0.22081014,  0.05728778, -2.49459362,  0.02851171,
  2.44975319,  0.61504266])


class MountedPanda238(MountedPanda):
    """
    Panda Robot New Init State 238
    """

    @property
    def init_qpos(self):
        return np.array([ 0.11035396, -0.0335248 , -0.01143045, -2.65001856, -0.05884448,
  2.35231688,  0.78318426])


class MountedPanda239(MountedPanda):
    """
    Panda Robot New Init State 239
    """

    @property
    def init_qpos(self):
        return np.array([-0.18577638, -0.25487813,  0.15627639, -2.34295636, -0.04441893,
  2.1585573 ,  0.71256793])


class MountedPanda240(MountedPanda):
    """
    Panda Robot New Init State 240
    """

    @property
    def init_qpos(self):
        return np.array([-0.09467623,  0.00476486, -0.03308299, -2.31966825,  0.15614796,
  2.32921888,  0.74109663])


class MountedPanda241(MountedPanda):
    """
    Panda Robot New Init State 241
    """

    @property
    def init_qpos(self):
        return np.array([ 0.05980143, -0.1646213 , -0.13385841, -2.39655262, -0.1746421 ,
  2.40267401,  0.71657798])


class MountedPanda242(MountedPanda):
    """
    Panda Robot New Init State 242
    """

    @property
    def init_qpos(self):
        return np.array([ 0.03307883, -0.11445334, -0.04257451, -2.34812115, -0.07809347,
  2.3700289 ,  0.56406438])


class MountedPanda243(MountedPanda):
    """
    Panda Robot New Init State 243
    """

    @property
    def init_qpos(self):
        return np.array([ 0.01594384,  0.08357472, -0.12620017, -2.53010067,  0.06483377,
  2.24940794,  0.82963308])


class MountedPanda244(MountedPanda):
    """
    Panda Robot New Init State 244
    """

    @property
    def init_qpos(self):
        return np.array([ 0.05912171, -0.08432683,  0.13411504, -2.395068  , -0.02915769,
  2.30803747,  1.01501227])


class MountedPanda245(MountedPanda):
    """
    Panda Robot New Init State 245
    """

    @property
    def init_qpos(self):
        return np.array([-0.01412756, -0.26483995,  0.11792135, -2.45741994, -0.23142318,
  2.31701582,  0.72837737])


class MountedPanda246(MountedPanda):
    """
    Panda Robot New Init State 246
    """

    @property
    def init_qpos(self):
        return np.array([-0.02502072, -0.06928785,  0.05264847, -2.24852598, -0.12035555,
  2.36855946,  0.71366714])


class MountedPanda247(MountedPanda):
    """
    Panda Robot New Init State 247
    """

    @property
    def init_qpos(self):
        return np.array([ 0.11008054, -0.06934948, -0.13937756, -2.2827894 , -0.07985694,
  2.34715038,  0.84012287])


class MountedPanda248(MountedPanda):
    """
    Panda Robot New Init State 248
    """

    @property
    def init_qpos(self):
        return np.array([ 0.21000324, -0.18656725, -0.03990769, -2.44681606, -0.03463677,
  2.31814154,  0.6007428 ])


class MountedPanda249(MountedPanda):
    """
    Panda Robot New Init State 249
    """

    @property
    def init_qpos(self):
        return np.array([-0.1277355 , -0.29048469,  0.11520361, -2.23722373, -0.01266214,
  2.20628633,  0.79389043])


class MountedPanda250(MountedPanda):
    """
    Panda Robot New Init State 250
    """

    @property
    def init_qpos(self):
        return np.array([ 0.16418848, -0.29208442,  0.03368463, -2.30678939,  0.07081111,
  2.25354254,  0.92685897])


class MountedPanda251(MountedPanda):
    """
    Panda Robot New Init State 251
    """

    @property
    def init_qpos(self):
        return np.array([-0.21899576, -0.10612098, -0.05298834, -2.59290172,  0.04141383,
  2.33803579,  0.77426311])


class MountedPanda252(MountedPanda):
    """
    Panda Robot New Init State 252
    """

    @property
    def init_qpos(self):
        return np.array([ 0.08600084, -0.01552265,  0.11867643, -2.29658393,  0.06698006,
  2.2335604 ,  0.64081417])


class MountedPanda253(MountedPanda):
    """
    Panda Robot New Init State 253
    """

    @property
    def init_qpos(self):
        return np.array([-0.0305549 , -0.26574785,  0.26957762, -2.46643816,  0.015299  ,
  2.2952331 ,  0.79081251])


class MountedPanda254(MountedPanda):
    """
    Panda Robot New Init State 254
    """

    @property
    def init_qpos(self):
        return np.array([ 0.20815428, -0.23749947,  0.02389803, -2.53577374, -0.16210033,
  2.15162623,  0.78085   ])


class MountedPanda255(MountedPanda):
    """
    Panda Robot New Init State 255
    """

    @property
    def init_qpos(self):
        return np.array([-0.05493127, -0.24963596, -0.17994504, -2.45523123, -0.19253617,
  2.32400499,  0.79594672])


class MountedPanda256(MountedPanda):
    """
    Panda Robot New Init State 256
    """

    @property
    def init_qpos(self):
        return np.array([-0.15256964, -0.19340394, -0.0787322 , -2.41118778,  0.14031205,
  2.03046796,  0.79743855])


class MountedPanda257(MountedPanda):
    """
    Panda Robot New Init State 257
    """

    @property
    def init_qpos(self):
        return np.array([-0.02782642, -0.04437801,  0.03704737, -2.57526787, -0.05544229,
  2.08842606,  0.97237691])


class MountedPanda258(MountedPanda):
    """
    Panda Robot New Init State 258
    """

    @property
    def init_qpos(self):
        return np.array([ 0.00183251,  0.00243888, -0.005216  , -2.50551071, -0.13015863,
  2.07170697,  0.64918425])


class MountedPanda259(MountedPanda):
    """
    Panda Robot New Init State 259
    """

    @property
    def init_qpos(self):
        return np.array([ 0.03146766, -0.10110931, -0.18626512, -2.33916147, -0.05690485,
  2.2440445 ,  0.97531862])


class MountedPanda260(MountedPanda):
    """
    Panda Robot New Init State 260
    """

    @property
    def init_qpos(self):
        return np.array([ 0.09610682, -0.18981359, -0.09550167, -2.67518966, -0.07706569,
  2.20412713,  0.67961033])


class MountedPanda261(MountedPanda):
    """
    Panda Robot New Init State 261
    """

    @property
    def init_qpos(self):
        return np.array([-0.02850029, -0.14435027, -0.04882377, -2.58573483,  0.03662059,
  2.32271234,  0.54863596])


class MountedPanda262(MountedPanda):
    """
    Panda Robot New Init State 262
    """

    @property
    def init_qpos(self):
        return np.array([ 0.22472723, -0.17674008,  0.00300229, -2.33179613,  0.03714566,
  2.3785637 ,  0.83113006])


class MountedPanda263(MountedPanda):
    """
    Panda Robot New Init State 263
    """

    @property
    def init_qpos(self):
        return np.array([-0.13376524, -0.08207976, -0.24980645, -2.42088645, -0.00725791,
  2.18129645,  0.7573237 ])


class MountedPanda264(MountedPanda):
    """
    Panda Robot New Init State 264
    """

    @property
    def init_qpos(self):
        return np.array([-0.20916579, -0.22202766,  0.01031963, -2.42682958, -0.11422514,
  2.36743446,  0.68912363])


class MountedPanda265(MountedPanda):
    """
    Panda Robot New Init State 265
    """

    @property
    def init_qpos(self):
        return np.array([ 0.10306858, -0.1156199 , -0.06870415, -2.25585394, -0.12690241,
  2.32447277,  0.89177583])


class MountedPanda266(MountedPanda):
    """
    Panda Robot New Init State 266
    """

    @property
    def init_qpos(self):
        return np.array([-0.11112412,  0.069699  , -0.04143358, -2.39675992,  0.04313624,
  2.20449677,  0.91975194])


class MountedPanda267(MountedPanda):
    """
    Panda Robot New Init State 267
    """

    @property
    def init_qpos(self):
        return np.array([ 0.0696591 , -0.3145382 ,  0.11450594, -2.59054497,  0.09064557,
  2.32566383,  0.68962397])


class MountedPanda268(MountedPanda):
    """
    Panda Robot New Init State 268
    """

    @property
    def init_qpos(self):
        return np.array([-0.05115174, -0.29871206, -0.02676047, -2.48295176,  0.09707792,
  2.38990275,  0.95917902])


class MountedPanda269(MountedPanda):
    """
    Panda Robot New Init State 269
    """

    @property
    def init_qpos(self):
        return np.array([-0.0565068 , -0.1820675 , -0.16921222, -2.52560018,  0.09698288,
  2.38914388,  0.66132654])


class MountedPanda270(MountedPanda):
    """
    Panda Robot New Init State 270
    """

    @property
    def init_qpos(self):
        return np.array([-0.14559896, -0.16752664,  0.06885001, -2.64355872,  0.10421362,
  2.11030549,  0.7818465 ])


class MountedPanda271(MountedPanda):
    """
    Panda Robot New Init State 271
    """

    @property
    def init_qpos(self):
        return np.array([ 0.19414004, -0.15875502, -0.15508405, -2.48355091, -0.03140268,
  2.09525663,  0.69342649])


class MountedPanda272(MountedPanda):
    """
    Panda Robot New Init State 272
    """

    @property
    def init_qpos(self):
        return np.array([ 0.21304463, -0.08628614, -0.03937735, -2.35171744, -0.04448664,
  2.38308932,  0.83465874])


class MountedPanda273(MountedPanda):
    """
    Panda Robot New Init State 273
    """

    @property
    def init_qpos(self):
        return np.array([-0.011109  , -0.11468523, -0.1626329 , -2.56125177, -0.12360712,
  2.24155111,  0.96476554])


class MountedPanda274(MountedPanda):
    """
    Panda Robot New Init State 274
    """

    @property
    def init_qpos(self):
        return np.array([ 0.2097895 , -0.10104706,  0.04708999, -2.61393506, -0.05102284,
  2.15268383,  0.72702533])


class MountedPanda275(MountedPanda):
    """
    Panda Robot New Init State 275
    """

    @property
    def init_qpos(self):
        return np.array([-0.01576504, -0.20530026, -0.18557771, -2.37447395, -0.1005796 ,
  2.24918381,  0.97986264])


class MountedPanda276(MountedPanda):
    """
    Panda Robot New Init State 276
    """

    @property
    def init_qpos(self):
        return np.array([-0.07336148, -0.36293945, -0.02282145, -2.35674737,  0.15626461,
  2.12864137,  0.82503184])


class MountedPanda277(MountedPanda):
    """
    Panda Robot New Init State 277
    """

    @property
    def init_qpos(self):
        return np.array([-0.04220174, -0.17064647,  0.23983103, -2.58091068,  0.02301314,
  2.26917004,  0.8838745 ])


class MountedPanda278(MountedPanda):
    """
    Panda Robot New Init State 278
    """

    @property
    def init_qpos(self):
        return np.array([ 0.02982978, -0.06725514, -0.11923265, -2.36266504, -0.10448072,
  2.41598221,  0.67287479])


class MountedPanda279(MountedPanda):
    """
    Panda Robot New Init State 279
    """

    @property
    def init_qpos(self):
        return np.array([ 0.04285068, -0.16436258,  0.12795537, -2.45043744, -0.20147327,
  2.37014689,  0.68247797])


class MountedPanda280(MountedPanda):
    """
    Panda Robot New Init State 280
    """

    @property
    def init_qpos(self):
        return np.array([-0.04591939, -0.20287558,  0.0965381 , -2.59460344,  0.2203899 ,
  2.30092776,  0.80107807])


class MountedPanda281(MountedPanda):
    """
    Panda Robot New Init State 281
    """

    @property
    def init_qpos(self):
        return np.array([-0.01550509, -0.08038118,  0.14719369, -2.26235303,  0.15693332,
  2.25205472,  0.72965892])


class MountedPanda282(MountedPanda):
    """
    Panda Robot New Init State 282
    """

    @property
    def init_qpos(self):
        return np.array([-0.01380123, -0.1788196 ,  0.10343811, -2.39044506, -0.26826271,
  2.28659249,  0.80316468])


class MountedPanda283(MountedPanda):
    """
    Panda Robot New Init State 283
    """

    @property
    def init_qpos(self):
        return np.array([-0.00279937, -0.21917087,  0.16819944, -2.30282039, -0.11564462,
  2.28765887,  0.64001051])


class MountedPanda284(MountedPanda):
    """
    Panda Robot New Init State 284
    """

    @property
    def init_qpos(self):
        return np.array([-0.18502581, -0.2450888 ,  0.11565885, -2.31677027,  0.06881182,
  2.32399754,  0.85462478])


class MountedPanda285(MountedPanda):
    """
    Panda Robot New Init State 285
    """

    @property
    def init_qpos(self):
        return np.array([-0.09812478, -0.09496799,  0.12243533, -2.32921665,  0.18186315,
  2.3350985 ,  0.83915774])


class MountedPanda286(MountedPanda):
    """
    Panda Robot New Init State 286
    """

    @property
    def init_qpos(self):
        return np.array([ 0.18591872, -0.16564496, -0.15320933, -2.47292732, -0.12941299,
  2.10945252,  0.76029073])


class MountedPanda287(MountedPanda):
    """
    Panda Robot New Init State 287
    """

    @property
    def init_qpos(self):
        return np.array([-0.101772  , -0.20058947, -0.24320053, -2.41720375,  0.00015482,
  2.12181195,  0.87006634])


class MountedPanda288(MountedPanda):
    """
    Panda Robot New Init State 288
    """

    @property
    def init_qpos(self):
        return np.array([ 0.11254777, -0.35401204, -0.09155898, -2.53692683, -0.11282782,
  2.32632399,  0.76213089])


class MountedPanda289(MountedPanda):
    """
    Panda Robot New Init State 289
    """

    @property
    def init_qpos(self):
        return np.array([-0.02813197, -0.37415433,  0.06757867, -2.57639642,  0.00637402,
  2.25625209,  0.9301084 ])


class MountedPanda290(MountedPanda):
    """
    Panda Robot New Init State 290
    """

    @property
    def init_qpos(self):
        return np.array([-0.09838698, -0.38802093,  0.01382038, -2.30924898,  0.09313439,
  2.24251277,  0.74844076])


class MountedPanda291(MountedPanda):
    """
    Panda Robot New Init State 291
    """

    @property
    def init_qpos(self):
        return np.array([ 0.07308871, -0.24919653,  0.04313641, -2.39882479,  0.09600694,
  2.47665226,  0.82093326])


class MountedPanda292(MountedPanda):
    """
    Panda Robot New Init State 292
    """

    @property
    def init_qpos(self):
        return np.array([-0.07331552, -0.16472133, -0.25483299, -2.54466996, -0.05895658,
  2.15125797,  0.80734468])


class MountedPanda293(MountedPanda):
    """
    Panda Robot New Init State 293
    """

    @property
    def init_qpos(self):
        return np.array([-0.1489017 ,  0.04192871,  0.00003749, -2.44628132, -0.05936636,
  2.25484996,  0.93478478])


class MountedPanda294(MountedPanda):
    """
    Panda Robot New Init State 294
    """

    @property
    def init_qpos(self):
        return np.array([-0.11934521, -0.25161554, -0.04180254, -2.62983291, -0.11275642,
  2.16120878,  0.90574244])


class MountedPanda295(MountedPanda):
    """
    Panda Robot New Init State 295
    """

    @property
    def init_qpos(self):
        return np.array([ 0.07569517, -0.10522635,  0.24530979, -2.4842011 ,  0.11256914,
  2.19699387,  0.86190085])


class MountedPanda296(MountedPanda):
    """
    Panda Robot New Init State 296
    """

    @property
    def init_qpos(self):
        return np.array([-0.05122623, -0.37541671, -0.01180354, -2.31953365,  0.08141711,
  2.31161353,  0.89405775])


class MountedPanda297(MountedPanda):
    """
    Panda Robot New Init State 297
    """

    @property
    def init_qpos(self):
        return np.array([ 0.01725443, -0.12263606, -0.12020663, -2.50611117,  0.1198406 ,
  2.46105066,  0.75825931])


class MountedPanda298(MountedPanda):
    """
    Panda Robot New Init State 298
    """

    @property
    def init_qpos(self):
        return np.array([-0.04333171, -0.16098657,  0.05786748, -2.58357169, -0.2210125 ,
  2.17376181,  0.66790189])


class MountedPanda299(MountedPanda):
    """
    Panda Robot New Init State 299
    """

    @property
    def init_qpos(self):
        return np.array([-0.08549427, -0.18590133, -0.07625876, -2.20022073,  0.0549746 ,
  2.27730682,  0.89008412])


class MountedPanda300(MountedPanda):
    """
    Panda Robot New Init State 300
    """

    @property
    def init_qpos(self):
        return np.array([-0.12854555, -0.04374787, -0.00855188, -2.36891711, -0.12098566,
  2.05137201,  0.87771728])


class MountedPanda301(MountedPanda):
    """
    Panda Robot New Init State 301
    """

    @property
    def init_qpos(self):
        return np.array([ 0.05355976, -0.28939137, -0.12374539, -2.28445803, -0.16924282,
  2.46049007,  0.65724778])


class MountedPanda302(MountedPanda):
    """
    Panda Robot New Init State 302
    """

    @property
    def init_qpos(self):
        return np.array([ 0.17184991, -0.24949881,  0.16237315, -2.59106853, -0.04382253,
  2.23776731,  0.51561746])


class MountedPanda303(MountedPanda):
    """
    Panda Robot New Init State 303
    """

    @property
    def init_qpos(self):
        return np.array([ 0.07842827, -0.21536937, -0.02420101, -2.54986548, -0.27126457,
  2.20652881,  1.04081263])


class MountedPanda304(MountedPanda):
    """
    Panda Robot New Init State 304
    """

    @property
    def init_qpos(self):
        return np.array([ 0.16492962, -0.16508606, -0.0552616 , -2.43987497,  0.11500849,
  1.97853661,  1.01963683])


class MountedPanda305(MountedPanda):
    """
    Panda Robot New Init State 305
    """

    @property
    def init_qpos(self):
        return np.array([ 0.19522079,  0.0349487 ,  0.11965588, -2.43924427,  0.18361746,
  2.38825268,  0.6887838 ])


class MountedPanda306(MountedPanda):
    """
    Panda Robot New Init State 306
    """

    @property
    def init_qpos(self):
        return np.array([-0.09028895, -0.14500449,  0.24382665, -2.63964162, -0.23099878,
  2.25268671,  0.79340824])


class MountedPanda307(MountedPanda):
    """
    Panda Robot New Init State 307
    """

    @property
    def init_qpos(self):
        return np.array([ 0.02925042, -0.11987168, -0.03766411, -2.19319151, -0.27014342,
  2.08694805,  0.76797451])


class MountedPanda308(MountedPanda):
    """
    Panda Robot New Init State 308
    """

    @property
    def init_qpos(self):
        return np.array([-0.23979451, -0.18670025,  0.24246235, -2.44154945,  0.03366674,
  2.04285332,  0.69545106])


class MountedPanda309(MountedPanda):
    """
    Panda Robot New Init State 309
    """

    @property
    def init_qpos(self):
        return np.array([-0.07760716, -0.22675488,  0.26510046, -2.32924236, -0.22989852,
  2.27774026,  0.88845239])


class MountedPanda310(MountedPanda):
    """
    Panda Robot New Init State 310
    """

    @property
    def init_qpos(self):
        return np.array([ 0.16833797,  0.07232247, -0.14987418, -2.564317  , -0.18567582,
  2.1922002 ,  0.85425392])


class MountedPanda311(MountedPanda):
    """
    Panda Robot New Init State 311
    """

    @property
    def init_qpos(self):
        return np.array([ 0.13348042, -0.17130541, -0.04324105, -2.4240809 , -0.11290522,
  2.50924257,  0.56805413])


class MountedPanda312(MountedPanda):
    """
    Panda Robot New Init State 312
    """

    @property
    def init_qpos(self):
        return np.array([ 0.02965284,  0.16377301,  0.00109145, -2.47510076, -0.05737381,
  2.19779741,  1.005755  ])


class MountedPanda313(MountedPanda):
    """
    Panda Robot New Init State 313
    """

    @property
    def init_qpos(self):
        return np.array([-0.25097595,  0.01295855, -0.16157383, -2.47485403, -0.04870558,
  2.29353427,  0.6040619 ])


class MountedPanda314(MountedPanda):
    """
    Panda Robot New Init State 314
    """

    @property
    def init_qpos(self):
        return np.array([ 0.07076081,  0.14889396, -0.20870837, -2.41555256, -0.10135259,
  2.29195754,  0.78832978])


class MountedPanda315(MountedPanda):
    """
    Panda Robot New Init State 315
    """

    @property
    def init_qpos(self):
        return np.array([-0.09665962, -0.08752451,  0.27188023, -2.47336296,  0.10845355,
  2.03188107,  0.64128177])


class MountedPanda316(MountedPanda):
    """
    Panda Robot New Init State 316
    """

    @property
    def init_qpos(self):
        return np.array([ 0.08830416,  0.1163798 ,  0.06647274, -2.50311559, -0.00013245,
  1.99302074,  0.8983967 ])


class MountedPanda317(MountedPanda):
    """
    Panda Robot New Init State 317
    """

    @property
    def init_qpos(self):
        return np.array([ 0.1455663 , -0.23562274, -0.07754702, -2.40073318, -0.07204953,
  2.21584172,  1.13180787])


class MountedPanda318(MountedPanda):
    """
    Panda Robot New Init State 318
    """

    @property
    def init_qpos(self):
        return np.array([-0.05388191, -0.23919951, -0.14124357, -2.28227231, -0.03952911,
  2.0851092 ,  1.07360456])


class MountedPanda319(MountedPanda):
    """
    Panda Robot New Init State 319
    """

    @property
    def init_qpos(self):
        return np.array([ 0.16495618, -0.23080589, -0.21688973, -2.32752558, -0.07955317,
  2.31862196,  0.55647444])


class MountedPanda320(MountedPanda):
    """
    Panda Robot New Init State 320
    """

    @property
    def init_qpos(self):
        return np.array([ 0.17129343, -0.39346671,  0.14023691, -2.4418584 , -0.18306525,
  2.14864011,  0.91710575])


class MountedPanda321(MountedPanda):
    """
    Panda Robot New Init State 321
    """

    @property
    def init_qpos(self):
        return np.array([ 0.01330192, -0.08762644,  0.24827029, -2.64108945,  0.20314592,
  2.23986086,  0.67249754])


class MountedPanda322(MountedPanda):
    """
    Panda Robot New Init State 322
    """

    @property
    def init_qpos(self):
        return np.array([-0.20025593, -0.18950622,  0.13873864, -2.46379781,  0.22747849,
  2.00830255,  0.78357838])


class MountedPanda323(MountedPanda):
    """
    Panda Robot New Init State 323
    """

    @property
    def init_qpos(self):
        return np.array([-0.16104227, -0.11427884,  0.11404272, -2.49870222, -0.33486961,
  2.25233446,  0.84155733])


class MountedPanda324(MountedPanda):
    """
    Panda Robot New Init State 324
    """

    @property
    def init_qpos(self):
        return np.array([ 0.05721853,  0.01385887, -0.15158277, -2.34832868,  0.16877174,
  1.97503444,  0.83063745])


class MountedPanda325(MountedPanda):
    """
    Panda Robot New Init State 325
    """

    @property
    def init_qpos(self):
        return np.array([-0.13418089, -0.18273359,  0.09252956, -2.574267  , -0.1798309 ,
  2.23584142,  0.49604183])


class MountedPanda326(MountedPanda):
    """
    Panda Robot New Init State 326
    """

    @property
    def init_qpos(self):
        return np.array([-0.1635114 , -0.05873244, -0.17299372, -2.34425931,  0.06966034,
  1.96686774,  0.68336558])


class MountedPanda327(MountedPanda):
    """
    Panda Robot New Init State 327
    """

    @property
    def init_qpos(self):
        return np.array([ 0.24337463, -0.36101136, -0.09465148, -2.3504236 ,  0.15966492,
  2.17548539,  0.66361801])


class MountedPanda328(MountedPanda):
    """
    Panda Robot New Init State 328
    """

    @property
    def init_qpos(self):
        return np.array([ 0.2059959 ,  0.0908002 ,  0.14176393, -2.4322412 , -0.11705776,
  2.1064824 ,  0.8610717 ])


class MountedPanda329(MountedPanda):
    """
    Panda Robot New Init State 329
    """

    @property
    def init_qpos(self):
        return np.array([ 0.06293293,  0.0497826 ,  0.09783677, -2.67068172,  0.1582276 ,
  2.07662996,  0.72764353])


class MountedPanda330(MountedPanda):
    """
    Panda Robot New Init State 330
    """

    @property
    def init_qpos(self):
        return np.array([ 0.0256766 ,  0.04376643,  0.31915297, -2.38233299, -0.01880041,
  2.28160543,  0.87649311])


class MountedPanda331(MountedPanda):
    """
    Panda Robot New Init State 331
    """

    @property
    def init_qpos(self):
        return np.array([-0.05764416,  0.16167511,  0.0413582 , -2.3551217 ,  0.11970851,
  2.30253786,  0.93620859])


class MountedPanda332(MountedPanda):
    """
    Panda Robot New Init State 332
    """

    @property
    def init_qpos(self):
        return np.array([-0.10326688, -0.25444165, -0.18405655, -2.37106862,  0.1303166 ,
  2.30333791,  1.06554188])


class MountedPanda333(MountedPanda):
    """
    Panda Robot New Init State 333
    """

    @property
    def init_qpos(self):
        return np.array([-0.00127544, -0.00861869, -0.248819  , -2.53282471,  0.15852603,
  2.42027841,  0.71839761])


class MountedPanda334(MountedPanda):
    """
    Panda Robot New Init State 334
    """

    @property
    def init_qpos(self):
        return np.array([-0.01466617, -0.47195392, -0.07313275, -2.26221416,  0.1178232 ,
  2.32930701,  0.79559365])


class MountedPanda335(MountedPanda):
    """
    Panda Robot New Init State 335
    """

    @property
    def init_qpos(self):
        return np.array([ 0.00112944, -0.26662684,  0.111747  , -2.37728128,  0.07867687,
  2.14263596,  0.44106527])


class MountedPanda336(MountedPanda):
    """
    Panda Robot New Init State 336
    """

    @property
    def init_qpos(self):
        return np.array([ 0.21124105, -0.25323069, -0.16611512, -2.31107803, -0.07547642,
  2.43082778,  0.90419383])


class MountedPanda337(MountedPanda):
    """
    Panda Robot New Init State 337
    """

    @property
    def init_qpos(self):
        return np.array([ 0.10202741, -0.36491564,  0.18089095, -2.40560404, -0.05511714,
  2.24815391,  0.52028288])


class MountedPanda338(MountedPanda):
    """
    Panda Robot New Init State 338
    """

    @property
    def init_qpos(self):
        return np.array([ 0.05342818, -0.29561972, -0.17685114, -2.64363411,  0.14547536,
  2.1444809 ,  0.98592115])


class MountedPanda339(MountedPanda):
    """
    Panda Robot New Init State 339
    """

    @property
    def init_qpos(self):
        return np.array([-0.08520807, -0.43255714, -0.17755997, -2.24806735, -0.02479236,
  2.13955852,  0.81084573])


class MountedPanda340(MountedPanda):
    """
    Panda Robot New Init State 340
    """

    @property
    def init_qpos(self):
        return np.array([ 0.08388533,  0.02576331,  0.24215228, -2.50474165, -0.22168149,
  2.15763133,  0.82891045])


class MountedPanda341(MountedPanda):
    """
    Panda Robot New Init State 341
    """

    @property
    def init_qpos(self):
        return np.array([-0.23930931, -0.39870587,  0.08536537, -2.45667825,  0.00814377,
  2.03841154,  0.72818776])


class MountedPanda342(MountedPanda):
    """
    Panda Robot New Init State 342
    """

    @property
    def init_qpos(self):
        return np.array([-0.11899039, -0.0464288 ,  0.13328317, -2.68228492, -0.11966987,
  2.03408155,  0.70171083])


class MountedPanda343(MountedPanda):
    """
    Panda Robot New Init State 343
    """

    @property
    def init_qpos(self):
        return np.array([-0.01567262, -0.43333238, -0.1731581 , -2.63317641,  0.03893219,
  2.23977754,  0.64983015])


class MountedPanda344(MountedPanda):
    """
    Panda Robot New Init State 344
    """

    @property
    def init_qpos(self):
        return np.array([ 0.02603799, -0.2415395 , -0.04000294, -2.19817104,  0.19739139,
  2.03538062,  0.90758152])


class MountedPanda345(MountedPanda):
    """
    Panda Robot New Init State 345
    """

    @property
    def init_qpos(self):
        return np.array([ 0.11368753, -0.32192446,  0.11472172, -2.22911261, -0.15728356,
  2.09819777,  0.92798617])


class MountedPanda346(MountedPanda):
    """
    Panda Robot New Init State 346
    """

    @property
    def init_qpos(self):
        return np.array([ 0.04920635, -0.06108674,  0.06172906, -2.37378697, -0.24041243,
  2.41657013,  0.99737697])


class MountedPanda347(MountedPanda):
    """
    Panda Robot New Init State 347
    """

    @property
    def init_qpos(self):
        return np.array([-0.04446366, -0.16946469, -0.29678024, -2.34491756,  0.11088401,
  2.21469748,  1.00333574])


class MountedPanda348(MountedPanda):
    """
    Panda Robot New Init State 348
    """

    @property
    def init_qpos(self):
        return np.array([-0.10284258,  0.01277131, -0.31389422, -2.45184778, -0.05367334,
  2.29611116,  0.67164272])


class MountedPanda349(MountedPanda):
    """
    Panda Robot New Init State 349
    """

    @property
    def init_qpos(self):
        return np.array([-0.01653866, -0.05712435, -0.05494299, -2.64669769,  0.22008984,
  2.40053295,  0.62283729])


class MountedPanda350(MountedPanda):
    """
    Panda Robot New Init State 350
    """

    @property
    def init_qpos(self):
        return np.array([ 0.19359593, -0.30221872,  0.07395893, -2.36201091, -0.06579655,
  2.00675492,  0.59156835])


class MountedPanda351(MountedPanda):
    """
    Panda Robot New Init State 351
    """

    @property
    def init_qpos(self):
        return np.array([-0.16289113, -0.3089283 , -0.01194373, -2.57185351,  0.12126305,
  2.49117191,  0.68226963])


class MountedPanda352(MountedPanda):
    """
    Panda Robot New Init State 352
    """

    @property
    def init_qpos(self):
        return np.array([ 0.05670943, -0.04267346,  0.1118065 , -2.42660541,  0.17592441,
  2.3409859 ,  0.49222436])


class MountedPanda353(MountedPanda):
    """
    Panda Robot New Init State 353
    """

    @property
    def init_qpos(self):
        return np.array([ 0.17861059,  0.02129988, -0.15685443, -2.43680129,  0.14042334,
  2.18373845,  1.00589394])


class MountedPanda354(MountedPanda):
    """
    Panda Robot New Init State 354
    """

    @property
    def init_qpos(self):
        return np.array([ 0.28651454, -0.25630328, -0.16739031, -2.40777296,  0.0874574 ,
  2.19876991,  0.96154147])


class MountedPanda355(MountedPanda):
    """
    Panda Robot New Init State 355
    """

    @property
    def init_qpos(self):
        return np.array([-0.16559456, -0.07531869, -0.07494907, -2.23189884,  0.24648758,
  2.13191161,  0.71741424])


class MountedPanda356(MountedPanda):
    """
    Panda Robot New Init State 356
    """

    @property
    def init_qpos(self):
        return np.array([ 0.15296634, -0.22992849, -0.00808107, -2.44007952, -0.3416118 ,
  2.22428973,  0.66265896])


class MountedPanda357(MountedPanda):
    """
    Panda Robot New Init State 357
    """

    @property
    def init_qpos(self):
        return np.array([-0.08529315,  0.08866088, -0.21733212, -2.30394118, -0.04825091,
  2.17836339,  0.92210509])


class MountedPanda358(MountedPanda):
    """
    Panda Robot New Init State 358
    """

    @property
    def init_qpos(self):
        return np.array([ 0.07050375, -0.04273066,  0.20073633, -2.51328853, -0.12693669,
  2.00176502,  0.61426358])


class MountedPanda359(MountedPanda):
    """
    Panda Robot New Init State 359
    """

    @property
    def init_qpos(self):
        return np.array([ 0.30815965, -0.03097712, -0.14117778, -2.58278504,  0.05035162,
  2.28393753,  0.84275267])


class MountedPanda360(MountedPanda):
    """
    Panda Robot New Init State 360
    """

    @property
    def init_qpos(self):
        return np.array([-0.04199037, -0.18169929,  0.05400067, -2.43045118, -0.36551139,
  2.18890497,  0.64517436])


class MountedPanda361(MountedPanda):
    """
    Panda Robot New Init State 361
    """

    @property
    def init_qpos(self):
        return np.array([ 0.02169583,  0.20861695,  0.04550399, -2.48342077,  0.1141435 ,
  2.28652178,  0.83741702])


class MountedPanda362(MountedPanda):
    """
    Panda Robot New Init State 362
    """

    @property
    def init_qpos(self):
        return np.array([ 0.08522903,  0.12779352, -0.15882743, -2.37807829, -0.13010544,
  2.30477108,  0.65637339])


class MountedPanda363(MountedPanda):
    """
    Panda Robot New Init State 363
    """

    @property
    def init_qpos(self):
        return np.array([-0.08229487, -0.21922215,  0.19240799, -2.26349996, -0.24562603,
  2.23758391,  0.64548579])


class MountedPanda364(MountedPanda):
    """
    Panda Robot New Init State 364
    """

    @property
    def init_qpos(self):
        return np.array([-0.15903629, -0.18846166, -0.03898273, -2.13793519,  0.0547128 ,
  2.05266375,  0.71402811])


class MountedPanda365(MountedPanda):
    """
    Panda Robot New Init State 365
    """

    @property
    def init_qpos(self):
        return np.array([ 0.10258646,  0.0487232 , -0.15929418, -2.63341473,  0.18791741,
  2.26588456,  0.69818792])


class MountedPanda366(MountedPanda):
    """
    Panda Robot New Init State 366
    """

    @property
    def init_qpos(self):
        return np.array([ 0.07038984, -0.50806275,  0.00044442, -2.40143618,  0.12169039,
  2.09326158,  0.79667983])


class MountedPanda367(MountedPanda):
    """
    Panda Robot New Init State 367
    """

    @property
    def init_qpos(self):
        return np.array([-0.18411198, -0.11022914, -0.17812794, -2.52578858, -0.19811613,
  2.35629488,  0.95618585])


class MountedPanda368(MountedPanda):
    """
    Panda Robot New Init State 368
    """

    @property
    def init_qpos(self):
        return np.array([-0.0291681 , -0.15628363, -0.1800893 , -2.48185336, -0.00624145,
  2.55256994,  0.64715811])


class MountedPanda369(MountedPanda):
    """
    Panda Robot New Init State 369
    """

    @property
    def init_qpos(self):
        return np.array([ 0.22794145, -0.00584755,  0.04368916, -2.17081743,  0.05792896,
  2.28733606,  0.7771284 ])


class MountedPanda370(MountedPanda):
    """
    Panda Robot New Init State 370
    """

    @property
    def init_qpos(self):
        return np.array([ 0.20837935,  0.02569631, -0.10655569, -2.66927127, -0.03456853,
  2.09050199,  0.77457964])


class MountedPanda371(MountedPanda):
    """
    Panda Robot New Init State 371
    """

    @property
    def init_qpos(self):
        return np.array([ 0.17630515, -0.22619187, -0.0061013 , -2.43633063,  0.12088098,
  1.90662456,  0.69895798])


class MountedPanda372(MountedPanda):
    """
    Panda Robot New Init State 372
    """

    @property
    def init_qpos(self):
        return np.array([ 0.03096068,  0.01481744, -0.07208144, -2.71958008,  0.09081006,
  2.13364871,  0.61106912])


class MountedPanda373(MountedPanda):
    """
    Panda Robot New Init State 373
    """

    @property
    def init_qpos(self):
        return np.array([-0.08561613, -0.18634125, -0.0829283 , -2.7470455 ,  0.12301418,
  2.4015751 ,  0.87474101])


class MountedPanda374(MountedPanda):
    """
    Panda Robot New Init State 374
    """

    @property
    def init_qpos(self):
        return np.array([-0.27767339, -0.21021515, -0.00159165, -2.29845054,  0.16312095,
  2.0475634 ,  0.80541071])


class MountedPanda375(MountedPanda):
    """
    Panda Robot New Init State 375
    """

    @property
    def init_qpos(self):
        return np.array([ 0.07721325,  0.08478283,  0.11895817, -2.31136274,  0.01415439,
  2.46491271,  0.85458791])


class MountedPanda376(MountedPanda):
    """
    Panda Robot New Init State 376
    """

    @property
    def init_qpos(self):
        return np.array([-0.17381783,  0.12224463,  0.14087764, -2.53402963, -0.09678968,
  2.28661847,  0.87889591])


class MountedPanda377(MountedPanda):
    """
    Panda Robot New Init State 377
    """

    @property
    def init_qpos(self):
        return np.array([ 0.03179031, -0.11087527,  0.27391935, -2.27193643,  0.0307177 ,
  2.37842847,  0.61902442])


class MountedPanda378(MountedPanda):
    """
    Panda Robot New Init State 378
    """

    @property
    def init_qpos(self):
        return np.array([ 0.06278875,  0.12445232, -0.03440537, -2.57834528, -0.10289284,
  2.05481945,  0.6615709 ])


class MountedPanda379(MountedPanda):
    """
    Panda Robot New Init State 379
    """

    @property
    def init_qpos(self):
        return np.array([-0.05895023, -0.38467219,  0.00248498, -2.30103934, -0.14442988,
  2.4684801 ,  0.86666849])


class MountedPanda380(MountedPanda):
    """
    Panda Robot New Init State 380
    """

    @property
    def init_qpos(self):
        return np.array([ 0.19335241, -0.28445704,  0.16032178, -2.64946069,  0.16702381,
  2.25899184,  0.88919641])


class MountedPanda381(MountedPanda):
    """
    Panda Robot New Init State 381
    """

    @property
    def init_qpos(self):
        return np.array([-0.18821196, -0.30433751,  0.09561586, -2.59642739,  0.07415904,
  2.00787429,  0.92119033])


class MountedPanda382(MountedPanda):
    """
    Panda Robot New Init State 382
    """

    @property
    def init_qpos(self):
        return np.array([-0.26745093, -0.04119127,  0.13575855, -2.59579321,  0.12629663,
  2.28326677,  0.66847238])


class MountedPanda383(MountedPanda):
    """
    Panda Robot New Init State 383
    """

    @property
    def init_qpos(self):
        return np.array([-0.1428127 , -0.43894783,  0.04951092, -2.66617565,  0.01350087,
  2.18944758,  0.88156569])


class MountedPanda384(MountedPanda):
    """
    Panda Robot New Init State 384
    """

    @property
    def init_qpos(self):
        return np.array([-0.0643432 , -0.23121795,  0.15590665, -2.37322178,  0.06734908,
  2.32671786,  0.45828314])


class MountedPanda385(MountedPanda):
    """
    Panda Robot New Init State 385
    """

    @property
    def init_qpos(self):
        return np.array([ 0.20539427,  0.00518812, -0.27593025, -2.5108415 , -0.01066809,
  2.19024945,  0.87600856])


class MountedPanda386(MountedPanda):
    """
    Panda Robot New Init State 386
    """

    @property
    def init_qpos(self):
        return np.array([ 0.19715063, -0.01203925, -0.18073972, -2.35891608,  0.18304186,
  2.28002963,  0.93566536])


class MountedPanda387(MountedPanda):
    """
    Panda Robot New Init State 387
    """

    @property
    def init_qpos(self):
        return np.array([ 0.31825159,  0.00086788,  0.03942952, -2.279311  ,  0.0229148 ,
  2.2305326 ,  0.72982008])


class MountedPanda388(MountedPanda):
    """
    Panda Robot New Init State 388
    """

    @property
    def init_qpos(self):
        return np.array([ 0.17892841, -0.25470433,  0.17537532, -2.38738938, -0.16014679,
  2.26882866,  0.54505583])


class MountedPanda389(MountedPanda):
    """
    Panda Robot New Init State 389
    """

    @property
    def init_qpos(self):
        return np.array([ 0.09639727, -0.02461404, -0.09128819, -2.42555749,  0.30115591,
  2.14881139,  0.62220953])


class MountedPanda390(MountedPanda):
    """
    Panda Robot New Init State 390
    """

    @property
    def init_qpos(self):
        return np.array([ 0.19072143,  0.0915891 ,  0.00047571, -2.31440078,  0.17267633,
  2.13117877,  0.72292904])


class MountedPanda391(MountedPanda):
    """
    Panda Robot New Init State 391
    """

    @property
    def init_qpos(self):
        return np.array([ 0.15737262,  0.05439896,  0.07196675, -2.55708553,  0.25405434,
  2.15886528,  0.74254935])


class MountedPanda392(MountedPanda):
    """
    Panda Robot New Init State 392
    """

    @property
    def init_qpos(self):
        return np.array([ 0.06879471,  0.15127987,  0.17161127, -2.36122792,  0.09310043,
  2.23549672,  0.89755943])


class MountedPanda393(MountedPanda):
    """
    Panda Robot New Init State 393
    """

    @property
    def init_qpos(self):
        return np.array([-0.30311978, -0.19658016,  0.02891785, -2.58252467,  0.04629352,
  2.2022502 ,  0.99575862])


class MountedPanda394(MountedPanda):
    """
    Panda Robot New Init State 394
    """

    @property
    def init_qpos(self):
        return np.array([ 0.33458051, -0.260114  , -0.03208311, -2.46032608, -0.09606775,
  2.26823392,  0.62411194])


class MountedPanda395(MountedPanda):
    """
    Panda Robot New Init State 395
    """

    @property
    def init_qpos(self):
        return np.array([-0.22686339, -0.03608739,  0.12631813, -2.60598729,  0.17965737,
  2.1000231 ,  0.83619698])


class MountedPanda396(MountedPanda):
    """
    Panda Robot New Init State 396
    """

    @property
    def init_qpos(self):
        return np.array([ 0.24554022, -0.17109809,  0.1361102 , -2.59917195, -0.05880514,
  2.16513565,  0.56193702])


class MountedPanda397(MountedPanda):
    """
    Panda Robot New Init State 397
    """

    @property
    def init_qpos(self):
        return np.array([-0.04166884, -0.52607445, -0.11577373, -2.51174772, -0.06038615,
  2.16982693,  0.77083924])


class MountedPanda398(MountedPanda):
    """
    Panda Robot New Init State 398
    """

    @property
    def init_qpos(self):
        return np.array([ 0.09235836,  0.0074273 , -0.12516908, -2.67637895, -0.20073672,
  2.1328987 ,  0.71760496])


class MountedPanda399(MountedPanda):
    """
    Panda Robot New Init State 399
    """

    @property
    def init_qpos(self):
        return np.array([-0.16829958, -0.44433219, -0.09969524, -2.40494269,  0.13005678,
  2.10491513,  0.69512841])


class MountedPanda400(MountedPanda):
    """
    Panda Robot New Init State 400
    """

    @property
    def init_qpos(self):
        return np.array([-0.16074941, -0.16649097, -0.14717239, -2.34518005, -0.30978473,
  2.16436916,  0.83762965])


class MountedPanda401(MountedPanda):
    """
    Panda Robot New Init State 401
    """

    @property
    def init_qpos(self):
        return np.array([ 0.02700254,  0.23452894,  0.05138039, -2.59207806, -0.01402227,
  1.96584821,  0.77355927])


class MountedPanda402(MountedPanda):
    """
    Panda Robot New Init State 402
    """

    @property
    def init_qpos(self):
        return np.array([-0.17960661,  0.03632045, -0.1184807 , -2.46977216, -0.13702606,
  2.60164797,  0.71613833])


class MountedPanda403(MountedPanda):
    """
    Panda Robot New Init State 403
    """

    @property
    def init_qpos(self):
        return np.array([ 0.31731489, -0.02006362,  0.08347186, -2.38074979,  0.06818764,
  1.89296766,  0.73710834])


class MountedPanda404(MountedPanda):
    """
    Panda Robot New Init State 404
    """

    @property
    def init_qpos(self):
        return np.array([ 0.15796328, -0.17017646, -0.26263539, -2.45439206,  0.31850388,
  2.45980106,  0.77380938])


class MountedPanda405(MountedPanda):
    """
    Panda Robot New Init State 405
    """

    @property
    def init_qpos(self):
        return np.array([ 0.17518399, -0.22851987, -0.31372248, -2.61481321,  0.28776425,
  2.17630369,  0.7405855 ])


class MountedPanda406(MountedPanda):
    """
    Panda Robot New Init State 406
    """

    @property
    def init_qpos(self):
        return np.array([-0.04051089,  0.09622894,  0.21968197, -2.1448199 , -0.02148847,
  2.26669378,  0.58048584])


class MountedPanda407(MountedPanda):
    """
    Panda Robot New Init State 407
    """

    @property
    def init_qpos(self):
        return np.array([ 0.29625575, -0.39528029,  0.04348264, -2.6647035 , -0.11811979,
  2.05880716,  0.66346551])


class MountedPanda408(MountedPanda):
    """
    Panda Robot New Init State 408
    """

    @property
    def init_qpos(self):
        return np.array([-0.12996603, -0.29320826,  0.20089138, -2.71094617,  0.05316581,
  2.52036498,  0.66168218])


class MountedPanda409(MountedPanda):
    """
    Panda Robot New Init State 409
    """

    @property
    def init_qpos(self):
        return np.array([-0.18587831, -0.22072314,  0.30544582, -2.39434054, -0.13753812,
  2.352034  ,  0.50000074])


class MountedPanda410(MountedPanda):
    """
    Panda Robot New Init State 410
    """

    @property
    def init_qpos(self):
        return np.array([ 0.170089  , -0.05903601, -0.07683878, -2.65393515, -0.22134375,
  2.36025961,  1.09219077])


class MountedPanda411(MountedPanda):
    """
    Panda Robot New Init State 411
    """

    @property
    def init_qpos(self):
        return np.array([-0.12027439, -0.07452945, -0.39386922, -2.55492806, -0.00586152,
  2.05507436,  0.60865822])


class MountedPanda412(MountedPanda):
    """
    Panda Robot New Init State 412
    """

    @property
    def init_qpos(self):
        return np.array([-0.10230661, -0.11627225, -0.02869958, -2.7875038 , -0.32490008,
  2.13377793,  0.85551765])


class MountedPanda413(MountedPanda):
    """
    Panda Robot New Init State 413
    """

    @property
    def init_qpos(self):
        return np.array([-0.42767588, -0.23882526, -0.0819156 , -2.22848918, -0.05993239,
  2.28636816,  0.80738621])


class MountedPanda414(MountedPanda):
    """
    Panda Robot New Init State 414
    """

    @property
    def init_qpos(self):
        return np.array([-0.20071965, -0.08866495,  0.0202924 , -2.39671372,  0.36988898,
  2.03040892,  0.94788101])


class MountedPanda415(MountedPanda):
    """
    Panda Robot New Init State 415
    """

    @property
    def init_qpos(self):
        return np.array([-0.08186321, -0.07920796, -0.0129616 , -2.56541725, -0.16228476,
  2.32059417,  0.35331683])


class MountedPanda416(MountedPanda):
    """
    Panda Robot New Init State 416
    """

    @property
    def init_qpos(self):
        return np.array([ 0.18286908, -0.26243787, -0.27532704, -2.55423738,  0.30628843,
  2.09632461,  0.87273083])


class MountedPanda417(MountedPanda):
    """
    Panda Robot New Init State 417
    """

    @property
    def init_qpos(self):
        return np.array([-0.15607387,  0.00079418,  0.1463697 , -2.04988228, -0.03748644,
  2.36952346,  0.80633756])


class MountedPanda418(MountedPanda):
    """
    Panda Robot New Init State 418
    """

    @property
    def init_qpos(self):
        return np.array([ 0.38279088, -0.37337512,  0.22006902, -2.5021096 , -0.01896071,
  2.2538552 ,  0.8599081 ])


class MountedPanda419(MountedPanda):
    """
    Panda Robot New Init State 419
    """

    @property
    def init_qpos(self):
        return np.array([-0.02530339, -0.11429373,  0.34356068, -2.23006234,  0.18921737,
  2.12757998,  0.59180797])


class MountedPanda420(MountedPanda):
    """
    Panda Robot New Init State 420
    """

    @property
    def init_qpos(self):
        return np.array([ 0.17915499,  0.00325452,  0.27618145, -2.55786668, -0.04304905,
  1.91928691,  0.71179499])


class MountedPanda421(MountedPanda):
    """
    Panda Robot New Init State 421
    """

    @property
    def init_qpos(self):
        return np.array([ 0.28361397, -0.29379729,  0.18227412, -2.49881274,  0.09222338,
  1.90013075,  0.80964792])


class MountedPanda422(MountedPanda):
    """
    Panda Robot New Init State 422
    """

    @property
    def init_qpos(self):
        return np.array([-0.37122991, -0.18423307, -0.20235805, -2.29492148,  0.09648403,
  2.28467074,  0.97417026])


class MountedPanda423(MountedPanda):
    """
    Panda Robot New Init State 423
    """

    @property
    def init_qpos(self):
        return np.array([-0.1669965 ,  0.14647495,  0.16405314, -2.48622801,  0.19985382,
  2.03205947,  0.6402948 ])


class MountedPanda424(MountedPanda):
    """
    Panda Robot New Init State 424
    """

    @property
    def init_qpos(self):
        return np.array([-0.15653286, -0.29709357, -0.27175285, -2.1300133 , -0.06657888,
  2.31741856,  0.63870043])


class MountedPanda425(MountedPanda):
    """
    Panda Robot New Init State 425
    """

    @property
    def init_qpos(self):
        return np.array([-0.24592787, -0.12455793, -0.25830822, -2.45386859, -0.18483258,
  2.49519429,  0.90850543])


class MountedPanda426(MountedPanda):
    """
    Panda Robot New Init State 426
    """

    @property
    def init_qpos(self):
        return np.array([ 0.02535756,  0.14197325, -0.20292901, -2.31428239,  0.0479104 ,
  2.29005874,  1.09048213])


class MountedPanda427(MountedPanda):
    """
    Panda Robot New Init State 427
    """

    @property
    def init_qpos(self):
        return np.array([ 0.1550251 , -0.09993173, -0.35024144, -2.64778228, -0.0259702 ,
  2.24651359,  0.54620112])


class MountedPanda428(MountedPanda):
    """
    Panda Robot New Init State 428
    """

    @property
    def init_qpos(self):
        return np.array([ 0.3924554 ,  0.08572746,  0.09798353, -2.43287481, -0.03453305,
  2.07184523,  0.79800667])


class MountedPanda429(MountedPanda):
    """
    Panda Robot New Init State 429
    """

    @property
    def init_qpos(self):
        return np.array([ 0.05136492, -0.40849539,  0.06659466, -2.47553749, -0.38018782,
  2.05529471,  0.70297487])


class MountedPanda430(MountedPanda):
    """
    Panda Robot New Init State 430
    """

    @property
    def init_qpos(self):
        return np.array([ 0.30334431, -0.07160139, -0.21460687, -2.36649327, -0.24864593,
  2.40073658,  0.7096902 ])


class MountedPanda431(MountedPanda):
    """
    Panda Robot New Init State 431
    """

    @property
    def init_qpos(self):
        return np.array([-0.01551795,  0.05868151, -0.21331639, -2.6761362 , -0.09671755,
  2.39883851,  0.53360311])


class MountedPanda432(MountedPanda):
    """
    Panda Robot New Init State 432
    """

    @property
    def init_qpos(self):
        return np.array([-0.41426024, -0.22702853,  0.08070329, -2.26903483, -0.06774675,
  2.16657271,  0.95417957])


class MountedPanda433(MountedPanda):
    """
    Panda Robot New Init State 433
    """

    @property
    def init_qpos(self):
        return np.array([-0.29076189, -0.22145591,  0.02961265, -2.37473986, -0.07927959,
  2.23210809,  0.39844056])


class MountedPanda434(MountedPanda):
    """
    Panda Robot New Init State 434
    """

    @property
    def init_qpos(self):
        return np.array([-0.23548344,  0.00120196, -0.35140834, -2.42349631,  0.15986529,
  2.10420779,  0.8463803 ])


class MountedPanda435(MountedPanda):
    """
    Panda Robot New Init State 435
    """

    @property
    def init_qpos(self):
        return np.array([-0.31938812, -0.22958102,  0.20244023, -2.39252143, -0.19458272,
  2.30113636,  1.02247115])


class MountedPanda436(MountedPanda):
    """
    Panda Robot New Init State 436
    """

    @property
    def init_qpos(self):
        return np.array([-0.32981276,  0.02544474, -0.05409965, -2.37614087,  0.19262372,
  2.00606515,  0.67125699])


class MountedPanda437(MountedPanda):
    """
    Panda Robot New Init State 437
    """

    @property
    def init_qpos(self):
        return np.array([-0.05776909, -0.31794931,  0.34734418, -2.61612812, -0.11122363,
  2.26767205,  0.54472098])


class MountedPanda438(MountedPanda):
    """
    Panda Robot New Init State 438
    """

    @property
    def init_qpos(self):
        return np.array([-0.11320714, -0.07021169, -0.09523379, -2.47131613, -0.15144579,
  1.79345118,  0.87740833])


class MountedPanda439(MountedPanda):
    """
    Panda Robot New Init State 439
    """

    @property
    def init_qpos(self):
        return np.array([ 0.30714781, -0.15781674,  0.03942565, -2.32335736, -0.22215781,
  2.52472575,  0.82080968])


class MountedPanda440(MountedPanda):
    """
    Panda Robot New Init State 440
    """

    @property
    def init_qpos(self):
        return np.array([ 0.29736264, -0.29461321,  0.33129357, -2.30334594,  0.10077689,
  2.26939695,  0.74011041])


class MountedPanda441(MountedPanda):
    """
    Panda Robot New Init State 441
    """

    @property
    def init_qpos(self):
        return np.array([ 0.23169059, -0.13394934, -0.37962475, -2.66020327, -0.03586637,
  2.2865789 ,  0.77447715])


class MountedPanda442(MountedPanda):
    """
    Panda Robot New Init State 442
    """

    @property
    def init_qpos(self):
        return np.array([ 0.1989986 , -0.47955737,  0.02344092, -2.741385  ,  0.12402404,
  2.21133169,  0.85390005])


class MountedPanda443(MountedPanda):
    """
    Panda Robot New Init State 443
    """

    @property
    def init_qpos(self):
        return np.array([-0.20005558,  0.13525749,  0.24123966, -2.30946193, -0.02999324,
  2.36103391,  0.62169798])


class MountedPanda444(MountedPanda):
    """
    Panda Robot New Init State 444
    """

    @property
    def init_qpos(self):
        return np.array([ 0.15874403,  0.14501545,  0.19864638, -2.57997152,  0.06705916,
  1.98681632,  0.89160229])


class MountedPanda445(MountedPanda):
    """
    Panda Robot New Init State 445
    """

    @property
    def init_qpos(self):
        return np.array([ 0.08618519, -0.51867002, -0.22220996, -2.49875237, -0.0575443 ,
  1.99542016,  0.85979664])


class MountedPanda446(MountedPanda):
    """
    Panda Robot New Init State 446
    """

    @property
    def init_qpos(self):
        return np.array([ 0.05556324, -0.20396127, -0.16699258, -2.21288501,  0.36576319,
  2.119909  ,  0.65013952])


class MountedPanda447(MountedPanda):
    """
    Panda Robot New Init State 447
    """

    @property
    def init_qpos(self):
        return np.array([ 0.09765863, -0.2322184 , -0.02771402, -2.29754457, -0.37960543,
  2.48884827,  0.79984802])


class MountedPanda448(MountedPanda):
    """
    Panda Robot New Init State 448
    """

    @property
    def init_qpos(self):
        return np.array([-0.08687744, -0.17987716, -0.00793219, -2.81010133,  0.31562228,
  2.23547717,  0.87892801])


class MountedPanda449(MountedPanda):
    """
    Panda Robot New Init State 449
    """

    @property
    def init_qpos(self):
        return np.array([ 0.17377519, -0.34752548,  0.08646701, -2.49347498,  0.1546401 ,
  1.83827085,  0.76730123])


class MountedPanda450(MountedPanda):
    """
    Panda Robot New Init State 450
    """

    @property
    def init_qpos(self):
        return np.array([-0.18709501, -0.02687769,  0.14588571, -2.66422046, -0.22103452,
  2.19240544,  0.50711094])


class MountedPanda451(MountedPanda):
    """
    Panda Robot New Init State 451
    """

    @property
    def init_qpos(self):
        return np.array([ 0.10977639, -0.20917399, -0.37900815, -2.49346613, -0.13540357,
  2.01406925,  0.62406459])


class MountedPanda452(MountedPanda):
    """
    Panda Robot New Init State 452
    """

    @property
    def init_qpos(self):
        return np.array([ 0.03085042,  0.21360568,  0.08831009, -2.61561607,  0.12855344,
  2.09730036,  0.58953845])


class MountedPanda453(MountedPanda):
    """
    Panda Robot New Init State 453
    """

    @property
    def init_qpos(self):
        return np.array([-0.24270865, -0.52936532, -0.08198278, -2.41136619, -0.02031545,
  2.13929375,  0.58655186])


class MountedPanda454(MountedPanda):
    """
    Panda Robot New Init State 454
    """

    @property
    def init_qpos(self):
        return np.array([ 0.11766453, -0.41583506,  0.15012171, -2.63847521, -0.14555774,
  1.94300428,  0.68840249])


class MountedPanda455(MountedPanda):
    """
    Panda Robot New Init State 455
    """

    @property
    def init_qpos(self):
        return np.array([ 0.23292426, -0.03789501,  0.23366691, -2.31989931, -0.26357859,
  2.14775307,  0.59907089])


class MountedPanda456(MountedPanda):
    """
    Panda Robot New Init State 456
    """

    @property
    def init_qpos(self):
        return np.array([ 0.22399425, -0.36250953,  0.15044861, -2.29427408,  0.30800551,
  2.15778394,  0.90531329])


class MountedPanda457(MountedPanda):
    """
    Panda Robot New Init State 457
    """

    @property
    def init_qpos(self):
        return np.array([ 0.28818443, -0.23904873, -0.19388352, -2.68205246,  0.09001922,
  1.98890059,  0.73837234])


class MountedPanda458(MountedPanda):
    """
    Panda Robot New Init State 458
    """

    @property
    def init_qpos(self):
        return np.array([ 0.11760703, -0.11123797,  0.11441527, -2.38481686,  0.34392652,
  1.97064491,  0.60333326])


class MountedPanda459(MountedPanda):
    """
    Panda Robot New Init State 459
    """

    @property
    def init_qpos(self):
        return np.array([-0.13772378, -0.01305867,  0.16049282, -2.36416817,  0.21095351,
  2.01677064,  1.08257489])


class MountedPanda460(MountedPanda):
    """
    Panda Robot New Init State 460
    """

    @property
    def init_qpos(self):
        return np.array([ 0.23945021, -0.04539901,  0.2800954 , -2.70501549, -0.08244225,
  2.11116389,  0.89881334])


class MountedPanda461(MountedPanda):
    """
    Panda Robot New Init State 461
    """

    @property
    def init_qpos(self):
        return np.array([-0.21780149, -0.00393124,  0.2842522 , -2.30041087,  0.05885058,
  2.02186511,  0.60975942])


class MountedPanda462(MountedPanda):
    """
    Panda Robot New Init State 462
    """

    @property
    def init_qpos(self):
        return np.array([ 0.18628841, -0.04485502, -0.00581688, -2.50689783, -0.30199969,
  2.06711361,  1.07034521])


class MountedPanda463(MountedPanda):
    """
    Panda Robot New Init State 463
    """

    @property
    def init_qpos(self):
        return np.array([-0.28132158, -0.11940786,  0.22133976, -2.72030936,  0.04058168,
  2.43281627,  0.78814432])


class MountedPanda464(MountedPanda):
    """
    Panda Robot New Init State 464
    """

    @property
    def init_qpos(self):
        return np.array([ 0.39047246, -0.38818192,  0.14412352, -2.46812078,  0.10508062,
  2.28682684,  0.68559444])


class MountedPanda465(MountedPanda):
    """
    Panda Robot New Init State 465
    """

    @property
    def init_qpos(self):
        return np.array([ 0.04829949, -0.55290245,  0.12723582, -2.38171045,  0.134705  ,
  2.44992852,  0.86293585])


class MountedPanda466(MountedPanda):
    """
    Panda Robot New Init State 466
    """

    @property
    def init_qpos(self):
        return np.array([ 0.03367171, -0.36136958, -0.16194062, -2.17476465,  0.32658559,
  2.17261203,  0.795858  ])


class MountedPanda467(MountedPanda):
    """
    Panda Robot New Init State 467
    """

    @property
    def init_qpos(self):
        return np.array([-0.16657899,  0.05491834,  0.12253228, -2.0871346 , -0.16173762,
  2.15714445,  0.82801746])


class MountedPanda468(MountedPanda):
    """
    Panda Robot New Init State 468
    """

    @property
    def init_qpos(self):
        return np.array([ 0.14174919, -0.20832464, -0.19140531, -2.25134019,  0.15811603,
  2.02585009,  1.08259234])


class MountedPanda469(MountedPanda):
    """
    Panda Robot New Init State 469
    """

    @property
    def init_qpos(self):
        return np.array([ 0.21994188, -0.13579601,  0.01496062, -2.34046088, -0.16195932,
  2.23239669,  0.38085211])


class MountedPanda470(MountedPanda):
    """
    Panda Robot New Init State 470
    """

    @property
    def init_qpos(self):
        return np.array([ 0.17746955, -0.47721711,  0.30442071, -2.511442  , -0.04285949,
  2.09964519,  0.72707447])


class MountedPanda471(MountedPanda):
    """
    Panda Robot New Init State 471
    """

    @property
    def init_qpos(self):
        return np.array([ 0.18091502, -0.47832467,  0.16286014, -2.40148702, -0.15393974,
  2.08961025,  0.99920005])


class MountedPanda472(MountedPanda):
    """
    Panda Robot New Init State 472
    """

    @property
    def init_qpos(self):
        return np.array([-0.19957705, -0.40448595,  0.2400637 , -2.37313592, -0.19797679,
  2.27247728,  0.56888287])


class MountedPanda473(MountedPanda):
    """
    Panda Robot New Init State 473
    """

    @property
    def init_qpos(self):
        return np.array([ 0.20440675,  0.03604398,  0.09878592, -2.3774422 ,  0.32702738,
  2.22669396,  0.56594108])


class MountedPanda474(MountedPanda):
    """
    Panda Robot New Init State 474
    """

    @property
    def init_qpos(self):
        return np.array([ 0.0858346 ,  0.23732836, -0.19378538, -2.51362486,  0.07952737,
  2.21645816,  0.59781078])


class MountedPanda475(MountedPanda):
    """
    Panda Robot New Init State 475
    """

    @property
    def init_qpos(self):
        return np.array([ 0.1677494 ,  0.18587416,  0.02932719, -2.57978119, -0.2316133 ,
  2.26532689,  0.95045897])


class MountedPanda476(MountedPanda):
    """
    Panda Robot New Init State 476
    """

    @property
    def init_qpos(self):
        return np.array([ 0.09962729, -0.1615414 , -0.12355119, -2.67358883,  0.17502033,
  2.41834053,  0.46130866])


class MountedPanda477(MountedPanda):
    """
    Panda Robot New Init State 477
    """

    @property
    def init_qpos(self):
        return np.array([ 0.26861248, -0.06404474,  0.2364696 , -2.67135771, -0.23076388,
  2.20158297,  0.8703465 ])


class MountedPanda478(MountedPanda):
    """
    Panda Robot New Init State 478
    """

    @property
    def init_qpos(self):
        return np.array([ 0.28498223, -0.04902206, -0.03849998, -2.60742499,  0.19513246,
  2.07370728,  1.04374526])


class MountedPanda479(MountedPanda):
    """
    Panda Robot New Init State 479
    """

    @property
    def init_qpos(self):
        return np.array([-0.07432525,  0.25466861,  0.17744184, -2.48040188, -0.13574665,
  2.13765435,  0.89734072])


class MountedPanda480(MountedPanda):
    """
    Panda Robot New Init State 480
    """

    @property
    def init_qpos(self):
        return np.array([ 0.01562826, -0.28131555, -0.03354911, -2.42868147, -0.43118104,
  2.31955543,  0.58695289])


class MountedPanda481(MountedPanda):
    """
    Panda Robot New Init State 481
    """

    @property
    def init_qpos(self):
        return np.array([-0.15508126, -0.38792383, -0.24121329, -2.62685586, -0.09533753,
  2.48939451,  0.71468647])


class MountedPanda482(MountedPanda):
    """
    Panda Robot New Init State 482
    """

    @property
    def init_qpos(self):
        return np.array([ 0.0335556 , -0.08543466,  0.02474244, -2.45921722, -0.22564711,
  2.50065776,  1.12656007])


class MountedPanda483(MountedPanda):
    """
    Panda Robot New Init State 483
    """

    @property
    def init_qpos(self):
        return np.array([-0.26135739, -0.32009368, -0.04346128, -2.5944134 ,  0.07858863,
  2.113634  ,  1.12168348])


class MountedPanda484(MountedPanda):
    """
    Panda Robot New Init State 484
    """

    @property
    def init_qpos(self):
        return np.array([ 0.27368821, -0.11217   ,  0.23608783, -2.19036853, -0.12964257,
  2.03911682,  0.80336201])


class MountedPanda485(MountedPanda):
    """
    Panda Robot New Init State 485
    """

    @property
    def init_qpos(self):
        return np.array([ 0.15430391, -0.13613411, -0.2824124 , -2.58836924, -0.01324724,
  2.21320182,  1.13864586])


class MountedPanda486(MountedPanda):
    """
    Panda Robot New Init State 486
    """

    @property
    def init_qpos(self):
        return np.array([-0.21610533, -0.08220515,  0.23218477, -2.09042959, -0.06305426,
  2.30337662,  0.87423606])


class MountedPanda487(MountedPanda):
    """
    Panda Robot New Init State 487
    """

    @property
    def init_qpos(self):
        return np.array([ 0.03647412, -0.1794805 ,  0.08134683, -2.91707871,  0.1272645 ,
  2.25029521,  0.7438875 ])


class MountedPanda488(MountedPanda):
    """
    Panda Robot New Init State 488
    """

    @property
    def init_qpos(self):
        return np.array([ 0.32708156, -0.26888953, -0.06441016, -2.12635408, -0.13296054,
  2.13884566,  0.761904  ])


class MountedPanda489(MountedPanda):
    """
    Panda Robot New Init State 489
    """

    @property
    def init_qpos(self):
        return np.array([ 0.23280969, -0.10349433, -0.35794382, -2.59196147, -0.16818389,
  2.20536875,  0.9033195 ])


class MountedPanda490(MountedPanda):
    """
    Panda Robot New Init State 490
    """

    @property
    def init_qpos(self):
        return np.array([ 0.07462238, -0.47793933,  0.18984388, -2.39953045,  0.24704842,
  2.30601255,  0.98191228])


class MountedPanda491(MountedPanda):
    """
    Panda Robot New Init State 491
    """

    @property
    def init_qpos(self):
        return np.array([ 0.07956451,  0.06854759,  0.08906448, -2.46129591,  0.38754041,
  2.37836835,  0.88324438])


class MountedPanda492(MountedPanda):
    """
    Panda Robot New Init State 492
    """

    @property
    def init_qpos(self):
        return np.array([-0.25391753, -0.41734292,  0.058235  , -2.4156464 , -0.12217813,
  2.25480035,  1.10145146])


class MountedPanda493(MountedPanda):
    """
    Panda Robot New Init State 493
    """

    @property
    def init_qpos(self):
        return np.array([ 0.20089866, -0.49066963, -0.0322144 , -2.67714868, -0.11200477,
  2.10726049,  0.6474085 ])


class MountedPanda494(MountedPanda):
    """
    Panda Robot New Init State 494
    """

    @property
    def init_qpos(self):
        return np.array([-0.15674833, -0.17697962, -0.07001182, -2.81044343,  0.25638664,
  2.36272385,  0.73840362])


class MountedPanda495(MountedPanda):
    """
    Panda Robot New Init State 495
    """

    @property
    def init_qpos(self):
        return np.array([ 0.00314505, -0.35684467,  0.2833898 , -2.12752193,  0.02618671,
  2.397805  ,  0.81469215])


class MountedPanda496(MountedPanda):
    """
    Panda Robot New Init State 496
    """

    @property
    def init_qpos(self):
        return np.array([-0.44625432, -0.29852109,  0.04404302, -2.59634433, -0.07299552,
  2.19623924,  0.75839757])


class MountedPanda497(MountedPanda):
    """
    Panda Robot New Init State 497
    """

    @property
    def init_qpos(self):
        return np.array([ 0.00825762, -0.32862885, -0.11329927, -2.64758099,  0.15450948,
  1.8940285 ,  0.96768268])


class MountedPanda498(MountedPanda):
    """
    Panda Robot New Init State 498
    """

    @property
    def init_qpos(self):
        return np.array([ 0.21692422, -0.20832005, -0.40731687, -2.50113229, -0.01004299,
  2.06862234,  0.70478115])


class MountedPanda499(MountedPanda):
    """
    Panda Robot New Init State 499
    """

    @property
    def init_qpos(self):
        return np.array([ 0.29805135, -0.03152291, -0.18008656, -2.75577777, -0.06369642,
  2.1851751 ,  0.88204782])


class MountedPanda500(MountedPanda):
    """
    Panda Robot New Init State 500
    """

    @property
    def init_qpos(self):
        return np.array([-0.11827633, -0.04802337,  0.42756544, -2.2644154 , -0.01062499,
  2.18628575,  0.7065925 ])

