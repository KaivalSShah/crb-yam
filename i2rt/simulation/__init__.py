"""MuJoCo replica of the real YAM teleop/recording setup (table, objects, cameras, robot)."""

import warnings

# numpy 2.x linked against Apple's Accelerate raises spurious floating-point warnings ("divide by
# zero / overflow / invalid value encountered in matmul") on perfectly finite inputs. The results
# are correct; the warnings would only bury real output from the calibration scripts.
warnings.filterwarnings("ignore", message=r".*encountered in matmul", category=RuntimeWarning)
