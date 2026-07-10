from __future__ import annotations

from copy import deepcopy

from config.model.UniField_legacy.DrivAer1210VolumeRANS import CONFIG as BASE_CONFIG


CONFIG = deepcopy(BASE_CONFIG)

# Channel layout:
# 0:7    U, p, k, omega, nut
# 7:15   gradU8 = ux, uy, uz, vx, vy, vz, wx, wy; wz is hard-set to -(ux + vy)
# 15:33  Hessian(U), 6 symmetric entries per velocity component: xx, yy, zz, xy, xz, yz
# 33:36  grad(p)
CONFIG["decoder_kwargs"]["out_channels"] = 36
