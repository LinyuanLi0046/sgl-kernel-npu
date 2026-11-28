```
yum install -y dos2unix pigz

cd /.../sgl-kernel-npu
TEST_DIR="$PWD/tests/python/sgl_kernel_npu/swiglu_group_cann91"
INSTALL_DIR=/usr/local/welm-ops/swiglu_group

bash "$TEST_DIR/build_install.sh" \
  --ops-nn /data2/hw_lly/ops-nn \
  --prefix "$INSTALL_DIR" \
  --jobs 16

source "$INSTALL_DIR/env.sh"
```