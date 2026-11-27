# The legacy ascendc_library auto_gen pass renames the original global to
# *_origin, including SK_BIND's target, and registers every entry at key zero.
# Only the two SK-enabled 950 WeLM kernels use this raw AIV ELF build. These
# device flags follow the CANN 9.1 aiv_obj command from the crash capture;
# ordinary/non-950 targets keep the SDK build pipeline unchanged.
function(sgl_welm_named_kernel target source kernel_name sk_define config_header)
    find_program(SGL_WELM_BISHENG bisheng HINTS
        "${ASCEND_HOME_PATH}/${CMAKE_SYSTEM_PROCESSOR}-linux/ccec_compiler/bin"
        "${ASCEND_HOME_PATH}/ccec_compiler/bin"
        "${ASCEND_HOME_PATH}/compiler/ccec_compiler/bin"
        "${ASCEND_HOME_PATH}/bin" NO_DEFAULT_PATH REQUIRED)
    get_filename_component(compiler_bin "${SGL_WELM_BISHENG}" DIRECTORY)
    find_program(SGL_WELM_DEVICE_LINKER ld.lld HINTS "${compiler_bin}" NO_DEFAULT_PATH REQUIRED)
    find_path(SGL_WELM_ASC_INCLUDE kernel_operator.h HINTS
        "${ASCEND_HOME_PATH}/${CMAKE_SYSTEM_PROCESSOR}-linux/asc/include"
        "${ASCEND_HOME_PATH}/asc/include"
        "${ASCEND_HOME_PATH}/include" NO_DEFAULT_PATH REQUIRED)
    get_filename_component(asc_root "${SGL_WELM_ASC_INCLUDE}" DIRECTORY)
    set(includes)
    foreach(relative "" impl/adv_api impl/basic_api impl/c_api impl/basic_api/reg_compute
            impl/simt_api impl/utils include include/adv_api include/basic_api
            include/aicpu_api include/c_api include/basic_api/reg_compute include/simt_api include/utils)
        if(IS_DIRECTORY "${asc_root}/${relative}")
            list(APPEND includes "-I${asc_root}/${relative}")
        endif()
    endforeach()
    # kernel_operator.h also includes the legacy kernel_tiling header on 9.1.
    find_path(SGL_WELM_TIKCFW kernel_tiling/kernel_tiling.h HINTS
        "${ASCEND_HOME_PATH}/${CMAKE_SYSTEM_PROCESSOR}-linux/tikcpp/tikcfw"
        "${ASCEND_HOME_PATH}/tikcpp/tikcfw"
        "${ASCEND_HOME_PATH}/compiler/tikcpp/tikcfw"
        "${ASCEND_HOME_PATH}/tools/tikcpp/tikcfw" NO_DEFAULT_PATH REQUIRED)
    list(APPEND includes "-I${SGL_WELM_TIKCFW}" "-I${SGL_WELM_TIKCFW}/interface" "-I${SGL_WELM_TIKCFW}/impl")
    find_file(SGL_WELM_DEVKIT_VERSION asc_devkit_version.h HINTS
        "${ASCEND_HOME_PATH}/include/version"
        "${ASCEND_HOME_PATH}/${CMAKE_SYSTEM_PROCESSOR}-linux/include/version"
        NO_DEFAULT_PATH REQUIRED)

    set(directory "${CMAKE_CURRENT_BINARY_DIR}/welm_named_kernels")
    set(object "${directory}/${kernel_name}.raw.o")
    set(binary "${directory}/${kernel_name}.o")
    set(candidate "${directory}/${kernel_name}.unchecked.o")
    set(checker "${PROJECT_SOURCE_DIR}/cmake/check_welm_kernel_elf.py")
    add_custom_command(OUTPUT "${binary}"
        BYPRODUCTS "${object}"
        COMMAND "${CMAKE_COMMAND}" -E make_directory "${directory}"
        # A failed compiler/linker/check must not leave an up-to-date OUTPUT.
        COMMAND "${CMAKE_COMMAND}" -E rm -f "${binary}"
        COMMAND "${SGL_WELM_BISHENG}" ${includes}
            -include "${SGL_WELM_DEVKIT_VERSION}"
            -O3 -std=c++17 -DNDEBUG -D__DAV_C310__ -DTILING_KEY_VAR=0
            -DSGL_WELM_NAMED_LAUNCH -D${sk_define} ${ARGN}
            --cce-disable-kernel-global-attr-check
            --cce-aicore-arch=dav-c310-vec --cce-aicore-only --cce-aicore-lang
            --cce-auto-sync=off --cce-mask-opt
            -mllvm -cce-aicore-stack-size=0x8000
            -mllvm -cce-aicore-function-stack-size=0x8000
            -mllvm -cce-aicore-record-overflow=false
            -mllvm -cce-aicore-addr-transform
            -mllvm -cce-aicore-jump-expand=true
            -mllvm -cce-aicore-dcci-insert-for-scalar=false
            -MMD -MF "${directory}/${kernel_name}.d"
            -MT "${binary}" -c "${source}" -o "${object}"
        COMMAND "${SGL_WELM_DEVICE_LINKER}" -m aicorelinux -Ttext=0 "${object}" -o "${candidate}"
        COMMAND "${PYTHON_EXECUTABLE}" "${checker}" "${candidate}" --kernel "${kernel_name}"
        COMMAND "${CMAKE_COMMAND}" -E rename "${candidate}" "${binary}"
        DEPENDS "${source}" "${config_header}" "${checker}"
            "${PROJECT_OP_SRC_BASE}/utils/welm_kernel_args.h"
            "${CMAKE_CURRENT_FUNCTION_LIST_FILE}"
        DEPFILE "${directory}/${kernel_name}.d"
        COMMENT "Building and checking named WeLM kernel ${kernel_name}"
        VERBATIM)
    # Re-copy if the wheel staging directory was cleaned between builds. The
    # existing package_data lib/** includes this payload beside the extension.
    add_custom_target(${target}
        COMMAND "${CMAKE_COMMAND}" -E make_directory "${CMAKE_LIBRARY_OUTPUT_DIRECTORY}"
        COMMAND "${CMAKE_COMMAND}" -E copy_if_different "${binary}" "${CMAKE_LIBRARY_OUTPUT_DIRECTORY}/"
        DEPENDS "${binary}" VERBATIM)
endfunction()
