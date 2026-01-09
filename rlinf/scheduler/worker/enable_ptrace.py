import ctypes

# see https://zhuanlan.zhihu.com/p/1916898559854376403
def enable_ptrace():
    # Constants from prctl.h
    PR_SET_PTRACER = 0x59616d61
    PR_SET_PTRACER_ANY = -1  # Allow any process with the same UID to ptrace
    
    libc = ctypes.CDLL("libc.so.6", use_errno=True)
    
    result = libc.prctl(PR_SET_PTRACER, PR_SET_PTRACER_ANY, 0, 0, 0)
    if result != 0:
        errno = ctypes.get_errno()
        raise OSError(errno, f"prctl(PR_SET_PTRACER, ANY) failed: {ctypes.cast(libc.strerror(errno), ctypes.c_char_p).value.decode()}")
    else:
        print("Allowed ptrace from any same-UID process (PR_SET_PTRACER_ANY)")
