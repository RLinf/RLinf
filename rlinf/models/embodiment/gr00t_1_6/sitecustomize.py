# Auto-apply LIBERO reward patch for all Python processes
import sys
import types

# Add repo to path if not already
repo_path = '/workspace/test/RLinf'
if repo_path not in sys.path:
    sys.path.insert(0, repo_path)

# Only try once
if 'RLINF_REWARD_PATCHED' not in sys.modules.get('builtins', {}).__dict__:
    try:
        from rlinf.models.embodiment.gr00t_1_6.patch_libero_reward import patch_libero_reward
        patch_libero_reward()
        import builtins
        builtins.RLINF_REWARD_PATCHED = True
    except Exception:
        pass
