"""Check if custom transformers_replace is installed for AdaRMS support."""

def check_transformers_replace():
    """Check if AdaRMS support is present in transformers. Raises error if not."""
    from transformers.models.gemma import modeling_gemma
    
    if not hasattr(modeling_gemma, '_gated_residual'):
        raise RuntimeError(
            "\n" + "=" * 70 + "\n"
            "ERROR: transformers_replace is NOT installed!\n"
            "=" * 70 + "\n\n"
            "PI0.5 requires custom AdaRMS modifications in transformers.\n\n"
            "Run this command from the project root:\n\n"
            "    cp -r ./rlinf/models/embodiment/vla_lib_value_model/openpi/transformers_replace/* \\\n"
            "        $(python -c 'import transformers; import os; print(os.path.dirname(transformers.__file__))')/\n\n"
            + "=" * 70
        )

# Legacy alias
def check_whether_transformers_replace_is_installed_correctly():
    try:
        check_transformers_replace()
        return True
    except RuntimeError:
        return False