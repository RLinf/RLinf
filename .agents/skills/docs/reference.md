# Docs check reference

Background for the rules [check.py](check.py) applies and for the judgment items in [SKILL.md](SKILL.md#check). The writing rules themselves are in [docs/STYLE_GUIDE.md](../../../docs/STYLE_GUIDE.md).

## Source of truth for models and envs

Docs name models by the string values of `SupportedModel` in `rlinf/config.py` (`openpi`, `openvla_oft`, `gr00t`) and envs by the string values of `SupportedEnvType` in `rlinf/envs/__init__.py` (`maniskill`, `libero`). Read the enums rather than a list on another page.

## The build gate

Both languages are separate Read the Docs projects, configured by
`docs/source-en/.readthedocs.yaml` and `docs/source-zh/.readthedocs.yaml`, and
both set `fail_on_warning: true`. A single Sphinx warning turns
`docs/readthedocs.org:rlinf` or `docs/readthedocs.org:rlinf-cn` red. They build
independently, so "the English build is green" says nothing about the Chinese
one. `check.py` builds both through [build_docs.py](build_docs.py), which can also
run alone (`--lang zh` to iterate on a Chinese-only failure, `--keep-output` to
inspect the HTML).

To read a failed Read the Docs check without leaving the terminal:

```bash
gh api repos/RLinf/RLinf/commits/<sha>/status \
  --jq '.statuses[] | "\(.context) \(.state) \(.target_url)"'
```

## Inline markup next to CJK punctuation

docutils only recognises inline markup when the *start*-string is preceded by
whitespace, an opener or a delimiter, and the *end*-string is followed by
whitespace, a closer or a delimiter. Full-width punctuation splits across those
classes:

| Character | docutils class | Before markup | After markup |
|-----------|----------------|---------------|--------------|
| `（` `《` `“` | opener | ok | **rejected** |
| `）` `》` `”` | closer | **rejected** | ok |
| `，` `、` `：` `；` `。` `！` `？` `—` | delimiter | ok | ok |
| any CJK ideograph | none | **rejected** | **rejected** |

Two distinct failures follow, and only the first is visible in CI:

1. **Build break.** No later end-string candidate satisfies the rule, so Sphinx
   emits `WARNING: Inline literal start-string without end-string. [docutils]`
   and the CN build fails.

   ```rst
   导出为 ``MUJOCO_EGL_DEVICE_ID``（MuJoCo）和 ``EGL_DEVICE_ID``（其他）。
   ```

2. **Silent mis-render.** A *later* ` `` ` in the same paragraph does satisfy
   the rule, so docutils closes the literal there and swallows the prose in
   between into the code span. No warning, wrong page.

   ```rst
   镜像提供两者：``reason``（SGLang，默认激活）与 ``reason-vllm``。
   ..                 ^ renders as one literal: "reason``（SGLang，默认激活）与 ``reason-vllm"
   ```

   The same happens to `**粗体**` written directly against a Chinese character:
   docutils sees no markup and prints the asterisks.

The fix in every case is an escaped space — a backslash followed by a space —
between the markup and the adjacent character. It suppresses the space in the
output, so Chinese typography is unaffected:

```rst
导出为 ``MUJOCO_EGL_DEVICE_ID``\ （MuJoCo）和 ``EGL_DEVICE_ID``\ （其他）。
CUDA 设备 0 通常\ **并不是** EGL 设备 0。
```

[check_rst_markup.py](check_rst_markup.py) reports both classes, and `check.py`
maps the build break to critical and the silent mis-render to major. Inline
markup does **not** nest: `` **stuck in ``reach``** `` is a strong containing
literal backtick characters, not a nested literal, and is legal.

## Hard-wrapped Chinese prose

reStructuredText joins source lines inside a paragraph with a space, so
hard-wrapping Chinese prose inserts a visible gap even when no space was typed:

```rst
定义奖励、成功条件
或复位吗？
```

This renders as `成功条件 或复位`. Keep each Chinese prose paragraph and prose
list item on one source line, and keep the newlines that define headings,
directives, tables, and code blocks.

## Code-font names

[check_doc_symbols.py](check_doc_symbols.py) reads every single-word literal on a
Concepts, Extending, or Reference page and reports the ones no file in the
repository spells. Example and guide pages name shell variables, metrics, and
dataset ids, so they are out of its scope. An illustrative name, such as the
`ExampleArm` of a tutorial, goes into `ALLOWED`; a name another project owns goes
into `EXTERNAL`.

## Asset references

Every reference to a downloadable model, adapter, or dataset in changed docs and
configs passes three checks. In `examples/embodiment/config/*.yaml` these are the
weight fields `model_path`, `lora_path`, `backbone_model_path`, and
`wan_wm_hf_ckpt_path`.

1. **Form.** The path is `/path/to/<repo-name>` with a leading slash, and its last
   component equals the repo in the `# https://huggingface.co/<org>/<repo>`
   comment above it, because the docs download with
   `hf download <org>/<repo> --local-dir <repo>`. Flag stale or placeholder
   basenames (`RLinf-Pi0-SFT`, `model`, `openpi`) and a missing leading slash.
2. **Exists.** `curl -sL -o /dev/null -w '%{http_code}' https://huggingface.co/api/models/<org>/<repo>`
   returns `200` (`/api/datasets/` for datasets); flag `401` and `404`. When the
   returned `id` differs from the requested name, the repo was renamed: point to
   the canonical `id`, since a `200` through a redirect is still a stale reference.
3. **Matches the docs.** The repo is the one the env's recipe page prescribes for
   that env, task suite, and model family, not just any repo that exists. For
   example, LIBERO spatial, object, or goal with π₀ uses
   `RLinf-Pi0-LIBERO-Spatial-Object-Goal-SFT`; LIBERO-10 (Long) with π₀ uses
   `RLinf-Pi0-LIBERO-Long-SFT`; any LIBERO suite with π₀.₅ uses
   `RLinf-Pi05-LIBERO-SFT`; GR00T N1.5 uses the per-suite
   `RLinf-Gr00t-SFT-{Spatial,Object,Goal,10}`. Watch for base-versus-adapter
   mixups (`model_path` is the full base model, `lora_path` the LoRA adapter) and
   casing that drifts from the canonical name. Intentional user-supplied
   placeholders, such as DAgger's `student_model` and `expert_model` or "any pi05
   checkpoint", are not findings; confirm against the recipe page before flagging.
