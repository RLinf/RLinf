# Posting a GitHub review

Read this file only when the user asks to post, update, or reply to a review on GitHub.

## 1. Build, lint, and post

Write the GitHub create-review API payload as JSON in the scratchpad (`json.dump` for clean escaping), lint it, then post it. `body` is the main comment from `review-pr/SKILL.md` step 8 (without `### Inline comments` or `## Check summary`):

```bash
python3 .agents/skills/review-pr/lint_review.py template      # main-comment skeleton
python3 .agents/skills/review-pr/lint_review.py lint --pr "$N" review.json
python3 .agents/skills/review-pr/lint_review.py post --pr "$N" review.json
python3 .agents/skills/review-pr/lint_review.py update --pr "$N" --review <id> review.json
```

```json
{
  "commit_id": "<PR head SHA>",
  "event": "COMMENT",
  "body": "<the main comment>",
  "comments": [
    {"path": "rlinf/x.py", "line": 42, "body": "<inline comment>"},
    {"path": "rlinf/y.py", "start_line": 10, "line": 12, "body": "<inline comment>"}
  ]
}
```

- `event` is `COMMENT`, or `REQUEST_CHANGES` when asked (`APPROVE` only when explicitly asked).
- `lint_review.py lint` enforces the step 8 structure (verdict, `Result:`/`Setup:` reproduction bullets, findings table, `####` sections only for `here` findings, paired inline comments with matching `**F<n> · <severity> · <category>: <title>**` headers plus `Suggested fix:` and `Evidence:`, plain Markdown without HTML/placeholders/hard-wrapped paragraphs/unclosed fences, 65,536-char GitHub body limit, and with `--pr`, diff-hunk line validity and existing-thread checks) plus warnings for long reproduction bullets, unquoted `Code walkthrough:` blocks, hedging phrases, and >5 minor findings.
- `post` runs the same lint, posts everything as one review, and prints its URL. Report the URL and inline comment count to the user.
- Update a posted review in place (`lint_review.py update` for the body; edit or reply on inline threads). Only a new review round on a new head posts a new review. If `post` fails without printing a URL, check `gh api "repos/RLinf/RLinf/pulls/$N/reviews"` before retrying so you do not double-post.

## 2. Where each finding goes

- **Inline (`comments[]`)**: any finding anchored to a diff line (or block `start_line`..`line`). Use `"side": "LEFT"` only for a deleted line that should stay. For a docs-style rule repeated on many lines of one page, comment on the first line and note the remaining lines rather than opening a thread per line.
- **Main comment (`here`)**: findings with no diff line (missing doc page, missing test, untouched file that should have changed, PR metadata).
- **Existing threads**: reply instead of opening a duplicate thread on the same line:

```bash
gh api "repos/RLinf/RLinf/pulls/$N/comments/<id>/replies" -f body="..."
```

## 3. Writing inline comments

Write for a reader who may not know both RLinf internals and this module. Each inline comment is self-contained (not repeated in the main comment) and has four parts:

1. **Header**: first line `**F<n> · <severity> · <category>: <plain title>**` matching its row in the Findings table.
2. **Problem (1–3 sentences)**: start from what the code or doc is trying to do, state what happens instead and why it matters, then cite identifiers or `main:<path>:<line>` as needed.
3. **`Suggested fix:`**: one sentence stating what to change, followed by a `suggestion` block when replacing the exact commented lines (verify indentation, target lines, and syntax), or a fenced code block otherwise.
4. **`Evidence:`**: one short line with the check/command result or exact trace (for unresolved findings, what settles it).

Keep prose plain and direct: short sentences, active voice, one line per paragraph (GitHub turns newlines into line breaks), no HTML, no hedging, no filler, and no praise. Write in English unless asked otherwise.

### Examples

Correctness:

````markdown
**F1 · critical · correctness: `dones` cast to `int` inverts the loss mask**

`dones` is cast to `int` before the mask is built, so `~dones` gives `-1` and `-2` instead of `False` and `True`. The loss then weights every step with a negative factor and pushes the policy the wrong way.

Suggested fix: keep `dones` boolean when building the mask.

```suggestion
        mask = ~dones.bool()
```

Evidence: `~torch.tensor([0, 1], dtype=torch.int)` returns `tensor([-1, -2])`.
````

Design:

````markdown
**F2 · major · design: model-specific wrist image branch in shared rollout worker**

The shared rollout worker now checks `model_type == "foo"` to add the wrist image, coupling the worker to one model. Model-specific inputs belong in the policy: the worker already passes raw `env_obs` to `predict_action_batch` for every model (`main:rlinf/workers/rollout/hf/huggingface_worker.py:565`).

Suggested fix: build the wrist image inside `FooPolicy.predict_action_batch` and drop the branch from the worker.

```python
class FooPolicy(BasePolicy):
    def predict_action_batch(self, env_obs, **kwargs):
        env_obs = {**env_obs, "wrist_image": self._wrist_view(env_obs)}
        ...
```

Evidence: `huggingface_worker.py:565` passes raw `env_obs` to `BasePolicy.predict_action_batch` (`patterns.md` § Contain component code).
````

Docs style:

````markdown
**F3 · minor · docs-style: rollout paragraph tours internal source paths**

This paragraph explains rollout by listing internal worker, model, and channel paths that a user never opens or edits (`docs/rules/base.md`, "Write about what the reader does, not about the code").

Suggested fix: describe the rollout loop in user-visible terms and keep only the config key the user edits.

```suggestion
During rollout, the policy reads each observation from the simulator, predicts the next action chunk, and sends it back; the simulator steps every environment and returns the next observation. Set `actor.model.num_action_chunks` to control chunk length.
```

Evidence: `docs/code-name` flagged 4 internal symbols on this recipe page.
````
