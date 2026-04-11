<!-- omit in toc -->
# Contributing to RLinf

Thanks for taking the time to contribute to RLinf! ❤️

All types of contributions are encouraged and valued. Please make sure to read the relevant section before making your contribution. It will make it a lot easier for us maintainers and smooth out the experience for all involved. The community looks forward to your contributions. 🎉

> And if you like the project, but just don't have time to contribute, that's fine. There are other easy ways to support the project and show your appreciation, which we would also be very happy about:
> - Star the project
> - Tweet about it
> - Refer this project in your project's readme
> - Mention the project at local meetups and tell your friends/colleagues

<!-- omit in toc -->
## Table of Contents

- [Contribution Procedure](#contribution-procedure)
- [Pull Request Guidelines](#pull-request-guidelines)
  - [Code Style and Formatting](#code-style-and-formatting)
  - [Commit Messages and Signed-off-by](#commit-messages-and-signed-off-by)
  - [PR Title and Description](#pr-title-and-description)
  - [Review Process](#review-process)



## Contribution Procedure

All contributions (including the project team's contribution) takes the form of [GitHub Pull Requests](https://github.com/RLinf/RLinf/pulls).
To contribute, first you need to [fork the repository](https://github.com/RLinf/RLinf/fork) and clone it to your local machine.
Then, create a new development branch from `main` for your contribution:
```bash
git checkout main
git pull origin main
git checkout -b feature/your-feature-name
```

Then, make sure you read and follow the [Pull Request Guidelines](#pull-request-guidelines) below before committing and pushing your changes.

If you have done that, push your changes to your forked repository:

```bash
git push origin feature/your-feature-name
```

Then, open a [Pull Request](https://github.com/RLinf/RLinf/compare) against the `main` branch of the original repository. 
We will review your changes and run CI tests before merging them.

## Pull Request Guidelines

Here documents the general guidelines that all contributors should follow to ensure the quality and consistency of the project.

### THE PRIME DIRECTIVE

**All user-facing changes must be accompanied by tests and most importantly, documentation, which must be followed and validated by at least one reviewer to ensure its reproducibility.**

### Code Style and Formatting

* We adhere to the [Google Python Style Guide](https://google.github.io/styleguide/pyguide.html). It's highly recommended to familiarize yourself with the last section of the guide:

  > BE CONSISTENT.
  >
  > If you’re editing code, take a few minutes to look at the code around you and determine its style. If they use _idx suffixes in index variable names, you should too. If their comments have little boxes of hash marks around them, make your comments have little boxes of hash marks around them too.

* **Lint**: The code should pass linter checks. You can run them locally using `pre-commit`.
  ```bash
  pip install pre-commit
  pre-commit install --hook-type commit-msg
  pre-commit run --all-files
  ```

  If any issues are found, the `pre-commit` tool will try fix them if possible. Otherwise, it will provide instructions on how to fix them.
  If your commit message fails the check, you can amend it with: `git commit --amend -s` after fixing the issues.

* **Comments & Docstrings**: All code should include sufficient comments and docstrings to ensure future contributors can easily understand the code. In particular, all public classes and methods must have docstrings that follow the [Google Python Style Guide for Docstrings](https://google.github.io/styleguide/pyguide.html#38-comments-and-docstrings).

* **Type Hints**: All functions and methods should include type hints for their parameters to improve code readability and facilitate static analysis. If the return type cannot be statically deduced by static analysis tools like Pylance, it should also be included.

* **Error Handling**: All assertions and exceptions should be accompanied with a clear and meaningful error messages. Empty messages and messages that reiterate the assertion itself like `xxx != yyy` is unacceptable. Add assertion to check for invalid inputs and states as early as possible, e.g., before performing division or array indexing.

* **Logging**: Use logging instead of print statements for logging information, warnings, and errors. When you are in `Worker`, use `self.log_info`, `self.log_warning`, and `self.log_error` methods for logging. Outside of `Worker`, you can use the following pattern:
  ```python
  from rlinf.utils.logging import get_logger

  logger = get_logger()
  logger.info("This is an info message.")
  logger.warning("This is a warning message.")
  logger.error("This is an error message.")
  ```

* **Configuration YAML**: If your contribution involves changes to configuration YAML files, please ensure that:
  - Copy existing configuration files as templates for new configurations instead of creating them from scratch if possible. Make sure you are copying from the latest version of the configuration files in the `main` branch.
  - DO NOT perform any calculation or set dynamic values in the YAML files. All values should be static. If you need to compute a value, do it in the code (likely in `config.py`) instead of the YAML file.
  - DO NOT modify configuration fields in code that can be set by users in any circumstances. All fields should be treated as read-only.
  - Avoid referencing other fields as much as possible. Assign values in code if necessary.

* **Tests**: Include CI tests for all new features. You can refer to existing tests in the `tests/` directory for examples. If your tests require new docker images, models, datasets, or other large dependencies, please ping the maintainers in your pull request for assistance.

### Commit Messages and Signed-off-by

All Commits must include a `Signed-off-by:` line at the end of the commit message.
Using the `-s` flag will automatically achieve this:
```bash
git add .
git commit -s
```
You can enable automatic sign-off in your IDE. In VSCode, you can open the [settings editor](https://code.visualstudio.com/docs/configure/settings) and enable the option `Git: Always Sign Off`.

The commit message should follow the [Conventional Commits](https://www.conventionalcommits.org/en/v1.0.0/) standard, which looks like this:
```
<type>(<scope>): <description>
```
Where `<type>` commonly includes the following (others can be found in the [Conventional Commits documentation](https://www.conventionalcommits.org/en/v1.0.0/)):
- `feat`: a new feature for the user
- `fix`: a bug fix for the user
- `docs`: changes to the documentation
- `style`: formatting, missing semi colons, etc; no code change
- `refactor`: refactoring production code, e.g. renaming a variable
- `test`: adding missing tests, refactoring tests; no production code change
- `chore`: updating build tasks, package manager configs, etc; no production code change.

### PR Title and Description

All PR titles should follow the same format as commit messages, i.e.:
```
<type>(<scope>): <description>
```
The PR description should fill in at least the `Description` and `Checklist` sections of the provided PR template, otherwise it will be marked as draft and not reviewed until completed.

If your PR addresses an existing issue, please link to the issue in the `Motivation and Context` section of the PR description.

If your PR has potential impact on training performance and stability (e.g., breaking the RL reward curve), please provide the testing results in the `How has this been tested?` section of the PR description.

### Review Process

* After you have submitted your PR, it will be assigned to at least two maintainers, and possibly other code owners depending on the scope of the changes.

* After the reviewers are assigned, the reviewers will provide feedback every 1-2 business days. If the PR is not reviewed within 3 business days, please feel free to ping the maintainers in the PR thread.

* After the review, the reviewer will put an `action-required` label on the PR if there are changes required. The contributor should address the comments and ping the reviewer to re-review the PR.

* Please respond to all comments within a reasonable time frame. If a comment isn't clear or you disagree with a suggestion, feel free to ask for clarification or discuss the suggestion.

* If you cannot respond to any comment within 7 days, the PR will be considered inactive and will be closed. You can always reopen the PR later when you are ready to address the comments.

* Note that not all CI checks will be executed immediately due to limited computational resources. The reviewer will add `run-ci` label to the PR when the PR is ready to merge or a full CI run is needed. For PRs from existing RLinf team members, please make sure to remove the `run-ci` label if you only wish to make minor changes that do not require a full CI run (e.g., fixing typos in documentation).