<!-- omit in toc -->
# Contributing to RLinf

Thanks for taking the time to contribute to RLinf! ❤️

All types of contributions are encouraged and valued. See the [Table of Contents](#table-of-contents) for different ways to help and details about how this project handles them. Please make sure to read the relevant section before making your contribution. It will make it a lot easier for us maintainers and smooth out the experience for all involved. The community looks forward to your contributions. 🎉

> And if you like the project, but just don't have time to contribute, that's fine. There are other easy ways to support the project and show your appreciation, which we would also be very happy about:
> - Star the project
> - Tweet about it
> - Refer this project in your project's readme
> - Mention the project at local meetups and tell your friends/colleagues

<!-- omit in toc -->
## Table of Contents

- [I Want To Contribute](#i-want-to-contribute)
- [I Have a Question](#i-have-a-question)
  - [Reporting Bugs](#reporting-bugs)
  - [Suggesting Enhancements](#suggesting-enhancements)


## I Want To Contribute
All contributions (including the project team's contribution) takes the form of [GitHub Pull Requests](https://github.com/RLinf/RLinf/pulls).
To contribute, first you need to [fork the repository](https://github.com/RLinf/RLinf/fork) and clone it to your local machine.
Then, create a new development branch from `main` for your contribution:
```bash
git checkout main
git pull origin main
git checkout -b feature/your-feature-name
```
After you have made your changes, commit them with a clear and descriptive commit message. The `-s` flag is necessary, which adds a "Signed-off-by" line at the end of the commit message:
```bash
git add .
git commit -m "feat(embodied): add a clear and descriptive commit message" -s
```
Note that we use [Conventional Commits](https://www.conventionalcommits.org/en/v1.0.0/) to structure commit messages, which looks like this:
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

Finally, before pushing your changes to your fork, please run the pre-commit checks to ensure that your code adheres to the project's coding standards:
```bash
pip install pre-commit
pre-commit install --hook-type commit-msg
pre-commit run --all-files
```
If any issues are found, please fix them and re-run the checks until they pass.
Particularly, if your commit message fails the check, you can amend it with: `git commit --amend -s`.

Once all checks pass, push your changes to your fork:
```bash
git push origin feature/your-feature-name
```
Then, open a [Pull Request](https://github.com/RLinf/RLinf/compare) against the `main` branch of the original repository. 
We will review your changes and run CI tests before merging them.

## I Have a Question

> If you want to ask a question, we assume that you have read the available [Documentation](https://rlinf.readthedocs.io/en/latest/).

Before you ask a question, it is best to search for existing [Issues](https://github.com/RLinf/RLinf/issues) that might help you. In case you have found a suitable issue and still need clarification, you can write your question in this issue. It is also advisable to search the internet for answers first.

If you then still feel the need to ask a question and need clarification, we recommend the following:

- Open an [Issue](https://github.com/RLinf/RLinf/issues/new).
- Provide as much context as you can about what you're running into.
- Provide project and platform versions (nodejs, npm, etc), depending on what seems relevant.

We will then take care of the issue as soon as possible.

### Reporting Bugs

<!-- omit in toc -->
#### Before Submitting a Bug Report

A good bug report shouldn't leave others needing to chase you up for more information. Therefore, we ask you to investigate carefully, collect information and describe the issue in detail in your report. Please complete the following steps in advance to help us fix any potential bug as fast as possible.

- Make sure that you are using the latest version.
- Determine if your bug is really a bug and not an error on your side e.g. using incompatible environment components/versions (Make sure that you have read the [documentation](https://rlinf.readthedocs.io/en/latest/). If you are looking for support, you might want to check [this section](#i-have-a-question)).
- To see if other users have experienced (and potentially already solved) the same issue you are having, check if there is not already a bug report existing for your bug or error in the [bug tracker](https://github.com/RLinf/RLinf/issues?q=label%3Abug).
- Also make sure to search the internet (including Stack Overflow) to see if users outside of the GitHub community have discussed the issue.
- Collect information about the bug:
  - Stack trace (Traceback)
  - OS, Platform and Version (Windows, Linux, macOS, x86, ARM)
  - Version of the interpreter, compiler, SDK, runtime environment, package manager, depending on what seems relevant.
  - Possibly your input and the output
  - Can you reliably reproduce the issue? And can you also reproduce it with older versions?

<!-- omit in toc -->
#### How Do I Submit a Good Bug Report?

We use GitHub issues to track bugs and errors. If you run into an issue with the project:

- Open an [Issue](https://github.com/RLinf/RLinf/issues/new).
- Explain the behavior you would expect and the actual behavior.
- Please provide as much context as possible and describe the *reproduction steps* that someone else can follow to recreate the issue on their own. This usually includes your code. For good bug reports you should isolate the problem and create a reduced test case.
- Provide the information you collected in the previous section.

Once it's filed, a team member will try to reproduce the issue with your provided steps. If there are no reproduction steps or no obvious way to reproduce the issue, the team will ask you for those steps and mark the issue as `needs-repro`. Bugs with the `needs-repro` tag will not be addressed until they are reproduced.


### Suggesting Enhancements

This section guides you through submitting an enhancement suggestion for RLinf, **including completely new features and minor improvements to existing functionality**. Following these guidelines will help maintainers and the community to understand your suggestion and find related suggestions.

<!-- omit in toc -->
#### Before Submitting an Enhancement

- Make sure that you are using the latest version.
- Read the [documentation](https://rlinf.readthedocs.io/en/latest/) carefully and find out if the functionality is already covered, maybe by an individual configuration.
- Perform a [search](https://github.com/RLinf/RLinf/issues) to see if the enhancement has already been suggested. If it has, add a comment to the existing issue instead of opening a new one.
- Find out whether your idea fits with the scope and aims of the project. It's up to you to make a strong case to convince the project's developers of the merits of this feature. Keep in mind that we want features that will be useful to the majority of our users and not just a small subset. If you're just targeting a minority of users, consider writing an add-on/plugin library.

<!-- omit in toc -->
#### How Do I Submit a Good Enhancement Suggestion?

Enhancement suggestions are tracked as [GitHub issues](https://github.com/RLinf/RLinf/issues).

- Use a **clear and descriptive title** for the issue to identify the suggestion.
- Provide a **step-by-step description of the suggested enhancement** in as many details as possible.
- **Describe the current behavior** and **explain which behavior you expected to see instead** and why. At this point you can also tell which alternatives do not work for you.
- You may want to **include screenshots or plots** which help you demonstrate the steps or point out the part which the suggestion is related to.
- **Explain why this enhancement would be useful** to most RLinf users. You may also want to point out the other projects that solved it better and which could serve as inspiration.