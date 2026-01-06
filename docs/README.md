# RLinf Documentations

Welcome to the documentation for RLinf! This README provides detailed instructions on how to generate the project documentation locally using Sphinx. It covers the entire process, from setting up your environment to building and viewing the documentation. Additionally, it includes information on cleaning the build directory and an introduction to Sphinx and reStructuredText (RST).

---

## Setting Up Your Environment

### Step 1: Set Environment Variables

Every time you open a new terminal session to work on the documentation, run these commands to set the locale for Sphinx:

```bash
export LC_ALL=C.UTF-8
export LANG=C.UTF-8
```

These ensure proper character encoding with the `C.UTF-8` locale.

**Note**: Repeat this step for every new terminal session before building the documentation.

### Step 2: Install Dependencies

```bash
cd <path_to_RLinf_project_root>  # Navigate to the RLinf project root directory
bash requirements/install.sh docs --venv .docs-venv
source .docs-venv/bin/activate
```

---

## Building the Documentation

With your environment ready, build the documentation using Sphinx. Source files are in the `source` directory, and output HTML files go to `build/html`.

You can simply run the following command to build the English docs and open a server for preview and live reloading:
```bash
bash autobuild.sh
```
To build the Chinese docs, run this:

```bash
bash autobuild.sh zh
```

To build without running the preview server, run this command:
```bash
sphinx-build source-en build/html # change to source-zh for Chinese docs
```

---

## Viewing the Documentation

After building, view the documentation in your browser.

### With `sphinx-build`

1. Go to the `build/html` directory.
2. Open `index.html` in your browser.

Or, serve it with a Python HTTP server:

```bash
cd build/html
python -m http.server 8000
```

Visit `http://localhost:8000` in your browser.

### With `sphinx-autobuild`

Running `sphinx-autobuild` automatically hosts the documentation at `http://localhost:8000`. Open this URL to view it with live reloading.

---

## Cleaning the Build Directory

To remove generated files and start fresh, clean the build directory:

```bash
make clean
```

This deletes the `build` directory and its contents.

---

## Writing reStructuredText (RST)

Sphinx uses reStructuredText (RST), a simple yet powerful markup language for documentation.

[RST grammer](https://zh-sphinx-doc.readthedocs.io/en/latest/rest.html)