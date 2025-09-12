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

Install the required packages, including Sphinx, from the `requirements.txt` file:

```bash
pip install -r requirements.txt
```

---

## Building the Documentation

With your environment ready, build the documentation using Sphinx. Source files are in the `source` directory, and output HTML files go to `build/html`.

Run this command to build the documentation:

```bash
sphinx-build source-en build/html # change to source-zh for Chinese docs
```

### Using `sphinx-autobuild` for Live Reloading

For a smoother development experience, use `sphinx-autobuild` to rebuild the documentation automatically when source files change:

```bash
sphinx-autobuild source-en build/html # change to source-zh for Chinese docs
```

This starts a local server and updates the documentation on file changes.

## To keep it simple  
Just run the following command. It will automatically set the environment variables and run the `sphinx-autobuild` command:  
```bash
bash autobuild.sh
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