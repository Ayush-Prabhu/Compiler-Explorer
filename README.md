# Compiler Explorer

A powerful, offline PyQt5-based IDE designed for C/C++ developers who want to analyse their code.

**Compiler Explorer** mimics functionality found in tools like Godbolt.org but runs locally on your machine, allowing you to visualize Control Flow Graphs (CFGs), analyze GIMPLE/RTL optimization passes, profile code execution, and view assembly mappings without sending code to a remote server.

## 🚀 Key Features

### 🔍 Deep Compiler Analysis
* **Optimization Timeline:** View side-by-side diffs of GCC internal passes (GIMPLE and RTL). See exactly how the compiler optimizes your code (e.g., inlining, dead code elimination, loop unrolling).
* **Control Flow Graphs (CFG):** visualizes the logic flow of functions using Graphviz.
* **Dominator Trees:** Visualize the immediate dominator relationships in your code.
* **Assembly Mapping:** View generated Assembly (`.s`) side-by-side with source code, with highlighting to map high-level code to instructions.

### 🛠️ Development Tools
* **Static Analysis:** Integrated **Cppcheck** support to find bugs before compiling.
* **Binary Analysis:** "Binary Size" tool to visualize symbol sizes (nm) and find bloat.
* **Sanitizers:** One-click compilation with AddressSanitizer (ASan) and UndefinedBehaviorSanitizer (UBSan).
* **AST Viewer:** Inspect the Abstract Syntax Tree (supports GCC and Clang).

### 📊 Visualization & Profiling
* **Header Dependency Graph:** Visualize your `#include` hierarchy to find bloated dependencies.
* **Heatmap Profiling:** Integrated **GCOV** support. Run your code and see a "Heatmap" overlay in the editor (Red=Hot, Green=Cold) to identify performance bottlenecks.

## 📦 Prerequisites

To run all features of the application, you need the following system tools installed:

### 1. Python Libraries
* Python 3.6+
* PyQt5

### 2. System Tools (Linux/Debian/Ubuntu)
The application relies on standard GNU tools.
```bash
sudo apt install gcc g++ graphviz cppcheck clang-format binutils
````

  * **GCC/G++:** The core compiler.
  * **Graphviz (`dot`):** Required to render CFG and Dominator graphs.
  * **Cppcheck:** Required for the "Static Analysis" feature.
  * **Clang-format:** Required for the "Format Code" feature.

## 🛠️ Installation & Usage

  Download the ```compiler-explorer-deb.deb``` file.

  ```bash
  sudo apt install ./compiler-explorer-deb.deb
  ```

## 📦 Building a Standalone Package (.deb)

You can convert this tool into an installable `.deb` package for Debian/Ubuntu systems.

1.  **Install PyInstaller:**

    ```bash
    pip install pyinstaller
    ```

2.  **Build the binary:**

    ```bash
    pyinstaller --onefile --windowed --name=compiler-explorer versioncpp_1.py
    ```

3.  **Package it (Manual Method):**

    ```bash
    # Create structure
    mkdir -p deb-package/usr/local/bin
    mkdir -p deb-package/DEBIAN

    # Copy binary
    cp dist/compiler-explorer deb-package/usr/local/bin/

    # Create control file (Paste the control content provided below)
    nano deb-package/DEBIAN/control 

    # Build
    dpkg-deb --build deb-package
    ```

    *Control file content:*

    ```text
    Package: compiler-explorer
    Version: 1.0.0
    Section: devel
    Priority: optional
    Architecture: amd64
    Depends: libc6, gcc, g++, graphviz, cppcheck
    Maintainer: Ayush Prabhu <ayushprabhu20@gmail.com>
    Description: Local Compiler Explorer IDE
    ```

## ⚠️ Troubleshooting common issues

**1. Graphs are not showing up?**
Ensure `graphviz` is installed. Type `dot -V` in your terminal. If it's not found, install it (`sudo apt install graphviz`).

**2. Profiling (Heatmap) says "Exit Code 1"?**
Profiling requires the code to run successfully. If your code crashes (e.g., Division by Zero), GCOV data may not be flushed. Ensure your code exits normally (`return 0;`).

**3. "Cppcheck not found"?**
Install it via your package manager or ensure it is added to your system PATH.

## 🤝 Contributing

Pull requests are welcome\! For major changes, please open an issue first to discuss what you would like to change.

## Known Issues

- Clang-tidy is not set as a necessary dependency, functionalities have also not been fully tested.
- Clang doesn't support some visualisations, gcc is preferred.



```
```
