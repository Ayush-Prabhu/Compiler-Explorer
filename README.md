# Compiler Explorer

A powerful, offline PyQt5-based IDE designed for C/C++ developers who want to analyse their code.

**Compiler Explorer** mimics functionality found in tools like Godbolt.org but runs locally on your machine, allowing you to visualize Control Flow Graphs (CFGs), analyze GIMPLE/RTL optimization passes, profile code execution, and view assembly mappings without sending code to a remote server.

## Key Features

### Deep Compiler Analysis
* **Optimization Timeline:** View side-by-side diffs of GCC internal passes (GIMPLE and RTL). See exactly how the compiler optimizes your code (e.g., inlining, dead code elimination, loop unrolling).
* **Control Flow Graphs (CFG):** visualizes the logic flow of functions using Graphviz.
* **Dominator Trees:** Visualize the immediate dominator relationships in your code.
* **Assembly Mapping:** View generated Assembly (`.s`) side-by-side with source code, with highlighting to map high-level code to instructions.

### Development Tools
* **Static Analysis:** Integrated **Cppcheck** support to find bugs before compiling.
* **Binary Analysis:** "Binary Size" tool to visualize symbol sizes (nm) and find bloat.
* **Sanitizers:** One-click compilation with AddressSanitizer (ASan) and UndefinedBehaviorSanitizer (UBSan).
* **AST Viewer:** Inspect the Abstract Syntax Tree (supports GCC and Clang).

### Visualization & Profiling
* **Header Dependency Graph:** Visualize your `#include` hierarchy to find bloated dependencies.
* **Heatmap Profiling:** Integrated **GCOV** support. Run your code and see a "Heatmap" overlay in the editor (Red=Hot, Green=Cold) to identify performance bottlenecks.

## Prerequisites

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

## Installation & Usage

  Download the [```compiler-explorer-deb.deb```](https://github.com/Ayush-Prabhu/Compiler-Explorer/releases/download/v0.1.0-beta/compiler-explorer-deb.deb) .

  ```bash
  sudo apt install ./compiler-explorer-deb.deb
  ```

## Usage

```bash
compiler-explorer
```
start screen with sample code | split screen with code and static analysis
:-: | :-:
<img width="1918" height="870" alt="image" src="https://github.com/user-attachments/assets/35e9b13e-53bf-4a97-9571-2ad812bfe2eb" /> | <img width="1917" height="866" alt="image" src="https://github.com/user-attachments/assets/269c8dc3-955b-4179-b871-86fc72ac5be0" />


split screen with sanitizer output and generated assembly | split screen with CFG and Header dependency tree
:-: | :-:
<img width="1917" height="868" alt="image" src="https://github.com/user-attachments/assets/26182925-6afc-49d5-ae9f-5e51111ac591" /> | <img width="1919" height="868" alt="image" src="https://github.com/user-attachments/assets/840a9751-1296-4c24-9005-ce34073c1884" />

split screen with GIMPLE optimisation and RTL optimisation | Comparison of binary size when compiled using -O0 vs -Os
:-: | :-:
<img width="1916" height="865" alt="image" src="https://github.com/user-attachments/assets/e27a3a62-7138-4e7e-b228-77f26d5d826d" /> | <img width="1917" height="862" alt="image" src="https://github.com/user-attachments/assets/cbc71a04-2018-4497-acee-b20e0ded4f54" />




## Troubleshooting common issues

**1. Graphs are not showing up?**
Ensure `graphviz` is installed. Type `dot -V` in your terminal. If it's not found, install it (`sudo apt install graphviz`).

**2. Profiling (Heatmap) says "Exit Code 1"?**
Profiling requires the code to run successfully. If your code crashes (e.g., Division by Zero), GCOV data may not be flushed. Ensure your code exits normally (`return 0;`).

**3. "Cppcheck not found"?**
Install it via your package manager or ensure it is added to your system PATH.

## Contributing

Pull requests are welcome\! For major changes, please open an issue first to discuss what you would like to change.

## Known Issues

- Clang-tidy is not set as a necessary dependency, functionalities have also not been fully tested.
- Clang doesn't support some visualisations, gcc is preferred.
- Profiling doesn't work for code with compile-time/runtime errors.

## Upcoming features

- Cross compilation support
- Support for other GNU Compiler Collection compilers

**Developed by** - Ayush Prabhu
