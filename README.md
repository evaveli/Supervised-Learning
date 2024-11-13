# IML Work 2

Second work of UB course "Introduction to Machine Learning" implementing classification with Lazy Learning and SVM

## Names of contributers

Eva Veli, Andras Kasa and Niklas Long Schiefelbein

## Table of Contents

- [Project Setup Instructions](#project-setup-instructions)
  - [Manual Virtual Environment Setup](#1-manual-virtual-environment-setup)
  - [Installing Dependencies](#2-installing-dependencies)
  - [Close Virtual Environment](#3-close-virtual-environment)
- [Project Execution instructions](#project-execution-instructions)
  - [Make sure to be in the root directory `work2`](#1-make-sure-to-be-in-the-root-directory-work2)
  - [Activate the Virtual Environment](#2-activate-the-virtual-environment)
  - [Run `app.py`](#3-run-apppy)
- [Project Structure](#project-structure)

## Prerequisites

- PyCharm IDE (Professional or Community Edition)
- Python 3.9 installed on your system

## Project Setup Instructions

### 1. Manual Virtual Environment Setup

1. Open the project `work2` in PyCharm
2. Open the terminal in PyCharm (View > Tool Windows > Terminal)
3. Optional: Verify current location being `work2` by `pwd`
4. Optional: Navigate to `work2` with `cd`
5. Create a virtual environment:

   ```bash
   # Windows
   py -3.9 -m venv venv

   # macOS/Linux
   python3.9 -m venv venv
   ```

6. Activate the virtual environment:

   ```bash
   # Windows
   venv\Scripts\activate

   # macOS/Linux
   source venv/bin/activate
   ```

In front of the input line in the terminal it should now say `(venv)`

### 2. Installing Dependencies

With the virtual environment activated:

```bash
pip install -r requirements.txt
```

From here you can directly jump to [Run `app.py`](#3-run-apppy)

### 3. Close Virtual Environment

With the virtual environment activated:

```bash
deactivate
```

The `(venv)` in front of the terminal should be gone

## Project Execution instructions

### 1. Make sure to be in the root directory `work2`

For this, just follow the optional steps 3 and 4 from the Manual Virtual Environment Setup

### 2. Activate the Virtual Environment

```bash
# Windows
venv\Scripts\activate

# macOS/Linux
source venv/bin/activate
```

In front of the input line in the terminal it should now say `(venv)`

### 3. Run `app.py`

```bash
python app.py
```

The first execution takes more time than usual due to the initial compilation of the whole project. Once compiled, it prompts the user to provide an input. The user must decide whether to use the `hepatitis` or the `pen-based` dataset for the analysis. By simply pressing enter, the hepatitis dataset will be selected by default.

Now the entire project pipeline will execute, including data preprocessing, KNN and SVM analyses, various reduction techniques, and final report generation. Progress is displayed in the console, but due to frequent calculations and multithreading, following along in real-time may be difficult. It is recommended to refer to the final reports for evaluation. The program completes once the `nemenyi test report` is generated.

For deeper insights please consider reading the report of the project.

## Project Structure

```
work2/
├── classifiers/               # SVM and KNN classifiers
├── csv-results/               # Performance metrics and results
├── datasetsCBR/               # Dataset files
├── metrics/                   # Performance metric calculations
├── preprocessing/             # Data preprocessing scripts
├── reduction_techniques/      # Instance reduction algorithms
├── reporting/                 # Reporting and analysis scripts
├── reports/                   # Generated reports
├── venv/                      # Virtual environment
├── app.py                     # Main application script
├── README.md                  # This file
├── requirements.txt           # Dependencies
└── utils.py                   # Utility functions
```
