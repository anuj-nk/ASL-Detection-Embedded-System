## Setup

### 1. Create and activate the environment

**Windows (PowerShell / CMD):**
```powershell
conda create -n asl-train python=3.11
conda activate asl-train
```

**macOS / Linux:**
```bash
conda create -n asl-train python=3.11
conda activate asl-train
```

> 💡 If `conda activate` fails on Windows, try running it from **Anaconda Prompt** instead.

### 2. Install dependencies

```bash
pip install -r requirements.txt
```

### 3. Run webcam inference

```bash
python webcam_inference.py
```

### 4. ⭐ Run word writing web interface

```bash
python word_writer.py
```