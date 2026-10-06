# Enterprise Python Automation & Systems Engineering Portfolio

A production-ready suite of automated systems administration tools, filesystem hygiene utilities, cryptographically verified backup pipelines, and background process surveillance daemons built in Python.

---

## Repository Architecture

```text
Automation Projects/
├── Automated Directory Scanner and Delete Empty Files/
│   ├── DirectoryScannerEmptyDeleteFinal.py
│   └── README.md
│
├── Automated Process Logger & Scheduling System/
│   ├── PlatformSurvillence_Process_logXX.py
│   └── README.md
│
├── Marvellous Data Shield – Automated Backup & File Monitoring System/
│   ├── Data-shield-workflow.png
│   ├── Data_Shield.py
│   └── README.md
│
├── Duplicate File Cleaner & Log Automation/
│   ├── Duplicate_File_Removal_Final.py
│   └── README.md
│
├── Marvellous Platform Surveillance System/
│   ├── PlatformSurveillance_Process_Final.py
│   └── README.md
│
├── System Process Logger with Scheduling/
│   ├── Process_logger.py
│   └── README.md
│
├── requirements.txt
└── README.md
```

---

## Detailed Project Catalog

### 1. Automated Directory Scanner and Delete Empty Files
* **Directory:** `Automated Directory Scanner and Delete Empty Files/`
* **Entry Script:** `DirectoryScannerEmptyDeleteFinal.py`
* **Primary Responsibilities:**
  * Recursively scans target directory trees using `os.walk`.
  * Identifies 0-byte (empty) files and evaluates path safety.
  * Safely removes orphaned empty files and records an audit log file complete with timestamps, file paths, and purge totals.
* **CLI Execution:**
  ```bash
  python "Automated Directory Scanner and Delete Empty Files/DirectoryScannerEmptyDeleteFinal.py" "/path/to/target_directory"
  ```

---

### 2. Automated Process Logger & Scheduling System
* **Directory:** `Automated Process Logger & Scheduling System/`
* **Entry Script:** `PlatformSurvillence_Process_logXX.py`
* **Primary Responsibilities:**
  * Interrogates the operating system process table at regular intervals using `psutil`.
  * Collects Process ID (PID), process name, memory footprint (RSS/VMS), thread count, and CPU consumption metrics.
  * Dispatches scheduled snapshots to structured log directories tagged with UTC/local timestamps.
* **CLI Execution:**
  ```bash
  python "Automated Process Logger & Scheduling System/PlatformSurvillence_Process_logXX.py" 10
  # (Runs recurring surveillance scans every 10 minutes)
  ```

---

### 3. Marvellous Data Shield – Automated Backup & File Monitoring System
* **Directory:** `done Marvellous Data Shield – Automated Backup & File Monitoring System/`
* **Entry Script:** `Data_Shield.py`
* **Asset:** `Data-shield-workflow.png` (Architectural pipeline visual)
* **Primary Responsibilities:**
  * Performs incremental and differential data replication from active workspaces into secure backup destinations.
  * Generates cryptographic checksum manifests (SHA-256 / MD5) to verify file integrity and detect post-backup silent bit rot or tampering.
  * Packages data into timestamped zip/tar archives and generates validation reports.
* **CLI Execution:**
  ```bash
  python "done Marvellous Data Shield   Automated Backup & File Monitoring System/Data_Shield.py" "/source/path" "/backup/destination"
  ```

---

### 4. Duplicate File Cleaner & Log Automation
* **Directory:** `Duplicate File Cleaner & Log Automation/`
* **Entry Script:** `Duplicate_File_Removal_Final.py`
* **Primary Responsibilities:**
  * Detects redundant files across complex directory structures via cryptographic file hashing (MD5/SHA-256) rather than superficial filename matching.
  * Implements chunked binary reading to safely hash large ISOs, videos, and archives without memory exhaustion.
  * Preserves the primary canonical file while eliminating secondary duplicates, writing an audit trail of reclaimed disk space to a timestamped log.
* **CLI Execution:**
  ```bash
  python "Duplicate File Cleaner & Log Automation/Duplicate_File_Removal_Final.py" "/path/to/scan"
  ```

---

### 5. Marvellous Platform Surveillance System
* **Directory:** `Marvellous Platform Surveillance System/`
* **Entry Script:** `PlatformSurveillance_Process_Final.py`
* **Primary Responsibilities:**
  * Background monitoring daemon tracking runaway processes and elevated RAM/CPU utilization.
  * Compiles comprehensive operational health reports into standalone log bundles.
  * Uses `smtplib` and MIME multipart handlers to automatically email process diagnostics and alert logs to system administrators when anomaly conditions are triggered.
* **CLI Execution:**
  ```bash
  python "Marvellous Platform Surveillance System/PlatformSurveillance_Process_Final.py" "admin@example.com"
  ```

---

### 6. System Process Logger with Scheduling
* **Directory:** `System Process Logger with Scheduling/`
* **Entry Script:** `Process_logger.py`
* **Primary Responsibilities:**
  * Lightweight CLI utility for process profiling and resource consumption tracking.
  * Formats process information into CSV/tabular logs for performance analysis and audits.
  * Supports custom scheduling intervals to fit lightweight background cron jobs.
* **CLI Execution:**
  ```bash
  python "System Process Logger with Scheduling/Process_logger.py" --interval 5
  ```

---

## Installation & Setup

### 1. Prerequisites
* Python 3.10+ installed.
* Standard administrative privileges (for inspecting privileged processes and performing directory modifications).

### 2. Environment Configuration
```bash
# Clone the repository
git clone https://github.com/<your-username>/automation-projects.git
cd automation-projects

# Create a virtual environment
python3 -m venv venv

# Activate the virtual environment
# On macOS / Linux:
source venv/bin/activate
# On Windows:
venv\Scripts\activate

# Install required dependencies
pip install --upgrade pip
pip install -r requirements.txt
```

---

## Shared Dependencies (`requirements.txt`)

```text
psutil>=5.9.0
schedule>=1.2.0
requests>=2.31.0
pandas>=2.0.0
```

---

## System Requirements & Permissions

* **Filesystem Operations:** Scripts performing deletion (`DirectoryScannerEmptyDeleteFinal.py`, `Duplicate_File_Removal_Final.py`) require read and write permissions on target folders.
* **Process Inspection:** Reading process statistics for all users requires elevated permissions (run terminal as Administrator on Windows or with appropriate capabilities on Linux).
* **Automated Emailing:** For projects sending automated alerts via SMTP, provide app-specific passwords or export environment variables:
  ```bash
  export SMTP_SENDER="monitor@yourdomain.com"
  export SMTP_PASSWORD="your-secure-app-password"
  ```

---

## License

Distributed under the MIT License. See `LICENSE` for details.
