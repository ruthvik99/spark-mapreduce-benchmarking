# PySpark vs Hadoop MapReduce Benchmarking

A benchmarking study comparing the performance of **Vanilla PySpark**, **Optimized PySpark**, and **Hadoop MapReduce** across multiple datasets and machine learning algorithms.

---

## 1. Datasets Used

| Dataset | Size | Type | Description |
|--------|------|------|-------------|
| **Iris** | 150 rows | Numeric | Classic multi-class classification dataset. |
| **SUSY** | 5,000,000 rows | Numeric | Particle physics classification dataset from UCI ML. |
| **Amazon Reviews** | ~12,000,000 rows | Text | Large Amazon review dataset (TSV). Used for binary sentiment classification. |

---

## 2. Algorithms Used

### PySpark Models
- Logistic Regression  
- Decision Tree Classifier  

Two execution variants were benchmarked:
- **Vanilla PySpark**
- **Optimized PySpark**  
  (Kryo serializer, caching, explicit schema, partition tuning)

### Hadoop MapReduce Models
Implemented using Hadoop Streaming:
- Logistic Regression (custom Python mapper + reducer)
- Decision Tree (custom Python mapper + reducer)

---

## 3. How to Run

### A) Run PySpark Benchmarks (Notebooks)

The PySpark benchmarks are provided as Jupyter Notebooks:

```
PySpark_benchmark/
│
├── 01_iris_classification_vanilla.ipynb
├── 02_susy_classification_vanilla.ipynb
├── 03_amazon_classification_vanilla.ipynb
├── 04_iris_classification_optimized.ipynb
├── 05_susy_classification_optimized.ipynb
└── 06_amazon_classification_optimized.ipynb
```

To run:

1. Open VS Code or Jupyter Lab  
2. Select your virtual environment  
3. Run each notebook 
4. Each notebook outputs:  
   - Accuracy  
   - F1 Score  
   - Training time  
   - End-to-end execution time  
   - Throughput  

---

### B) Run MapReduce Benchmarks (Docker Hadoop Cluster)

Start Hadoop cluster:

```bash
docker-compose up -d
```

Run jobs:

```bash
bash run_iris.sh
bash run_susy.sh
bash run_amazon.sh
```

Each script:
- Uploads data to HDFS  
- Runs mapper & reducer  
- Computes true end-to-end job time  
- Outputs accuracy, F1, throughput  

---

## 4. Requirements

### Python
- Python **3.10 or 3.11**  
- PySpark 3.5+
- Required libraries:
```bash
pip install pyspark numpy pandas matplotlib seaborn
```

### Hadoop (Dockerized)
- Docker installed  
- Hadoop 3.2.1 cluster via docker-compose  
- Hadoop Streaming JAR (auto-detected)

### Dataset Files
- `iris.csv`
- `SUSY.csv`
- `amazon_data/*.tsv`

---

## 5. Test Bed (Hardware / Environment)

Benchmarks were executed on:

### Machine
- **MacBook Pro (M3 Pro)**  
- **18 GB Unified Memory**  
- **SSD storage**

### Software
- macOS  
- Local Python virtual environment  
- PySpark in local[*] mode  
- Hadoop cluster running in Docker (amd64 emulated through QEMU)

---

## 6. Docker & Hadoop Cluster Details

The project includes a full Dockerized Hadoop 3.2.1 cluster with:

| Service | Role | Ports |
|---------|------|-------|
| NameNode | HDFS Metadata Manager | 9870 (UI), 9000 (FS) |
| DataNode | HDFS Block Storage | — |
| ResourceManager | YARN Scheduler | 8088 |
| NodeManager | YARN Executor | — |

### Apple Silicon Compatibility
The cluster uses:
```
platform: linux/amd64
```
because Hadoop official images are x86_64 only.  
Docker automatically emulates amd64 via QEMU on Apple Silicon.

### JVM Stability Settings
To avoid JVM crashes in emulation, these flags are used:

```
-XX:-UseCompressedOops 
-XX:-UseCompressedClassPointers 
-Xint
-XX:+UseSerialGC
```

### Python Support in Hadoop Containers
The Dockerfile installs:

- Python3  
- numpy  
- pandas  
- scikit-learn  

This enables Python-based Hadoop Streaming jobs.

---

## 7. Project Structure

```
SPARK-BENCHMARKING/
│
├── datasets/
│   ├── iris.csv
│   ├── SUSY.csv
│   └── amazon_data/
│
├── hadoop_benchmark/
│   ├── docker-compose.yml
│   ├── Dockerfile
│   ├── hadoop.env
│   ├── mapper_iris.py
│   ├── mapper_susy.py
│   ├── mapper_amazon.py
│   ├── reducer.py
│   ├── run_iris.sh
│   ├── run_susy.sh
│   ├── run_amazon.sh
│   └── amazon_data/
│
├── PySpark_benchmark/
│   ├── 01_iris_classification_vanilla.ipynb
│   ├── 02_susy_classification_vanilla.ipynb
│   ├── 03_amazon_classification_vanilla.ipynb
│   ├── 04_iris_classification_optimized.ipynb
│   ├── 05_susy_classification_optimized.ipynb
│   ├── 06_amazon_classification_optimized.ipynb
│
├── spark-env/          # Local Python virtual environment
│
├── Evaluation_Metrics.ipynb
│
└── README.md
```

---

