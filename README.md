
# 🌾 KG-MASFO: Knowledge Graph-Based Multi-Agent System for Fertilizer Optimization

![KG-MASFO Pipeline](https://github.com/naabuzai/KG-MASFO/blob/main/images/MASFO_KG.png)

**KG-MASFO** is an explainable AI system for optimizing fertilizer recommendations using a knowledge graph (KG) constructed from real agricultural data. The system includes GNN-based multi-agent reasoning, attention fusion, symbolic interpretation, and interactive querying.

---

## 🧩 Key Modules

- **Neo4j Knowledge Graph**: Stores soil, crop, and weather information as structured nodes and semantic relationships.
- **GNN Domain Agents**: GATv2 and Transformer-based agents trained separately for each domain.
- **Post-hoc Explainability**: Symbolic and LLM-based reasoning to interpret each prediction.

---

## 🗂️ Project Structure

```
KG-MASFO/
├── Codes/           # GNN models, training pipeline, interactive mode
├── DataFiles/       # Input CSVs (soil, crop, weather)
├── DumpFile/        # Neo4j .dump file or populated CSVs
├── PaperDraft/      # Paper LaTeX draft and figures
├── Presenattion/    # Slides and visual pipeline diagrams
├── README.md        # This file
└── requirements.txt # Python requirements
```

---

## 🚀 Quick Start

### 1. Install Dependencies
```bash
pip install -r requirements.txt
```

### 2. Launch Neo4j and Load KG
- **Option 1**: Open Neo4j Desktop, create a DB, and import from `DumpFile/*.dump`
- **Option 2**: Use `Codes/populate_neo4j.py` to load from CSVs in `DataFiles/`

### 3. Train the Model
```bash
python Codes/train_multi_agent.py
```

### 4. Evaluate and Visualize
```bash
python Codes/evaluate_model.py
```

### 5. Interactive Fertilizer Inference
```bash
python Codes/train_multi_agent.py --interactive
```

---

## 🧠 System Workflow

### 📍 KG Construction
- Input: Structured CSVs (soil, crop, weather)
- Output: Neo4j KG with schema-compliant node types and relations

### 👥 Agent Creation
- SoilAgent, WeatherAgent, CropAgent: GATv2 + residuals
- Fusion: MetaAgent uses attention to weight domain-specific embeddings

### 🧪 Explainability
- Symbolic rule extraction
- Top-k feature interpretation
- Optional LLM (e.g., Llama3) for natural language feedback

---

## 📈 Example Output

```yaml
Predicted (kg/ha):
 - pH: 6.4
 - Nitrogen: 22.7
 - Phosphorous: 18.3
 - Potassium: 31.2
```

**Explanation**:  
Soil→acidic, low NO3 × Weather→humid, cool × Crop→early stage  
**Recommendation**:  
"Apply NPK fertilizer with a higher nitrogen ratio to support early-stage growth under nitrogen stress."

---

## 📊 Experimental Results

### 📋 Model Comparison

![Table I](Presentation/kgmasfo_results_table.png)

> **Table I**: The proposed KG-MASFO outperforms both single-agent and vanilla GNN models on MSE and R² metrics.

| Model Variant     | MSE ↓   | R² Score ↑ |
|-------------------|---------|------------|
| GNN_GAT           | 0.0074  | 0.9025     |
| Single-Agent GAT  | 0.0058  | 0.9515     |
| **KG-MASFO**      | **0.0020** | **0.9739** |

### 📉 Loss Curve

![Fig. 3](https://github.com/naabuzai/KG-MASFO/blob/main/images/loss.png)

> **Fig. 3**: KG-MASFO achieves faster convergence and lowest final loss.

---

## 📡 Data Sources

| Source           | Description                              |
|------------------|------------------------------------------|
| Visual Crossing  | Real-time weather data (temp, humidity)  |
| USDA             | Soil lab properties and field metadata   |
| EDI Portal       | Crop growth, pesticide, and fertilizer   |

---

## 🏗️ Model Architecture

- **Agents**: GATv2 + Residual + TransformerConv
- **Fusion**: Attention mechanism with layer normalization
- **Loss**: Weighted MSE / Huber loss
- **Targets**:
```python
["pH", "Nitrogen", "Phosphorous", "Potassium", "Calcium", "Magnesium", "NH4", "NO3"]
```

---

## ✅ Final Metrics (On Ph as an example) 

| Metric              | Value       |
|---------------------|-------------|
| R² on pH            | 0.81        |
| MAE (8 nutrients)   | 2.47        |
| Explanation coverage| 100%        |
| Inference time      | ~0.2s (CPU) |

---

##  Author Info

**Author**: Nahed Abu Zaid  
**Advisor**: Prof. Ranga R. Vatsavai  
**Affiliation**: Department of Computer Science, NC State University  
**Email**: naabuzai@ncsu.edu

---

## 📜 License

This project is released for academic use. Please cite appropriately if you use it.
"# KG-MASFO" 

