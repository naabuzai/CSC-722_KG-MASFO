
# 🌿 KG-F: A Knowledge Graph Framework for Fertilizer Recommendation

<p align="center">
  <img src="https://github.com/naabuzai/KG-F/blob/main/images/KGF_Architecture.png?raw=true" width="700"/>
</p>

KG-F is an explainable AI system for precision agriculture that recommends optimal fertilizer quantities by reasoning over a domain-specific knowledge graph (KG). It combines Graph Neural Networks (GATv2), multi-agent learning, and symbolic explainability to support dynamic, site-specific recommendations.

---

## 🔍 Core Features

- Domain-Enriched KG: Integrates soil, crop, fertilizer, and weather data into a unified Neo4j-based graph.
- Multi-Agent Learning: Specialized GATv2 agents per domain (soil, crop, weather) with attention-based fusion.
- Post-Hoc Explainability: Gradient-based attribution and symbolic rule mapping, enhanced via LLM (Llama 3).
- Real-Time Reasoning: Supports interactive predictions based on updated field conditions.

---

## 📁 Project Structure

```
KG-F/
├── Codes/           # GNN models, agent training, explainability
├── DataFiles/       # Tabular inputs (soil, crop, weather)
├── DumpFile/        # Neo4j database or CSVs
├── PaperDraft/      # LaTeX paper and figures
├── Presentation/    # Slides and architecture visuals
└── README.md        # This file
```

---

## ⚡ Quick Start

1. Install dependencies
```bash
pip install -r requirements.txt
```

2. Start Neo4j and load KG
- Option A: Load Neo4j `.dump` from `DumpFile/`
- Option B: Run `populate_neo4j.py` on `DataFiles/`

3. Train Multi-Agent Model
```bash
python Codes/train_multi_agent.py
```

4. Run Interactive Prediction
```bash
python Codes/train_multi_agent.py --interactive
```

---

## 🧠 System Overview

- Agents: GATv2 agents per domain, with residual layers
- Fusion: Meta-agent combines embeddings via attention MLP
- Targets: ["pH", "Nitrogen", "Phosphorous", "Potassium", "Calcium", "Magnesium", "NH4", "NO3"]
- Explanation: Natural language output from symbolic + LLM-based module

---

## 📉 Experimental Results

### Model Comparison

| Model         | Accuracy | R²     | Final Loss |
|---------------|----------|--------|------------|
| GNN_GAT       | 0.7285   | 0.7865 | 0.15       |
| Single-Agent  | 0.8476   | 0.8700 | 0.34       |
| KG-F (ours)   | 0.8991   | 0.9685 | 0.002      |

### Training Loss

<p align="center">
  <img src="https://github.com/naabuzai/KG-F/blob/main/images/loss.png?raw=true" width="500"/>
</p>

The Multi-Agent system achieves 99.4% lower loss compared to Single-Agent and 98.7% lower than GNN_GAT.

---

## 🗃️ Data Sources

| Source | Description |
|--------|-------------|
| Visual Crossing | Real-time weather metrics |
| USDA Soil Survey | Field-level soil composition |
| EDI Portal | Fertilizer treatment and nitrate levels |
| NASS Crop Data | Crop stage and development rates |

---

## 📣 Example Output

```yaml
Prediction (kg/ha):
  pH: 6.4
  Nitrogen: 22.7
  Phosphorous: 18.3
  Potassium: 31.2
```

Explanation:  
Soil: slightly acidic, Crop: early stage, Weather: humid  
Recommendation: Increase nitrogen ratio to improve early-stage growth.

---

## 🔬 Citation

If you use KG-F in your work, please cite our paper:

```
Nahed Abu Zaid, Jaya Shruti Chintalapati, Rada Chirkova, Ranga Vatsavai, Mihir Shah.  
KG-F: A Knowledge Graph Framework for Fertilizer Recommendation.  
North Carolina State University, 2025.
```

---

## 👤 Contributors

- Nahed Abu Zaid (naabuzai@ncsu.edu)  
- Advisor: Prof. Ranga R. Vatsavai  
- Collaborators: Shruti Chintalapati, Rada Chirkova, Mihir Shah

---

## 📜 License

KG-F is open for academic and research use. For commercial use or contributions, please contact the maintainers.
