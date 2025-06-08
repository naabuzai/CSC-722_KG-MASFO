import torch 
import torch.nn as nn 
import torch.nn.functional as F 
from torch_geometric.nn import GATv2Conv, GCNConv 
from torch_geometric.data import Data 
from neo4j import GraphDatabase 
import numpy as np 
from sklearn.preprocessing import StandardScaler, RobustScaler
from sklearn.metrics import mean_squared_error, r2_score 
import pandas as pd 
import matplotlib.pyplot as plt 
import random 
import logging 
import os 
from tqdm import tqdm 
 
from langchain_ollama import OllamaLLM 
 
logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s') 
 
SEED = 42 
torch.manual_seed(SEED) 
np.random.seed(SEED) 
random.seed(SEED) 
if torch.cuda.is_available(): 
    torch.cuda.manual_seed(SEED) 
    torch.backends.cudnn.deterministic = True 
 
URI = "bolt://localhost:7687" 
USER = "neo4j" 
PASSWORD = "*******" 
DATABASE = "neo4j" 
 
TARGET_LABELS = ["pH", "Nitrogen", "Phosphorous", "Potassium", "Calcium", "Magnesium", "NH4", "NO3"] 
FEATURE_LABELS = [ 
    "Humidity", "temp", "tempmin", "tempmax",  
    "Expected Yield (tons/ha)", "Growth Percentage (%)", 
    "Water Retention (%)", "Organic Matter (%)",  
    "Cation Exchange Capacity (cmol/kg)", "Soil Moisture (%)" 
] 
 
WEATHER_FEATURES = ["Humidity", "temp", "tempmin", "tempmax"]       
CROP_FEATURES = ["Expected Yield (tons/ha)", "Growth Percentage (%)"]  
SOIL_FEATURES = ["Water Retention (%)", "Organic Matter (%)", "Cation Exchange Capacity (cmol/kg)", "Soil Moisture (%)"] 
 
NODE_TYPES = [ 
    "Soil Type", "Crop", "Weather", "Fertilizer", "PlotNumber",  
    "pH", "% Nitrogen", "Phosphorous (ppm P in soil)", "Potassium (ppm K in soil)",  
    "Calcium (ppm Ca in soil)", "NH4 in soil (mg/kg)", "NO3 in soil (mg/kg)", 
    "ID1", "ID2" 
] 
 
class Neo4jConnector: 
    def __init__(self): 
        try: 
            self.driver = GraphDatabase.driver(URI, auth=(USER, PASSWORD)) 
            logging.info("Successfully connected to Neo4j.") 
        except Exception as e: 
            logging.error(f"Failed to connect to Neo4j: {e}") 
            raise 
 
    def close(self): 
        if hasattr(self, 'driver'): 
            self.driver.close() 
            logging.info("Neo4j driver closed.") 
 
    def execute_query(self, query, params=None): 
        with self.driver.session(database=DATABASE) as session: 
            try: 
                return session.execute_read(lambda tx: list(tx.run(query, params or {}))) 
            except Exception as e: 
                logging.error(f"Query failed: {e}") 
                return [] 
 
    def fetch_nodes(self): 
        query = """ 
        MATCH (n)  
        RETURN elementId(n) AS id, labels(n) AS labels, properties(n) AS props 
        """ 
        result = self.execute_query(query) 
        nodes = [(r["id"], r["labels"], r["props"]) for r in result] 
         
        node_counts = {} 
        for _, labels, _ in nodes: 
            for label in labels: 
                node_counts[label] = node_counts.get(label, 0) + 1 
         
        for node_type, count in sorted(node_counts.items(), key=lambda x: x[1], reverse=True)[:10]: 
            if count > 10: 
                logging.info(f"{node_type} nodes: {count}") 
        return nodes 
 
    def fetch_edges(self): 
        query = """ 
        MATCH (a)-[r]->(b)  
        RETURN elementId(a) AS src, elementId(b) AS tgt, type(r) AS rel_type 
        """ 
        result = self.execute_query(query) 
        return [(r["src"], r["tgt"], r["rel_type"]) for r in result] 
     
    def fetch_soil_properties(self): 
        rel_query = """ 
        MATCH (p:PlotNumber)-[r]->(prop) 
        WHERE any(label IN labels(prop) WHERE  
              label CONTAINS 'pH' OR  
              label CONTAINS 'Nitrogen' OR  
              label CONTAINS 'Phosphorous' OR  
              label CONTAINS 'Potassium' OR 
              label CONTAINS 'Calcium' OR 
              label CONTAINS 'Magnesium' OR 
              label CONTAINS 'NH4' OR 
              label CONTAINS 'NO3') 
        RETURN DISTINCT type(r) as rel_type LIMIT 5 
        """ 
        rel_types = self.execute_query(rel_query) 
        rel_types_list = [r["rel_type"] for r in rel_types] 
         
        if rel_types_list: 
            logging.info(f"Found relationship types: {rel_types_list}") 
        else: 
            logging.warning("No specific relationship types found for soil properties") 
            rel_types_list = ["*"] 
         
        query = """ 
        MATCH (p:PlotNumber)-[r]->(prop) 
        WHERE any(label IN labels(prop) WHERE  
              label CONTAINS 'pH' OR  
              label CONTAINS 'Nitrogen' OR  
              label CONTAINS 'Phosphorous' OR  
              label CONTAINS 'Potassium' OR 
              label CONTAINS 'Calcium' OR 
              label CONTAINS 'Magnesium' OR 
              label CONTAINS 'NH4' OR 
              label CONTAINS 'NO3') 
        RETURN elementId(p) AS plot_id, labels(prop)[0] AS property_type, properties(prop) AS properties 
        """ 
        result = self.execute_query(query) 
        plot_properties = {} 
        for r in result: 
            plot_id = r["plot_id"] 
            prop_type = r["property_type"] 
            props = r["properties"] 
            if plot_id not in plot_properties: 
                plot_properties[plot_id] = {} 
            if props and len(props) > 0: 
                prop_value = next(iter(props.values()), None) 
                if prop_value is not None: 
                    clean_type = prop_type 
                    if "NH4" in prop_type: 
                        clean_type = "NH4" 
                    elif "NO3" in prop_type: 
                        clean_type = "NO3" 
                    elif "Nitrogen" in prop_type: 
                        clean_type = "Nitrogen" 
                    elif "Phosphorous" in prop_type: 
                        clean_type = "Phosphorous" 
                    elif "Potassium" in prop_type: 
                        clean_type = "Potassium"  
                    elif "Calcium" in prop_type: 
                        clean_type = "Calcium" 
                    elif "Magnesium" in prop_type: 
                        clean_type = "Magnesium" 
                    elif "pH" in prop_type: 
                        clean_type = "pH" 
                    try: 
                        plot_properties[plot_id][clean_type] = float(prop_value) 
                    except (ValueError, TypeError): 
                        plot_properties[plot_id][clean_type] = prop_value 
        logging.info(f"Found soil properties for {len(plot_properties)} plots") 
        return plot_properties 
         
    def fetch_graph_with_properties(self): 
        nodes_query = """ 
        MATCH (n) 
        WHERE any(label IN labels(n) WHERE label IN $node_types) 
        RETURN elementId(n) AS id, labels(n) AS labels, properties(n) AS props 
        """ 
        edges_query = """ 
        MATCH (a)-[r]->(b) 
        WHERE any(label IN labels(a) WHERE label IN $node_types) 
          AND any(label IN labels(b) WHERE label IN $node_types) 
        RETURN elementId(a) AS src, elementId(b) AS tgt, type(r) AS rel_type 
        """ 
        nodes = self.execute_query(nodes_query, {"node_types": NODE_TYPES}) 
        edges = self.execute_query(edges_query, {"node_types": NODE_TYPES}) 
        return nodes, edges 
class AgentGAT(nn.Module): 
    def __init__(self, in_dim, hidden_dim, embedding_dim, num_heads=4, dropout=0.2): 
        super().__init__() 
        self.dropout = dropout 
        self.conv1 = GATv2Conv(in_dim, hidden_dim, heads=num_heads, dropout=dropout) 
        self.bn1 = nn.BatchNorm1d(hidden_dim * num_heads) 
        self.conv_mid = GATv2Conv(hidden_dim * num_heads, hidden_dim * num_heads, heads=1, dropout=dropout)
        self.bn_mid = nn.BatchNorm1d(hidden_dim * num_heads)
        self.conv2 = GATv2Conv(hidden_dim * num_heads, embedding_dim, heads=1, dropout=dropout) 
        self.bn2 = nn.BatchNorm1d(embedding_dim) 
        self.layer_norm = nn.LayerNorm(embedding_dim)
         
    def forward(self, x, edge_index): 
        x = self.conv1(x, edge_index) 
        x = self.bn1(x) 
        x = F.leaky_relu(x, negative_slope=0.2)  
        x = F.dropout(x, p=self.dropout, training=self.training) 
        
        residual = x
        x = self.conv_mid(x, edge_index)
        x = self.bn_mid(x)
        x = F.leaky_relu(x, negative_slope=0.2)
        x = F.dropout(x, p=self.dropout, training=self.training)
        x = x + residual  
        
        x = self.conv2(x, edge_index) 
        x = self.bn2(x) 
        x = F.leaky_relu(x, negative_slope=0.2)
        x = F.dropout(x, p=self.dropout, training=self.training) 
        x = self.layer_norm(x)  
        
        return x 
 
class MetaAgent(nn.Module): 
    def __init__(self, total_embedding_dim, hidden_dims, out_dim, dropout=0.2): 
        super().__init__() 
        self.dropout = dropout
        layers = []
        input_dim = total_embedding_dim
        
        for hidden_dim in hidden_dims:
            layers.append(nn.Linear(input_dim, hidden_dim))
            layers.append(nn.LayerNorm(hidden_dim))
            layers.append(nn.LeakyReLU(negative_slope=0.2))
            layers.append(nn.Dropout(dropout))
            input_dim = hidden_dim
            
        self.hidden_layers = nn.Sequential(*layers)
        self.output_layer = nn.Linear(hidden_dims[-1], out_dim)
        
        self.apply(self._init_weights)
         
    def _init_weights(self, module):
        if isinstance(module, nn.Linear):
            nn.init.xavier_uniform_(module.weight)
            if module.bias is not None:
                nn.init.zeros_(module.bias)
                
    def forward(self, x): 
        x = self.hidden_layers(x)
        x = self.output_layer(x)
        return x 
 
def prepare_node_features(nodes, id_map, feature_labels, soil_properties=None): 
    all_features = [] 
    node_to_plot = {} 
    
    available_features = {label: [] for label in feature_labels}
    
    for i, (neo_id, labels, props) in enumerate(nodes): 
        if neo_id not in id_map: 
            continue
            
        for label in feature_labels:
            if label in props:
                try:
                    value = float(props[label])
                    available_features[label].append(value)
                except (ValueError, TypeError):
                    pass
    
    feature_stats = {}
    for label in feature_labels:
        values = available_features[label]
        if values:
            feature_stats[label] = {
                'mean': np.mean(values),
                'std': np.std(values) if len(values) > 1 else 1.0
            }
        else:
            feature_stats[label] = {'mean': 0.0, 'std': 1.0}
    
    for i, (neo_id, labels, props) in enumerate(nodes): 
        if neo_id not in id_map: 
            continue
            
        node_idx = id_map[neo_id] 
        features = [] 
        
        for label in feature_labels: 
            if label in props: 
                try: 
                    value = float(props[label]) 
                except (ValueError, TypeError): 
                    value = feature_stats[label]['mean']
            elif label in labels: 
                value = 1.0 
            else: 
                value = feature_stats[label]['mean']
                
            features.append(value) 
            
        all_features.append(features) 
        if "PlotNumber" in labels: 
            node_to_plot[node_idx] = neo_id 
            
    return np.array(all_features), node_to_plot 
 
def prepare_target_values(node_to_plot, soil_properties, target_labels, num_nodes): 
    targets = np.full((num_nodes, len(target_labels)), np.nan) 
    mask = np.zeros(num_nodes, dtype=bool) 
    
    target_values = {label: [] for label in target_labels}
    
    for plot_id, props in soil_properties.items():
        for label in target_labels:
            if label in props:
                try:
                    value = float(props[label])
                    target_values[label].append(value)
                except (ValueError, TypeError):
                    continue
    
    target_stats = {}
    for label in target_labels:
        values = target_values[label]
        if values:
            target_stats[label] = {
                'median': np.median(values),
                'q1': np.percentile(values, 25) if len(values) >= 4 else np.min(values),
                'q3': np.percentile(values, 75) if len(values) >= 4 else np.max(values)
            }
        else:
            target_stats[label] = {'median': 0.0, 'q1': -1.0, 'q3': 1.0}
    
    for node_idx, plot_id in node_to_plot.items(): 
        if plot_id in soil_properties: 
            props = soil_properties[plot_id] 
            valid_props = 0
            
            for i, label in enumerate(target_labels): 
                if label in props: 
                    try: 
                        value = float(props[label])
                        targets[node_idx, i] = value
                        valid_props += 1
                    except (ValueError, TypeError): 
                        targets[node_idx, i] = target_stats[label]['median']
                else:
                    targets[node_idx, i] = target_stats[label]['median']
                    
            if valid_props >= len(target_labels) // 2: 
                mask[node_idx] = True 
                
    logging.info(f"Found {mask.sum()} nodes with valid target values") 
    
    for i in range(len(target_labels)): 
        col_mask = np.isnan(targets[:, i]) 
        if col_mask.any(): 
            targets[col_mask, i] = target_stats[target_labels[i]]['median']
            
    return targets, mask 

class LlamaExplainability: 
 
    def __init__(self, model_name="llama3"): 
        try:
            self.llm = OllamaLLM(model=model_name) 
            logging.info(f"Successfully initialized LlamaExplainability with {model_name}")
        except Exception as e:
            logging.error(f"Failed to initialize LlamaExplainability: {e}")
            self.llm = None
     
    def explain_prediction(self, features, prediction, feature_names, target_names, true_values=None): 
        """Enhanced explanation with feature and target names."""
        if self.llm is None:
            return "Explainability module not available."
            
        formatted_features = {name: value for name, value in zip(feature_names, features)}
        formatted_predictions = {name: value for name, value in zip(target_names, prediction)}
        
        prompt = ( 
            "You are a precision agriculture expert. Analyze this soil data and recommendation:\n\n" 
            f"Input Measurements:\n{formatted_features}\n\n" 
            f"Model Recommendation (predicted nutrient levels):\n{formatted_predictions}\n" 
        ) 
        
        if true_values is not None: 
            formatted_true = {name: value for name, value in zip(target_names, true_values)}
            prompt += f"\nActual measured values:\n{formatted_true}\n" 
            
        prompt += ( 
            "\nProvide a clear, actionable explanation of this fertilizer recommendation:" 
            "\n1. Which nutrients should be adjusted and why?" 
            "\n2. How do the weather and soil factors influence this recommendation?"
            "\n3. What would you expect to happen to crop yield with these recommendations?"
            "\n\nKeep your response under 200 words and focus on practical insights."
        ) 
        
        try:
            explanation = self.llm.invoke(prompt)
            return explanation
        except Exception as e:
            logging.error(f"Failed to generate explanation: {e}")
            return "Could not generate explanation due to an error."
def main(): 
    conn = Neo4jConnector() 
    try: 
        logging.info("Fetching nodes and edges from Neo4j...") 
        nodes = conn.fetch_nodes() 
        soil_properties = conn.fetch_soil_properties() 
        node_id_map = {neo_id: i for i, (neo_id, _, _) in enumerate(nodes)} 
         
        features, node_to_plot = prepare_node_features(nodes, node_id_map, FEATURE_LABELS, soil_properties) 
        targets, mask = prepare_target_values(node_to_plot, soil_properties, TARGET_LABELS, len(features)) 
        
        if not mask.any(): 
            logging.warning("No valid targets found. Using synthetic data for demonstration.") 
            node_indices = np.random.choice(len(features), size=min(50, len(features)), replace=False) 
            mask = np.zeros(len(features), dtype=bool) 
            mask[node_indices] = True 
            targets = np.random.rand(len(features), len(TARGET_LABELS)) * 10 
        else:
            for i, label in enumerate(TARGET_LABELS):
                target_min = np.min(targets[mask, i])
                target_max = np.max(targets[mask, i])
                target_mean = np.mean(targets[mask, i])
                target_std = np.std(targets[mask, i])
                logging.info(f"Target {label}: min={target_min:.2f}, max={target_max:.2f}, mean={target_mean:.2f}, std={target_std:.2f}")
         
        edges = conn.fetch_edges() 
        edge_list = [] 
        for src, tgt, _ in edges: 
            if src in node_id_map and tgt in node_id_map: 
                edge_list.append((node_id_map[src], node_id_map[tgt])) 
        logging.info(f"Processed {len(features)} nodes with {len(edge_list)} edges") 
        logging.info(f"Nodes with complete target values: {mask.sum()}") 
        
        if len(edge_list) < 100: 
            logging.warning("Very few edges found! Creating fallback k-nearest neighbors graph.") 
            from sklearn.neighbors import NearestNeighbors 
            k = min(10, len(features) - 1) 
            nbrs = NearestNeighbors(n_neighbors=k).fit(features) 
            _, indices = nbrs.kneighbors(features) 
            for i in range(len(features)): 
                for j in indices[i]: 
                    if i != j: 
                        edge_list.append((i, j)) 
                         
        X = torch.tensor(features, dtype=torch.float) 
        
        Y_np = targets[mask]
        feature_scaler = RobustScaler()
        target_scaler = RobustScaler()
        X_normalized_np = feature_scaler.fit_transform(features)
        X_normalized = torch.tensor(X_normalized_np, dtype=torch.float)
        Y_scaled_np = target_scaler.fit_transform(Y_np)
        Y = torch.tensor(Y_scaled_np, dtype=torch.float)
        edge_index = torch.tensor(edge_list, dtype=torch.long).t().contiguous() 
        node_indices = np.where(mask)[0] 
        if len(node_indices) == 0: 
            logging.error("No valid target nodes found. Cannot train model.") 
            return 
        
        idx = torch.randperm(len(node_indices)) 
        train_size = int(0.7 * len(idx)) 
        val_size = int(0.15 * len(idx)) 
        train_idx = node_indices[idx[:train_size]] 
        val_idx = node_indices[idx[train_size:train_size+val_size]] 
        test_idx = node_indices[idx[train_size+val_size:]] 
        logging.info(f"Train: {len(train_idx)}, Val: {len(val_idx)}, Test: {len(test_idx)} nodes") 
        
        train_mask = torch.tensor([i in train_idx for i in range(len(features))], dtype=torch.bool) 
        val_mask = torch.tensor([i in val_idx for i in range(len(features))], dtype=torch.bool) 
        test_mask = torch.tensor([i in test_idx for i in range(len(features))], dtype=torch.bool) 
        X_weather = X_normalized[:, 0:4] 
        X_crop = X_normalized[:, 4:6] 
        X_soil = X_normalized[:, 6:10] 
        hidden_dim = 32  
        emb_dim = 64     
        agent_weather = AgentGAT(in_dim=4, hidden_dim=hidden_dim, embedding_dim=emb_dim, num_heads=4, dropout=0.2) 
        agent_crop = AgentGAT(in_dim=2, hidden_dim=hidden_dim, embedding_dim=emb_dim, num_heads=4, dropout=0.2) 
        agent_soil = AgentGAT(in_dim=4, hidden_dim=hidden_dim, embedding_dim=emb_dim, num_heads=4, dropout=0.2) 
        total_embedding_dim = emb_dim * 3 
        meta_agent = MetaAgent(
            total_embedding_dim=total_embedding_dim, 
            hidden_dims=[256, 128, 64], 
            out_dim=len(TARGET_LABELS),
            dropout=0.2
        )
        optimizer = torch.optim.AdamW( 
            list(agent_weather.parameters()) + 
            list(agent_crop.parameters()) + 
            list(agent_soil.parameters()) + 
            list(meta_agent.parameters()), 
            lr=0.0005,  
            weight_decay=1e-4  
        ) 
        
        scheduler = torch.optim.lr_scheduler.ReduceLROnPlateau( 
            optimizer, mode='min', factor=0.5, patience=15, verbose=True, min_lr=1e-6
        ) 
        
        loss_fn = nn.HuberLoss(delta=1.0)  
         
        best_val_loss = float('inf') 
        best_state = None 
        patience = 40  
        patience_counter = 0 
        train_losses = [] 
        val_losses = [] 
        device = torch.device('cuda' if torch.cuda.is_available() else 'cpu') 
        logging.info(f"Using device: {device}") 
        agent_weather = agent_weather.to(device) 
        agent_crop = agent_crop.to(device) 
        agent_soil = agent_soil.to(device) 
        meta_agent = meta_agent.to(device) 
        
        X_weather = X_weather.to(device) 
        X_crop = X_crop.to(device) 
        X_soil = X_soil.to(device) 
        edge_index = edge_index.to(device) 
        
        Y_train = torch.zeros((len(features), len(TARGET_LABELS)), dtype=torch.float).to(device) 
        Y_train[mask] = Y.to(device) 
         
        logging.info("Starting multi-agent training with improved implementation...") 
        for epoch in range(500):  
            agent_weather.train()
            agent_crop.train()
            agent_soil.train()
            meta_agent.train() 
            
            optimizer.zero_grad() 
            
            emb_weather = agent_weather(X_weather, edge_index) 
            emb_crop = agent_crop(X_crop, edge_index) 
            emb_soil = agent_soil(X_soil, edge_index) 
            fused_embedding = torch.cat([emb_weather, emb_crop, emb_soil], dim=1) 
            pred = meta_agent(fused_embedding) 
             
            loss = loss_fn(pred[train_mask], Y_train[train_mask]) 
            
            l2_reg = 0
            for name, param in meta_agent.named_parameters():
                if 'weight' in name:
                    l2_reg += torch.norm(param, p=2)
            
            loss += 5e-5 * l2_reg  
             
            if torch.isnan(loss): 
                logging.warning(f"NaN loss detected at epoch {epoch}, skipping update")
                continue
             
            loss.backward() 
            
            torch.nn.utils.clip_grad_norm_(
                list(agent_weather.parameters()) + 
                list(agent_crop.parameters()) + 
                list(agent_soil.parameters()) + 
                list(meta_agent.parameters()), 
                max_norm=0.5 
            )
            
            optimizer.step() 
            train_losses.append(loss.item()) 
            
            agent_weather.eval(); agent_crop.eval(); agent_soil.eval(); meta_agent.eval()
            with torch.no_grad():
                emb_weather_val = agent_weather(X_weather, edge_index)
                emb_crop_val = agent_crop(X_crop, edge_index)
                emb_soil_val = agent_soil(X_soil, edge_index)
                fused_embedding_val = torch.cat([emb_weather_val, emb_crop_val, emb_soil_val], dim=1)
                pred_val = meta_agent(fused_embedding_val)
                val_loss = loss_fn(pred_val[val_mask], Y_train[val_mask])
                val_losses.append(val_loss.item())
            
            if not torch.isnan(val_loss) and val_loss < best_val_loss:
                best_val_loss = val_loss.item()
                best_state = {
                    'agent_weather': agent_weather.state_dict(),
                    'agent_crop': agent_crop.state_dict(),
                    'agent_soil': agent_soil.state_dict(),
                    'meta_agent': meta_agent.state_dict()
                }
                patience_counter = 0
            else:
                patience_counter += 1
            
            if not torch.isnan(val_loss):
                scheduler.step(val_loss)
            
            if patience_counter >= patience:
                logging.info(f"Early stopping at epoch {epoch}")
                break
            
            if epoch % 10 == 0:
                logging.info(f"📉 Epoch {epoch} | Train Loss: {loss.item():.5f} | Val Loss: {val_loss.item():.5f}")
        
        if best_state is None:
            logging.error("Training did not improve validation loss. Using latest state.")
        else:
            agent_weather.load_state_dict(best_state['agent_weather'])
            agent_crop.load_state_dict(best_state['agent_crop'])
            agent_soil.load_state_dict(best_state['agent_soil'])
            meta_agent.load_state_dict(best_state['meta_agent'])
        
        agent_weather.eval(); agent_crop.eval(); agent_soil.eval(); meta_agent.eval()
        with torch.no_grad():
            emb_weather_test = agent_weather(X_weather, edge_index)
            emb_crop_test = agent_crop(X_crop, edge_index)
            emb_soil_test = agent_soil(X_soil, edge_index)
            fused_embedding_test = torch.cat([emb_weather_test, emb_crop_test, emb_soil_test], dim=1)
            pred_test = meta_agent(fused_embedding_test)
            
            if test_mask.sum() > 0:
                test_pred = pred_test[test_mask].cpu().numpy()
                test_true = Y_train[test_mask].cpu().numpy()
                mse = mean_squared_error(test_true, test_pred)
                r2 = r2_score(test_true, test_pred)
                logging.info(f" Test MSE: {mse:.5f}, R²: {r2:.5f}")
                
                test_indices = torch.where(test_mask)[0].tolist()
                if test_indices:
                    df_pred = pd.DataFrame(test_pred, columns=[f"Pred_{c}" for c in TARGET_LABELS])
                    df_true = pd.DataFrame(test_true, columns=[f"True_{c}" for c in TARGET_LABELS])
                    
                    additional_features = []
                    for idx in test_indices:
                        feat_dict = {}
                        for i, label in enumerate(FEATURE_LABELS):
                            feat_dict[label] = X[idx, i].item()
                        additional_features.append(feat_dict)
                    
                    df_features = pd.DataFrame(additional_features)
                    result_df = pd.concat([df_true, df_pred, df_features], axis=1)
                    result_df.to_csv("fertilizer_predictions_multi_agent.csv", index=False)
                    
                    plt.figure(figsize=(10, 5))
                    plt.plot(train_losses, label='Training Loss')
                    plt.plot(val_losses, label='Validation Loss')
                    plt.xlabel('Epoch')
                    plt.ylabel('Loss')
                    plt.title('Training and Validation Loss (Multi-Agent)')
                    plt.legend()
                    plt.savefig('learning_curves_multi_agent.png')
                    plt.close()
                    explainer = LlamaExplainability(model_name="llama3")
                    sample_idx = test_indices[0]
                    sample_features = X[sample_idx].tolist()
                    sample_prediction = test_pred[0].tolist()
                    sample_truth = test_true[0].tolist()
                    explanation = explainer.explain_prediction(sample_features, sample_prediction, feature_names=FEATURE_LABELS, target_names=TARGET_LABELS, true_values=sample_truth)
                    logging.info("LLAMA Explanation for first test sample:")
                    logging.info(explanation)
            else:
                logging.warning("No test samples available for evaluation.")
                
    except Exception as e:
        logging.error(f"An error occurred: {str(e)}")
        import traceback
        logging.error(traceback.format_exc())
    finally:
        conn.close()

if __name__ == "__main__":
    main()
