
import torch
import torch.nn as nn
import torch.nn.functional as F
from torch_geometric.nn import GATv2Conv
import numpy as np
from sklearn.preprocessing import RobustScaler
from sklearn.metrics import mean_squared_error, r2_score
import logging
import pandas as pd
import matplotlib.pyplot as plt

# Dummy data loading functions and variables for brevity
# Replace these with actual logic from nahed.py
from nahed import (
    Neo4jConnector, prepare_node_features, prepare_target_values,
    FEATURE_LABELS, TARGET_LABELS, NODE_TYPES, AgentGAT, MetaAgent
)

def main():
    logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')
    conn = Neo4jConnector()
    try:
        nodes = conn.fetch_nodes()
        soil_properties = conn.fetch_soil_properties()
        node_id_map = {neo_id: i for i, (neo_id, _, _) in enumerate(nodes)}
        features, node_to_plot = prepare_node_features(nodes, node_id_map, FEATURE_LABELS, soil_properties)
        targets, mask = prepare_target_values(node_to_plot, soil_properties, TARGET_LABELS, len(features))

        # Basic checks
        if not mask.any():
            logging.warning("No valid targets found. Exiting.")
            return

        from sklearn.neighbors import NearestNeighbors
        edge_list = []
        k = min(10, len(features) - 1)
        nbrs = NearestNeighbors(n_neighbors=k).fit(features)
        _, indices = nbrs.kneighbors(features)
        for i in range(len(features)):
            for j in indices[i]:
                if i != j:
                    edge_list.append((i, j))

        # Preprocess
        feature_scaler = RobustScaler()
        target_scaler = RobustScaler()
        X_normalized = feature_scaler.fit_transform(features)
        Y_scaled_np = target_scaler.fit_transform(targets[mask])
        Y = torch.tensor(Y_scaled_np, dtype=torch.float)

        edge_index = torch.tensor(edge_list, dtype=torch.long).t().contiguous()
        X_soil = torch.tensor(X_normalized[:, 6:10], dtype=torch.float)

        node_indices = np.where(mask)[0]
        idx = torch.randperm(len(node_indices))
        train_size = int(0.7 * len(idx))
        val_size = int(0.15 * len(idx))
        train_idx = node_indices[idx[:train_size]]
        val_idx = node_indices[idx[train_size:train_size+val_size]]
        test_idx = node_indices[idx[train_size+val_size:]]

        train_mask = torch.tensor([i in train_idx for i in range(len(features))], dtype=torch.bool)
        val_mask = torch.tensor([i in val_idx for i in range(len(features))], dtype=torch.bool)
        test_mask = torch.tensor([i in test_idx for i in range(len(features))], dtype=torch.bool)

        # Models
        hidden_dim = 32
        emb_dim = 64
        agent_soil = AgentGAT(in_dim=4, hidden_dim=hidden_dim, embedding_dim=emb_dim, num_heads=4, dropout=0.2)
        meta_agent = MetaAgent(total_embedding_dim=emb_dim, hidden_dims=[128, 64], out_dim=len(TARGET_LABELS), dropout=0.2)

        optimizer = torch.optim.AdamW(
            list(agent_soil.parameters()) + list(meta_agent.parameters()), lr=0.0005, weight_decay=1e-4
        )
        loss_fn = nn.HuberLoss(delta=1.0)
        device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
        agent_soil.to(device)
        meta_agent.to(device)
        X_soil = X_soil.to(device)
        edge_index = edge_index.to(device)
        Y_train = torch.zeros((len(features), len(TARGET_LABELS)), dtype=torch.float).to(device)
        Y_train[mask] = Y.to(device)

        train_losses = []
        val_losses = []

        for epoch in range(300):
            agent_soil.train()
            meta_agent.train()
            optimizer.zero_grad()

            emb_soil = agent_soil(X_soil, edge_index)
            fused_embedding = emb_soil
            pred = meta_agent(fused_embedding)
            loss = loss_fn(pred[train_mask], Y_train[train_mask])
            loss.backward()
            optimizer.step()
            train_losses.append(loss.item())

            agent_soil.eval()
            meta_agent.eval()
            with torch.no_grad():
                val_pred = meta_agent(agent_soil(X_soil, edge_index))
                val_loss = loss_fn(val_pred[val_mask], Y_train[val_mask])
                val_losses.append(val_loss.item())

            if epoch % 10 == 0:
                logging.info(f"Epoch {epoch}: Train Loss = {loss.item():.4f}, Val Loss = {val_loss.item():.4f}")

        # Test Evaluation
        agent_soil.eval()
        meta_agent.eval()
        with torch.no_grad():
            test_pred = meta_agent(agent_soil(X_soil, edge_index))[test_mask].cpu().numpy()
            test_true = Y_train[test_mask].cpu().numpy()
            mse = mean_squared_error(test_true, test_pred)
            r2 = r2_score(test_true, test_pred)
            logging.info(f"✅ [SOIL-ONLY] Test MSE: {mse:.5f}, R²: {r2:.5f}")

    finally:
        conn.close()

if __name__ == "__main__":
    main()
