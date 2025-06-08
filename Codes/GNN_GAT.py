import os
import torch
import numpy as np
from neo4j import GraphDatabase
from torch_geometric.data import Data
from torch_geometric.nn import GATConv
from torch_geometric.utils import add_self_loops, softmax
import torch.nn.functional as F
from sklearn.model_selection import train_test_split
import matplotlib.pyplot as plt
import networkx as nx
from matplotlib.collections import LineCollection
from sklearn.preprocessing import MinMaxScaler
############################################
# Part 1: Fetch Data from Neo4j
############################################
def fetch_nodes_and_edges(driver):
    with driver.session() as session:
        nodes_result = session.run("""
            MATCH (n)
            WHERE n:Weather OR n:Soil OR n:Crop OR n:Fertilizer
            RETURN elementId(n) AS id, labels(n) AS labels, n AS props
        """)
        nodes = {}
        for record in nodes_result:
            node_id = record["id"]
            labels = record["labels"]
            props = dict(record["props"])
            nodes[node_id] = {"labels": labels, "props": props}
        
        edges_result = session.run("""
            MATCH (n)-[r]->(m)
            RETURN elementId(n) AS source, elementId(m) AS target
        """)
        edges = []
        for record in edges_result:
            edges.append((record["source"], record["target"]))
    return nodes, edges

############################################
# Part 2: Build Unified Node Feature Vectors
############################################
def build_feature_vectors(nodes):
    # Collect unique categories for one-hot encoding.
    soil_set = set()
    crop_set = set()
    fertilizer_set = set()
    for node in nodes.values():
        labels = node["labels"]
        props = node["props"]
        if "Soil" in labels:
            soil_set.add(props.get("Soil Type", ""))
        elif "Crop" in labels:
            crop_set.add(props.get("Crop Type", ""))
        elif "Fertilizer" in labels:
            fertilizer_set.add(props.get("Fertilizer_Name", ""))
    soil_types = sorted(list(soil_set))
    crop_types = sorted(list(crop_set))
    fertilizer_names = sorted(list(fertilizer_set))
    
    # Define dimensions:
    dim_weather = 3  # Temparature, Humidity, Moisture
    dim_soil = len(soil_types)
    dim_crop = len(crop_types)
    dim_fertilizer_numeric = 3  # Nitrogen, Potassium, Phosphorous
    dim_fertilizer_onehot = len(fertilizer_names)
    total_dim = dim_weather + dim_soil + dim_crop + dim_fertilizer_numeric + dim_fertilizer_onehot
    
    def one_hot_encode(value, categories):
        vec = [0] * len(categories)
        if value in categories:
            vec[categories.index(value)] = 1
        return vec

    def create_feature_vector(label, props):
        vector = [0] * total_dim
        if "Weather" in label:
            vector[0:3] = [
                float(props.get("Temparature", 0)),
                float(props.get("Humidity", 0)),
                float(props.get("Moisture", 0))
            ]
        elif "Soil" in label:
            start = dim_weather
            vector[start:start+dim_soil] = one_hot_encode(props.get("Soil Type", ""), soil_types)
        elif "Crop" in label:
            start = dim_weather + dim_soil
            vector[start:start+dim_crop] = one_hot_encode(props.get("Crop Type", ""), crop_types)
        elif "Fertilizer" in label:
            start = dim_weather + dim_soil + dim_crop
            vector[start:start+3] = [
                float(props.get("Nitrogen", 0)),
                float(props.get("Potassium", 0)),
                float(props.get("Phosphorous", 0))
            ]
            start2 = start + 3
            vector[start2:start2+dim_fertilizer_onehot] = one_hot_encode(props.get("Fertilizer_Name", ""), fertilizer_names)
        return vector

    node_features = {}
    for node_id, node in nodes.items():
        label = node["labels"][0]  # Assume one primary label per node.
        node_features[node_id] = create_feature_vector(label, node["props"])
    return node_features, total_dim

############################################
# Part 3: Build PyG Data Object from Neo4j Data
############################################
def build_pyg_data(nodes, edges, node_features):
    neo_ids = list(nodes.keys())
    id_to_idx = {node_id: idx for idx, node_id in enumerate(neo_ids)}
    features = [node_features[node_id] for node_id in neo_ids]

    # Normalize features
    scaler = MinMaxScaler()
    features = scaler.fit_transform(features)
    x = torch.tensor(features, dtype=torch.float)
    
    if x.dim() < 2:
        raise ValueError("No node features found. Ensure that your Neo4j query returned nodes.")

    # Map edges
    mapped_edges = []
    for src, tgt in edges:
        if src in id_to_idx and tgt in id_to_idx:
            mapped_edges.append((id_to_idx[src], id_to_idx[tgt]))
    edge_index = torch.tensor(mapped_edges, dtype=torch.long).t().contiguous()
    num_nodes = x.shape[0]
    
    # Create and normalize y AFTER knowing num_nodes
    y = torch.rand(num_nodes, dtype=torch.float)  # Dummy targets
    scaler_y = MinMaxScaler()
    y = scaler_y.fit_transform(y.reshape(-1, 1)).flatten()  # Scale y into [0, 1]
    y = torch.tensor(y, dtype=torch.float)

    data = Data(x=x, edge_index=edge_index, y=y)
    return data, id_to_idx
############################################
# Part 4: Define Custom GATConv with Attention Storage
############################################
from torch_geometric.nn import GATConv

class GATConvWithAttentions(GATConv):
    def __init__(self, in_channels, out_channels, heads=1, concat=True,
                 negative_slope=0.2, dropout=0., add_self_loops=True, bias=True, **kwargs):
        super(GATConvWithAttentions, self).__init__(
            in_channels, out_channels, heads=heads, concat=concat,
            negative_slope=negative_slope, dropout=dropout, add_self_loops=add_self_loops,
            bias=bias, **kwargs)
        self.attention_weights = None  # To store attention coefficients

    def forward(self, x, edge_index, size=None, return_attention_weights=False):
        if self.add_self_loops:
            edge_index, _ = add_self_loops(edge_index, num_nodes=x.size(0))
        
        x = self.lin(x) if self.lin is not None else x
        N = x.size(0)
        H = self.heads
        C = self.out_channels
        x = x.view(N, H, C)
        
        x_src = x[edge_index[0]]
        x_dst = x[edge_index[1]]
        alpha = (x_src * self.att_src).sum(dim=-1) + (x_dst * self.att_dst).sum(dim=-1)
        alpha = F.leaky_relu(alpha, self.negative_slope)
        alpha = softmax(alpha, edge_index[0])
        alpha = F.dropout(alpha, p=self.dropout, training=self.training)
        
        self.attention_weights = alpha  # Shape: [num_edges, heads]
        
        out = self.propagate(edge_index, x=x, alpha=alpha)
        if self.concat:
            out = out.view(-1, H * C)
        else:
            out = out.mean(dim=1)
        if self.bias is not None:
            out = out + self.bias
        return out

    def message(self, x_j, alpha):
        return alpha.unsqueeze(-1) * x_j

############################################
# Part 5: Define the GAT Model Using Custom Layers
############################################
class GAT(torch.nn.Module):
    def __init__(self, num_node_features, hidden_channels, num_classes, heads):
        super(GAT, self).__init__()
        torch.manual_seed(12345)
        self.conv1 = GATConvWithAttentions(num_node_features, hidden_channels, heads=heads)
        self.conv2 = GATConvWithAttentions(hidden_channels * heads, num_classes, heads=1, concat=False)
    
    def forward(self, x, edge_index):
        x = self.conv1(x, edge_index)
        x = F.relu(x)
        x = F.dropout(x, p=0.5, training=self.training)
        x = self.conv2(x, edge_index)
        return x

############################################
# Part 6: Train the GAT Model with Smooth L1 Loss and Early Stopping
############################################
def train_gat(data, num_epochs=400, heads=8, hidden_channels=8, patience=30, learning_rate=0.001):
    num_node_features = data.x.shape[1]
    num_classes = 1  # Regression dummy target (replace if needed)
    model = GAT(num_node_features, hidden_channels, num_classes, heads)
    print(model)
    
    optimizer = torch.optim.AdamW(model.parameters(), lr=learning_rate)  # AdamW as requested
    criterion = torch.nn.SmoothL1Loss()  # Use Smooth L1 Loss instead of MSE

    # Split nodes into train/test sets
    num_nodes = data.num_nodes
    indices = np.arange(num_nodes)
    train_indices, test_indices = train_test_split(indices, test_size=0.2, random_state=42)
    train_mask = torch.zeros(num_nodes, dtype=torch.bool)
    test_mask = torch.zeros(num_nodes, dtype=torch.bool)
    train_mask[train_indices] = True
    test_mask[test_indices] = True
    data.train_mask = train_mask
    data.test_mask = test_mask

    best_loss = float('inf')
    best_model_state = None
    patience_counter = 0

    for epoch in range(num_epochs):
        model.train()
        optimizer.zero_grad()
        out = model(data.x, data.edge_index)
        loss = criterion(out[data.train_mask].squeeze(), data.y[data.train_mask])
        loss.backward()
        optimizer.step()

        model.eval()
        with torch.no_grad():
            out = model(data.x, data.edge_index)
            test_loss = criterion(out[data.test_mask].squeeze(), data.y[data.test_mask])

        print(f"Epoch {epoch:03d}: Train Loss={loss:.4f}, Test Loss={test_loss:.4f}")

        # Early stopping logic
        if test_loss < best_loss:
            best_loss = test_loss
            best_model_state = model.state_dict()
            patience_counter = 0  # reset patience
        else:
            patience_counter += 1

        if patience_counter >= patience:
            print(f"Early stopping triggered at epoch {epoch:03d}.")
            break

    # Load best model
    model.load_state_dict(best_model_state)
    return model


def sample_connected_subgraph(G, sample_size):
    """Find a connected subgraph by performing a BFS from a random start node."""
    start = np.random.choice(list(G.nodes()))
    nodes = set()
    queue = [start]
    while queue and len(nodes) < sample_size:
        cur = queue.pop(0)
        if cur not in nodes:
            nodes.add(cur)
            neighbors = list(G.successors(cur)) + list(G.predecessors(cur))
            queue.extend(neighbors)
    return G.subgraph(nodes)

def visualize_attention(data, model, nodes, id_to_idx, sample_size=500):  # Increased sample size
    """
    Visualizes attention weights from the first GAT layer on a connected subgraph.
    Includes node labels, color-coding, and textual output of top relationships.
    """
    model.eval()
    with torch.no_grad():
        _ = model(data.x, data.edge_index)
        att_weights = model.conv1.attention_weights  # Shape: [E, heads]
        att_weights_avg = att_weights.mean(dim=1).cpu().numpy()  # Average over heads

    # Convert data to NetworkX graph
    G = nx.DiGraph()
    edge_index_np = data.edge_index.cpu().numpy()
    for i in range(edge_index_np.shape[1]):
        src = int(edge_index_np[0, i])
        tgt = int(edge_index_np[1, i])
        G.add_edge(src, tgt, weight=att_weights_avg[i])

    # Sample a connected subgraph of desired size
    subG = sample_connected_subgraph(G, sample_size)

    if subG.number_of_edges() == 0:
        print("No edges in the subgraph to visualize.")
        return

    pos = nx.spring_layout(subG, seed=42)  # Generate positions for nodes

    # Node labels and colors
    node_labels = {idx: nodes[list(nodes.keys())[list(id_to_idx.values()).index(idx)]]['labels'][0] for idx in subG.nodes()}  # Get node labels
    node_colors = []
    for node_idx in subG.nodes():
        node_id = list(nodes.keys())[list(id_to_idx.values()).index(node_idx)]
        node_type = nodes[node_id]['labels'][0]
        if node_type == "Weather":
            node_colors.append('red')
        elif node_type == "Soil":
            node_colors.append('green')
        elif node_type == "Crop":
            node_colors.append('blue')
        elif node_type == "Fertilizer":
            node_colors.append('orange')
        else:
            node_colors.append('gray')  # Default color

    # Extract edge weights for visualization
    edge_weights = [subG[u][v]['weight'] for u, v in subG.edges()]
    if len(edge_weights) == 0:
        print("No edge weights available for visualization.")
        return

    # Normalize edge weights for color mapping
    norm_weights = (np.array(edge_weights) - np.min(edge_weights)) / (np.max(edge_weights) - np.min(edge_weights) + 1e-9)

    # Prepare edge positions correctly for LineCollection
    edges = np.array([[pos[u], pos[v]] for u, v in subG.edges()])

    # Draw the graph using matplotlib
    fig, ax = plt.subplots(figsize=(20, 16))  # Adjusted figure size

    # Draw nodes and labels
    nx.draw_networkx_nodes(subG, pos, node_size=500, ax=ax, node_color=node_colors)  # Adjusted node size
    nx.draw_networkx_labels(subG, pos, labels=node_labels, font_size=10, ax=ax)  # Use node labels

    # Prepare edge colors using LineCollection for proper visualization
    lc = LineCollection(edges, array=norm_weights, cmap=plt.cm.viridis, linewidths=2)

    ax.add_collection(lc)
    plt.colorbar(lc, ax=ax, label="Attention Weight")

    ax.set_title("Attention Weights Visualization on Sampled Subgraph", fontsize=16)  # Adjusted title fontsize
    plt.axis('off')

    # Textual Output: Top Relationships
    top_relationships = sorted(((u, v, subG[u][v]['weight']) for u, v in subG.edges()), key=lambda x: x[2], reverse=True)[:10]
    print("\nTop Relationships by Attention Weight:")
    for src, tgt, weight in top_relationships:
        src_node_id = list(nodes.keys())[list(id_to_idx.values()).index(src)]
        tgt_node_id = list(nodes.keys())[list(id_to_idx.values()).index(tgt)]
        src_label = nodes[src_node_id]['labels'][0]
        tgt_label = nodes[tgt_node_id]['labels'][0]
        print(f"  {src_label} ({src}) -> {tgt_label} ({tgt}): {weight:.4f}")

    plt.show()

############################################
# Main Execution
############################################
neo4j_uri = "bolt://localhost:7689"
neo4j_user = "neo4j"
neo4j_pass = "123456789"
driver = GraphDatabase.driver(neo4j_uri, auth=(neo4j_user, neo4j_pass))
nodes, edges = fetch_nodes_and_edges(driver)
driver.close()

node_features, total_dim = build_feature_vectors(nodes)
print("Total feature dimension:", total_dim)
data_neo4j, id_to_idx = build_pyg_data(nodes, edges, node_features)
print("Constructed KG Data from Neo4j:")
print(data_neo4j)

trained_model = train_gat(data_neo4j, num_epochs=400, heads=8, hidden_channels=8, patience=30,learning_rate=0.001)

# Visualize attention on a connected subgraph (sample of 500 nodes)
visualize_attention(data_neo4j, trained_model, nodes, id_to_idx, sample_size=500)
