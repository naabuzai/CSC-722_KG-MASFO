import pandas as pd
import numpy as np
import torch
from torch_geometric.data import Data
from sklearn.model_selection import train_test_split
from torch_geometric.nn import GCNConv
import torch.nn.functional as F
import os

# ----- Step 1. Read the CSV Data -----
file_path = "C:\\Users\\ALIENWARE\\Downloads\\Fertilizer_Data.csv"  # Your file path
if not os.path.exists(file_path):
    raise FileNotFoundError(f"File not found: {file_path}")

# Since the file is CSV, we use pd.read_csv.
df = pd.read_csv(file_path)

# ----- Step 2. Set Up Unified Node Feature Dimensions -----
# We have four node types:
# - Weather: using 3 numeric features: [Temparature, Humidity, Moisture]
# - Soil: one-hot for "Soil Type"
# - Crop: one-hot for "Crop Type"
# - Fertilizer: using 3 numeric features [Nitrogen, Potassium, Phosphorous] plus one-hot for "Fertilizer Name"

# Get unique categorical values (for one-hot encoding)
soil_types = sorted(df["Soil Type"].unique())
crop_types = sorted(df["Crop Type"].unique())
fertilizer_names = sorted(df["Fertilizer Name"].unique())

# Define dimensions for each node type
dim_weather = 3
dim_soil = len(soil_types)
dim_crop = len(crop_types)
dim_fertilizer_numeric = 3
dim_fertilizer_onehot = len(fertilizer_names)

# Total unified dimension: every node will be represented in this space.
total_dim = dim_weather + dim_soil + dim_crop + dim_fertilizer_numeric + dim_fertilizer_onehot

def one_hot_encode(value, categories):
    vec = [0] * len(categories)
    if value in categories:
        vec[categories.index(value)] = 1
    return vec

def create_feature_vector(node_type, attributes):
    # Create a vector of length total_dim filled with zeros.
    vector = [0] * total_dim
    if node_type == "weather":
        # Place weather numeric features at positions [0:3]
        vector[0:3] = [attributes.get("Temparature", 0),
                       attributes.get("Humidity", 0),
                       attributes.get("Moisture", 0)]
    elif node_type == "soil":
        # Place soil one-hot encoding in positions [dim_weather : dim_weather+dim_soil]
        onehot = one_hot_encode(attributes.get("Soil Type"), soil_types)
        start = dim_weather
        vector[start : start + dim_soil] = onehot
    elif node_type == "crop":
        # Place crop one-hot encoding in positions [dim_weather+dim_soil : dim_weather+dim_soil+dim_crop]
        onehot = one_hot_encode(attributes.get("Crop Type"), crop_types)
        start = dim_weather + dim_soil
        vector[start : start + dim_crop] = onehot
    elif node_type == "fertilizer":
        # For fertilizer, place numeric values in the next block
        start = dim_weather + dim_soil + dim_crop
        vector[start : start + dim_fertilizer_numeric] = [attributes.get("Nitrogen", 0),
                                                          attributes.get("Potassium", 0),
                                                          attributes.get("Phosphorous", 0)]
        # Then place one-hot encoding for fertilizer name
        start2 = start + dim_fertilizer_numeric
        vector[start2 : start2 + dim_fertilizer_onehot] = one_hot_encode(attributes.get("Fertilizer Name"), fertilizer_names)
    return vector

# ----- Step 3. Construct the Knowledge Graph (KG) -----
# We define four node types and add edges between them.
# For each row, we create (or reuse) nodes:
#   Weather → Soil → Crop → Fertilizer, and add a feedback edge Fertilizer → Soil.

node_id_map = {}        # Maps unique node key to an index.
node_features_list = [] # List of unified feature vectors.
edges = []              # List of edges (as tuples of node indices).
current_index = 0

for idx, row in df.iterrows():
    # Weather node: key based on (Temparature, Humidity, Moisture)
    weather_key = ("weather", (row["Temparature"], row["Humidity"], row["Moisture"]))
    if weather_key not in node_id_map:
        node_id_map[weather_key] = current_index
        feat = create_feature_vector("weather", {"Temparature": row["Temparature"],
                                                   "Humidity": row["Humidity"],
                                                   "Moisture": row["Moisture"]})
        node_features_list.append(feat)
        weather_node = current_index
        current_index += 1
    else:
        weather_node = node_id_map[weather_key]
    
    # Soil node: key based on Soil Type.
    soil_key = ("soil", row["Soil Type"])
    if soil_key not in node_id_map:
        node_id_map[soil_key] = current_index
        feat = create_feature_vector("soil", {"Soil Type": row["Soil Type"]})
        node_features_list.append(feat)
        soil_node = current_index
        current_index += 1
    else:
        soil_node = node_id_map[soil_key]
    
    # Crop node: key based on Crop Type.
    crop_key = ("crop", row["Crop Type"])
    if crop_key not in node_id_map:
        node_id_map[crop_key] = current_index
        feat = create_feature_vector("crop", {"Crop Type": row["Crop Type"]})
        node_features_list.append(feat)
        crop_node = current_index
        current_index += 1
    else:
        crop_node = node_id_map[crop_key]
    
    # Fertilizer node: key based on Fertilizer Name and nutrient values.
    fert_key = ("fertilizer", (row["Fertilizer Name"], row["Nitrogen"], row["Potassium"], row["Phosphorous"]))
    if fert_key not in node_id_map:
        node_id_map[fert_key] = current_index
        feat = create_feature_vector("fertilizer", {"Fertilizer Name": row["Fertilizer Name"],
                                                     "Nitrogen": row["Nitrogen"],
                                                     "Potassium": row["Potassium"],
                                                     "Phosphorous": row["Phosphorous"]})
        node_features_list.append(feat)
        fertilizer_node = current_index
        current_index += 1
    else:
        fertilizer_node = node_id_map[fert_key]
    
    # Add edges between nodes:
    # Weather → Soil, Soil → Crop, Crop → Fertilizer, and feedback Fertilizer → Soil.
    edges.append((weather_node, soil_node))
    edges.append((soil_node, crop_node))
    edges.append((crop_node, fertilizer_node))
    edges.append((fertilizer_node, soil_node))

# ----- Step 4. Build the PyTorch Geometric Data Object -----
# All node feature vectors now have the same length (total_dim)
x = torch.tensor(node_features_list, dtype=torch.float)
edge_index = torch.tensor(edges, dtype=torch.long).t().contiguous()

# For demonstration, we create a dummy target vector (e.g., for a regression task).
num_nodes = x.shape[0]
y = torch.rand(num_nodes, dtype=torch.float)

data = Data(x=x, edge_index=edge_index, y=y)
print("Constructed Knowledge Graph Data:")
print(data)

# ----- Optional: Define and Train a GCN Model -----
class GCN(torch.nn.Module):
    def __init__(self, in_channels, hidden_channels, out_channels):
        super(GCN, self).__init__()
        self.conv1 = GCNConv(in_channels, hidden_channels)
        self.conv2 = GCNConv(hidden_channels, out_channels)

    def forward(self, x, edge_index):
        x = self.conv1(x, edge_index)
        x = F.relu(x)
        x = F.dropout(x, p=0.5, training=self.training)
        x = self.conv2(x, edge_index)
        return x

# Create train/test masks.
indices = np.arange(num_nodes)
train_indices, test_indices = train_test_split(indices, test_size=0.2, random_state=42)
train_mask = torch.zeros(num_nodes, dtype=torch.bool)
test_mask = torch.zeros(num_nodes, dtype=torch.bool)
train_mask[train_indices] = True
test_mask[test_indices] = True
data.train_mask = train_mask
data.test_mask = test_mask

# Set up the GCN model.
in_channels = total_dim
hidden_channels = 16
out_channels = 1  # For regression output
model = GCN(in_channels, hidden_channels, out_channels)
optimizer = torch.optim.Adam(model.parameters(), lr=0.01)
criterion = torch.nn.MSELoss()

def train_model():
    model.train()
    optimizer.zero_grad()
    out = model(data.x, data.edge_index)
    loss = criterion(out[data.train_mask].squeeze(), data.y[data.train_mask])
    loss.backward()
    optimizer.step()
    return loss.item()

def test_model():
    model.eval()
    with torch.no_grad():
        out = model(data.x, data.edge_index)
        loss = criterion(out[data.test_mask].squeeze(), data.y[data.test_mask])
    return loss.item()

print("\nTraining the GCN Model:")
for epoch in range(400):
    loss = train_model()
    test_loss = test_model()
    if epoch % 50 == 0:
        print(f'Epoch: {epoch:03d}, Loss: {loss:.4f}, Test Loss: {test_loss:.4f}')
