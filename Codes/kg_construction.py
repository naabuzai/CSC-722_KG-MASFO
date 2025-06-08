import pandas as pd
import torch
from torch_geometric.data import Data
import itertools
from torch_geometric.nn import GCNConv
import torch.nn.functional as F
from sklearn.model_selection import train_test_split

file_path = "C:\\Users\\ALIENWARE\\Downloads\\Fertilizer_Data.csv"
df = pd.read_csv(file_path)

soil_types = sorted(df["Soil Type"].unique())
crop_types = sorted(df["Crop Type"].unique())
fertilizer_names = sorted(df["Fertilizer Name"].unique())

T = {
    'weather': df[['Temparature', 'Humidity', 'Moisture']],
    'soil': df[['Soil Type']],
    'crop': df[['Crop Type']],
    'fertilizer': df[['Fertilizer Name', 'Nitrogen', 'Phosphorous', 'Potassium']]
}

node_id_map = {}
node_features_list = []
current_index = 0

V = set()
E = set()
H = {}

dim_weather = 3
dim_soil = len(soil_types)
dim_crop = len(crop_types)
dim_fert_numeric = 3
dim_fert_onehot = len(fertilizer_names)
total_dim = dim_weather + dim_soil + dim_crop + dim_fert_numeric + dim_fert_onehot

def one_hot(val, classes):
    vec = [0] * len(classes)
    if val in classes:
        vec[classes.index(val)] = 1
    return vec

def create_vector(t, row):
    vec = [0] * total_dim
    if t == "weather":
        vec[0:3] = [row["Temparature"], row["Humidity"], row["Moisture"]]
    elif t == "soil":
        vec[3:3+dim_soil] = one_hot(row["Soil Type"], soil_types)
    elif t == "crop":
        vec[3+dim_soil:3+dim_soil+dim_crop] = one_hot(row["Crop Type"], crop_types)
    elif t == "fertilizer":
        start = 3+dim_soil+dim_crop
        vec[start:start+3] = [row["Nitrogen"], row["Phosphorous"], row["Potassium"]]
        vec[start+3:start+3+dim_fert_onehot] = one_hot(row["Fertilizer Name"], fertilizer_names)
    return vec

def Key(table):
    return tuple(table.columns)

def Type(name):
    return name

def Node(type_name, row):
    return f"{type_name}_{'_'.join(map(str, row.values))}"

for Ti_name, Ti in T.items():
    τi = Type(Ti_name)
    ki = Key(Ti)
    for _, r in Ti.iterrows():
        v_id = tuple(r[ki].values)
        if v_id not in H:
            v = Node(τi, r)
            V.add(v)
            H[v_id] = v
            node_id_map[v] = current_index
            node_features_list.append(create_vector(Ti_name, r))
            current_index += 1

def Link(ri, rj):
    return True if set(ri.index) & set(rj.index) else False

def RelType(ri, rj):
    return f"{ri.name}_to_{rj.name}"

T_items = list(T.items())
for (Ti_name, Ti), (Tj_name, Tj) in itertools.product(T_items, repeat=2):
    for _, ri in Ti.iterrows():
        for _, rj in Tj.iterrows():
            if Link(ri, rj):
                ρ = RelType(ri, rj)
                vi_id = tuple(ri.values)
                vj_id = tuple(rj.values)
                if vi_id in H and vj_id in H:
                    E.add((H[vi_id], ρ, H[vj_id]))

edge_index = [[], []]
for src, _, tgt in E:
    edge_index[0].append(node_id_map[src])
    edge_index[1].append(node_id_map[tgt])

x = torch.tensor(node_features_list, dtype=torch.float)
edge_index = torch.tensor(edge_index, dtype=torch.long)
y = torch.rand(x.size(0))
data = Data(x=x, edge_index=edge_index, y=y)

train_idx, test_idx = train_test_split(torch.arange(x.size(0)), test_size=0.2, random_state=42)
train_mask = torch.zeros(x.size(0), dtype=torch.bool)
test_mask = torch.zeros(x.size(0), dtype=torch.bool)
train_mask[train_idx] = True
test_mask[test_idx] = True
data.train_mask = train_mask
data.test_mask = test_mask

class GCN(torch.nn.Module):
    def __init__(self, in_channels, hidden_channels, out_channels):
        super().__init__()
        self.conv1 = GCNConv(in_channels, hidden_channels)
        self.conv2 = GCNConv(hidden_channels, out_channels)

    def forward(self, x, edge_index):
        x = self.conv1(x, edge_index)
        x = F.relu(x)
        x = F.dropout(x, p=0.5, training=self.training)
        return self.conv2(x, edge_index)

model = GCN(total_dim, 16, 1)
optimizer = torch.optim.Adam(model.parameters(), lr=0.01)
loss_fn = torch.nn.MSELoss()

def train():
    model.train()
    optimizer.zero_grad()
    out = model(data.x, data.edge_index).squeeze()
    loss = loss_fn(out[data.train_mask], data.y[data.train_mask])
    loss.backward()
    optimizer.step()
    return loss.item()

def test():
    model.eval()
    with torch.no_grad():
        out = model(data.x, data.edge_index).squeeze()
        loss = loss_fn(out[data.test_mask], data.y[data.test_mask])
    return loss.item()

for epoch in range(400):
    loss = train()
    test_loss = test()
    if epoch % 50 == 0:
        print(f"Epoch {epoch:03d}, Train Loss: {loss:.4f}, Test Loss: {test_loss:.4f}")
