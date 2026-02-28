"""
Seed script for Forge hackathon demo.
Creates compelling demo data so judges see a rich app immediately.

Usage:
  cd forge
  python -m backend.seed
  # OR
  cd forge/backend
  python seed.py
"""
import sqlite3, os, json, datetime

# Resolve DB path: forge.db lives next to the backend/ folder
SCRIPT_DIR = os.path.dirname(os.path.abspath(__file__))
DB_PATH = os.path.join(SCRIPT_DIR, "..", "forge.db")
DB_PATH = os.path.normpath(DB_PATH)

def now_iso():
    return datetime.datetime.now(datetime.timezone.utc).isoformat()

def seed():
    print(f"Seeding database at: {DB_PATH}")
    conn = sqlite3.connect(DB_PATH)
    conn.execute("PRAGMA journal_mode=WAL")
    c = conn.cursor()

    ts = now_iso()

    # ── 1. Create "Image Classifier" graph ──────────────────────────
    c.execute(
        "INSERT INTO graphs (name, description, created_at) VALUES (?, ?, ?)",
        ("Image Classifier", "CNN-based image classification pipeline", ts),
    )
    graph_id = c.lastrowid
    print(f"  Created graph id={graph_id}")

    # ── 2. Version 1: base model ────────────────────────────────────
    v1_nodes = [
        {"id": "1", "type": "input",  "label": "Input",  "params": {}, "position": {"x": 250, "y": 0}},
        {"id": "2", "type": "dense",  "label": "Dense",  "params": {"units": 128}, "position": {"x": 250, "y": 120}},
        {"id": "3", "type": "relu",   "label": "ReLU",   "params": {}, "position": {"x": 250, "y": 240}},
        {"id": "4", "type": "output", "label": "Output", "params": {}, "position": {"x": 250, "y": 360}},
    ]
    v1_edges = [
        {"id": "e1-2", "source": "1", "target": "2"},
        {"id": "e2-3", "source": "2", "target": "3"},
        {"id": "e3-4", "source": "3", "target": "4"},
    ]
    c.execute(
        "INSERT INTO graph_versions (graph_id, version_number, parent_version_id, nodes, edges, created_at) VALUES (?, ?, ?, ?, ?, ?)",
        (graph_id, 1, None, json.dumps(v1_nodes), json.dumps(v1_edges), ts),
    )
    v1_id = c.lastrowid
    print(f"  Created V1 id={v1_id}  (4 nodes, 3 edges)")

    # ── 3. Version 2: + BatchNorm (child of V1) ────────────────────
    v2_nodes = [
        {"id": "1", "type": "input",     "label": "Input",     "params": {}, "position": {"x": 250, "y": 0}},
        {"id": "2", "type": "dense",     "label": "Dense",     "params": {"units": 128}, "position": {"x": 250, "y": 120}},
        {"id": "5", "type": "batchnorm", "label": "BatchNorm", "params": {}, "position": {"x": 250, "y": 240}},
        {"id": "3", "type": "relu",      "label": "ReLU",      "params": {}, "position": {"x": 250, "y": 360}},
        {"id": "4", "type": "output",    "label": "Output",    "params": {}, "position": {"x": 250, "y": 480}},
    ]
    v2_edges = [
        {"id": "e1-2", "source": "1", "target": "2"},
        {"id": "e2-5", "source": "2", "target": "5"},
        {"id": "e5-3", "source": "5", "target": "3"},
        {"id": "e3-4", "source": "3", "target": "4"},
    ]
    c.execute(
        "INSERT INTO graph_versions (graph_id, version_number, parent_version_id, nodes, edges, created_at) VALUES (?, ?, ?, ?, ?, ?)",
        (graph_id, 2, v1_id, json.dumps(v2_nodes), json.dumps(v2_edges), ts),
    )
    v2_id = c.lastrowid
    print(f"  Created V2 id={v2_id}  (5 nodes, 4 edges) [+BatchNorm]")

    # ── 4. Version 3: + Dropout (alternative branch, child of V1) ──
    v3_nodes = [
        {"id": "1", "type": "input",   "label": "Input",   "params": {}, "position": {"x": 250, "y": 0}},
        {"id": "2", "type": "dense",   "label": "Dense",   "params": {"units": 128}, "position": {"x": 250, "y": 120}},
        {"id": "5", "type": "dropout", "label": "Dropout", "params": {"rate": 0.3}, "position": {"x": 250, "y": 240}},
        {"id": "3", "type": "relu",    "label": "ReLU",    "params": {}, "position": {"x": 250, "y": 360}},
        {"id": "4", "type": "output",  "label": "Output",  "params": {}, "position": {"x": 250, "y": 480}},
    ]
    v3_edges = [
        {"id": "e1-2", "source": "1", "target": "2"},
        {"id": "e2-5", "source": "2", "target": "5"},
        {"id": "e5-3", "source": "5", "target": "3"},
        {"id": "e3-4", "source": "3", "target": "4"},
    ]
    c.execute(
        "INSERT INTO graph_versions (graph_id, version_number, parent_version_id, nodes, edges, created_at) VALUES (?, ?, ?, ?, ?, ?)",
        (graph_id, 3, v1_id, json.dumps(v3_nodes), json.dumps(v3_edges), ts),
    )
    v3_id = c.lastrowid
    print(f"  Created V3 id={v3_id}  (5 nodes, 4 edges) [+Dropout]")

    # ── 5. Experiments for V1 (4 completed) ─────────────────────────
    v1_experiments = [
        {"hp": {"learning_rate": 0.01,  "batch_size": 32, "epochs": 10, "optimizer": "adam"}, "m": {"train_loss": 0.42, "val_loss": 0.48, "accuracy": 0.81}},
        {"hp": {"learning_rate": 0.001, "batch_size": 32, "epochs": 10, "optimizer": "adam"}, "m": {"train_loss": 0.38, "val_loss": 0.44, "accuracy": 0.83}},
        {"hp": {"learning_rate": 0.01,  "batch_size": 64, "epochs": 10, "optimizer": "sgd"},  "m": {"train_loss": 0.51, "val_loss": 0.55, "accuracy": 0.77}},
        {"hp": {"learning_rate": 0.001, "batch_size": 64, "epochs": 10, "optimizer": "adam"}, "m": {"train_loss": 0.35, "val_loss": 0.41, "accuracy": 0.85}},
    ]
    for exp in v1_experiments:
        c.execute(
            "INSERT INTO experiment_runs (version_id, status, hyperparameters, metrics, created_at, completed_at) VALUES (?, ?, ?, ?, ?, ?)",
            (v1_id, "COMPLETED", json.dumps(exp["hp"]), json.dumps(exp["m"]), ts, ts),
        )
    print(f"  Created 4 experiments for V1 (best acc=0.85)")

    # ── 6. Experiments for V2 (4 completed, better metrics) ─────────
    v2_experiments = [
        {"hp": {"learning_rate": 0.01,  "batch_size": 32, "epochs": 10, "optimizer": "adam"}, "m": {"train_loss": 0.31, "val_loss": 0.36, "accuracy": 0.88}},
        {"hp": {"learning_rate": 0.001, "batch_size": 32, "epochs": 10, "optimizer": "adam"}, "m": {"train_loss": 0.28, "val_loss": 0.33, "accuracy": 0.91}},
        {"hp": {"learning_rate": 0.01,  "batch_size": 64, "epochs": 10, "optimizer": "sgd"},  "m": {"train_loss": 0.39, "val_loss": 0.43, "accuracy": 0.86}},
        {"hp": {"learning_rate": 0.001, "batch_size": 64, "epochs": 10, "optimizer": "adam"}, "m": {"train_loss": 0.26, "val_loss": 0.30, "accuracy": 0.93}},
    ]
    for exp in v2_experiments:
        c.execute(
            "INSERT INTO experiment_runs (version_id, status, hyperparameters, metrics, created_at, completed_at) VALUES (?, ?, ?, ?, ?, ?)",
            (v2_id, "COMPLETED", json.dumps(exp["hp"]), json.dumps(exp["m"]), ts, ts),
        )
    print(f"  Created 4 experiments for V2 (best acc=0.93)")

    # ── 7. Custom algorithm ─────────────────────────────────────────
    algo_code = (
        "class SelfAttention(nn.Module):\n"
        "    def __init__(self, dim):\n"
        "        super().__init__()\n"
        "        self.scale = dim ** -0.5\n"
        "    def forward(self, x):\n"
        "        return x * self.scale"
    )
    c.execute(
        "INSERT INTO custom_algorithms (name, description, code, created_at) VALUES (?, ?, ?, ?)",
        ("SelfAttention", "Custom self-attention mechanism", algo_code, ts),
    )
    algo_id = c.lastrowid
    print(f"  Created algorithm id={algo_id} 'SelfAttention'")

    conn.commit()
    conn.close()

    print()
    print("Done! Seed data:")
    print(f"  Graph:       'Image Classifier' (id={graph_id})")
    print(f"  Versions:    V1 (id={v1_id}), V2 +BatchNorm (id={v2_id}), V3 +Dropout (id={v3_id})")
    print(f"  Experiments: 4 for V1, 4 for V2 (8 total)")
    print(f"  Algorithm:   'SelfAttention' (id={algo_id})")
    print()
    print(f"  V1 id={v1_id}  V2 id={v2_id}  (use these for diff/impact endpoints)")

if __name__ == "__main__":
    seed()
