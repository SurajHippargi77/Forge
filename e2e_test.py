import urllib.request, json, time, sys

API = 'http://localhost:8000/api/v1'

def api(method, path, data=None):
    url = API + path
    body = json.dumps(data).encode() if data else None
    req = urllib.request.Request(url, data=body, method=method)
    req.add_header('Content-Type', 'application/json')
    try:
        resp = urllib.request.urlopen(req)
        return json.loads(resp.read().decode())
    except urllib.error.HTTPError as e:
        err = e.read().decode()
        print(f'  ERROR {e.code}: {method} {path} -> {err[:300]}')
        return None

print('='*60)
print('STEP 1: Create Graph')
print('='*60)
graph = api('POST', '/graphs', {'name': 'E2E Test Model', 'description': 'End-to-end test'})
if graph:
    print(f'  OK: Created graph id={graph["id"]}')
    gid = graph['id']
else:
    print('  FAIL: Could not create graph'); sys.exit(1)

# Check if graph appears in list WITH versions
graphs = api('GET', '/graphs')
gobj = [g for g in graphs if g['id'] == gid][0]
vers_from_list = gobj.get('versions', [])
print(f'  Graph list: versions_count={len(vers_from_list)}')
if len(vers_from_list) > 0:
    print(f'  V1 auto-created: id={vers_from_list[0]["id"]}, nodes={vers_from_list[0]["nodes"]}, edges={vers_from_list[0]["edges"]}')
    v1_id = vers_from_list[0]['id']
else:
    print('  WARNING: No versions in list response')
    v1_id = None

# Fetch lineage tree
tree = api('GET', f'/versions/graphs/{gid}/lineage-tree')
print(f'  Lineage tree: V{tree.get("version_number")}, children={len(tree.get("children", []))}')

print()
print('='*60)
print('STEP 2: Save V2 (with parent_version_id=V1)')
print('='*60)
nodes = [
    {'id': 'input-1', 'type': 'input', 'label': 'Input', 'params': {}, 'position': {'x': 250, 'y': 0}},
    {'id': 'dense-1', 'type': 'dense', 'label': 'Dense', 'params': {'units': 64}, 'position': {'x': 250, 'y': 100}},
    {'id': 'relu-1',  'type': 'relu',  'label': 'ReLU',  'params': {}, 'position': {'x': 250, 'y': 200}},
    {'id': 'output-1','type': 'output','label': 'Output', 'params': {}, 'position': {'x': 250, 'y': 300}},
]
edges = [
    {'id': 'e-1', 'source': 'input-1', 'target': 'dense-1'},
    {'id': 'e-2', 'source': 'dense-1', 'target': 'relu-1'},
    {'id': 'e-3', 'source': 'relu-1',  'target': 'output-1'},
]
v2 = api('POST', '/versions', {'graph_id': gid, 'parent_version_id': v1_id, 'nodes': nodes, 'edges': edges})
if v2:
    v2_id = v2['id']
    print(f'  OK: V2 id={v2_id}, version_number={v2["version_number"]}, nodes={len(v2["nodes"])}, edges={len(v2["edges"])}')
else:
    print('  FAIL: Could not save V2'); sys.exit(1)

# Check lineage tree now shows V1->V2
tree = api('GET', f'/versions/graphs/{gid}/lineage-tree')
print(f'  Lineage: V{tree["version_number"]} -> children={[f"V{c["version_number"]}" for c in tree["children"]]}')

print()
print('='*60)
print('STEP 3: Run sweep on V2')
print('='*60)
param_grid = {'learning_rate': [0.01, 0.001], 'batch_size': [32, 64], 'epochs': [10], 'optimizer': ['adam', 'sgd']}
exp_ids = api('POST', f'/experiments/sweep?version_id={v2_id}', param_grid)
if exp_ids:
    print(f'  OK: Sweep created {len(exp_ids)} experiments (expected 8)')
else:
    print('  FAIL: Sweep failed'); sys.exit(1)

print('  Polling...')
for i in range(20):
    time.sleep(2)
    exps = api('GET', f'/experiments/versions/{v2_id}/experiments')
    sts = [e['status'] for e in exps]
    done = sts.count('completed') + sts.count('failed')
    print(f'    Poll {i+1}: completed={sts.count("completed")}, running={sts.count("running")}, pending={sts.count("pending")}')
    if done == len(exp_ids):
        break

exps = api('GET', f'/experiments/versions/{v2_id}/experiments')
completed = [e for e in exps if e['status']=='completed']
print(f'  Final: {len(completed)}/{len(exps)} completed')
best = min(completed, key=lambda e: e['metrics']['val_loss']) if completed else None
if best:
    print(f'  Best: #{best["id"]} val_loss={best["metrics"]["val_loss"]} acc={best["metrics"]["accuracy"]}')

print()
print('='*60)
print('STEP 4: Add BatchNorm -> V3')
print('='*60)
nodes_v3 = [
    {'id': 'input-1',    'type': 'input',     'label': 'Input',     'params': {}, 'position': {'x': 250, 'y': 0}},
    {'id': 'dense-1',    'type': 'dense',     'label': 'Dense',     'params': {'units': 64}, 'position': {'x': 250, 'y': 100}},
    {'id': 'batchnorm-1','type': 'batchnorm', 'label': 'BatchNorm', 'params': {}, 'position': {'x': 250, 'y': 200}},
    {'id': 'relu-1',     'type': 'relu',      'label': 'ReLU',      'params': {}, 'position': {'x': 250, 'y': 300}},
    {'id': 'output-1',   'type': 'output',    'label': 'Output',    'params': {}, 'position': {'x': 250, 'y': 400}},
]
edges_v3 = [
    {'id': 'e-1', 'source': 'input-1',     'target': 'dense-1'},
    {'id': 'e-2', 'source': 'dense-1',     'target': 'batchnorm-1'},
    {'id': 'e-bn','source': 'batchnorm-1', 'target': 'relu-1'},
    {'id': 'e-3', 'source': 'relu-1',      'target': 'output-1'},
]
v3 = api('POST', '/versions', {'graph_id': gid, 'parent_version_id': v2_id, 'nodes': nodes_v3, 'edges': edges_v3})
if v3:
    v3_id = v3['id']
    print(f'  OK: V3 id={v3_id}, version_number={v3["version_number"]}, nodes={len(v3["nodes"])}, edges={len(v3["edges"])}')
else:
    print('  FAIL: Could not save V3'); sys.exit(1)

tree = api('GET', f'/versions/graphs/{gid}/lineage-tree')
def print_tree(node, indent=0):
    print(f'  {"  "*indent}V{node["version_number"]} (id={node["id"]})')
    for c in node.get('children', []):
        print_tree(c, indent+1)
print('  Lineage tree:')
print_tree(tree)

print()
print('='*60)
print('STEP 5: Compare V2 vs V3 (diff + impact)')
print('='*60)
diff = api('POST', f'/versions/diff?version_a_id={v2_id}&version_b_id={v3_id}')
if diff:
    print(f'  Diff OK: added_nodes={len(diff["added_nodes"])}, removed={len(diff["removed_nodes"])}, modified={len(diff["modified_nodes"])}, +edges={len(diff["added_edges"])}, -edges={len(diff["removed_edges"])}')
    for n in diff['added_nodes']:
        print(f'    Added: {n["label"]} ({n["type"]})')
else:
    print('  FAIL: Diff failed')

impact = api('POST', f'/versions/impact?version_a_id={v2_id}&version_b_id={v3_id}')
if impact:
    print(f'  Impact OK: level={impact["impact_level"]}, score={impact.get("impact_score")}, delta={impact.get("metric_delta")}')
else:
    print('  FAIL: Impact analysis failed (likely schema validation error)')

print()
print('='*60)
print('STEP 6: Custom Algorithm')
print('='*60)
code = """class MyAttention(nn.Module):
    def __init__(self, dim):
        super().__init__()
        self.scale = dim ** -0.5
    def forward(self, x):
        return x * self.scale"""
algo = api('POST', '/algorithms', {'name': 'MyAttention', 'description': 'Custom attention layer', 'code': code})
if algo:
    print(f'  OK: Algorithm id={algo["id"]}, name={algo["name"]}')
else:
    print('  FAIL: Could not save algorithm')

algos = api('GET', '/algorithms')
print(f'  Total algorithms: {len(algos)}')

print()
print('='*60)
print('SUMMARY')
print('='*60)
print(f'Step 1 (Create Graph):   PASS')
print(f'Step 2 (Save Version):   PASS')
print(f'Step 3 (Run Sweep):      {"PASS" if exp_ids and len(exp_ids)==8 else "FAIL"}')
print(f'Step 4 (BatchNorm V3):   PASS')
print(f'Step 5 (Compare):        {"PASS" if diff and impact else "FAIL - impact=" + str(impact is not None)}')
print(f'Step 6 (Algorithm):      {"PASS" if algo else "FAIL"}')
