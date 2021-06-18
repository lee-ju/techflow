# TechFlow
## Installation

`pip install git+https://github.com/lee-ju/techflow.git`

## Usage

### IPC Code Network
```python
from techflow.nx_tech import nx_preps
from techflow.nx_tech import nx_utils

## Read dataset
sample_ipcs = pd.read_csv('sample_ipc.csv')
ipcs = sample_ipcs['all_ipcs'].tolist()

## Preprocessing
ipcs_preps = nx_preps(ipcs)
ipcs_df = ipcs_preps.edges(obj='ipcs', num_slice=4, spliter='||')

## Visualizing
ipcs_utils = nx_utils(ipcs_df, direct=False)
ipcs_G = ipcs_utils.nx_viz(fs=[5, 5], with_labels=True,
                           node_size=100, node_color='red', seed=15)
                           
## Network Centrality
ipcs_central = ipcs_utils.nx_centrality(top_k=10)
print(ipcs_central)
```

### Forward Citation Network
```python
from techflow.nx_tech import nx_preps
from techflow.nx_tech import nx_utils

## Read dataset
sample_forws = pd.read_csv('sample_forw.csv')
apps = sample_forws['Reg_id'].tolist()
forws = sample_forws['Forw_in_id'].tolist()

## Preprocessing
forws_preps = nx_preps(x=apps, y=forws)
forws_df = forws_preps.edges(obj='forws', num_slice=0, spliter='||')

## Visualizing
forws_utils = nx_utils(forws_df, direct=False)
forws_G = forws_utils.nx_viz(fs=[5, 5], with_labels=True,
                             node_size=100, node_color='blue', seed=15)

## Network Centrality
forws_central = forws_utils.nx_centrality(top_k=10)
print(forws_central)
```

## Parameters

### `techflow.nx_tech`
- `nx_preps` constructor:
    1. `x`: The data for social network analysis. On the input will always be list.
    2. `y`: Second data for citation network analysis On the input will always be list. (default: None)

- `nx_preps.edges` constructor:
    1. `obj`: 'ipcs' for IPC code network, 'forws' for citation network. (default: 'ipcs')
    2. `num_slice`: An argument to how much to truncate the code from behind. (default: 4)
    3. `spliter`: An arguments to break code(default: '||')

- `nx_utils` constructor:
    1. `df`: Dataframe of edgelist.
    2. `direct`: Boolean controlling the DiGraph. (default: True)

- `nx_utils.viz` constructor:
    1. `fs`: List of figsize=[horizontal_size, vertical_size]. (default: [10, 10])
    2. `with_labels`: Boolean controlling the use of node labels. (default: True)
    3. `node_size`: Size of nodes. (default: 100)
    4. `node_color`: Node color. (default: 'red')
    5. `font_size`: Size of labels. (default: 12)
    6. `font_color`: Node label. (default: 'black')
    7. `seed`: Seed for random visualization. (default: 10)

- `nx_utils.centrality` constructor:
    1. `top_k`: Return centrality by top_k. (default: 10)

## TODO
- [x] IPC Network
- [x] Citation Network
- [ ] ...
