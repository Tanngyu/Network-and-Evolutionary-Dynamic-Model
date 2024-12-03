import numpy as np
import networkx as nx
import matplotlib.pyplot as plt
from scipy.stats import expon

# Parameters
num_initial_nodes = 5
num_new_nodes = 1000
m = 3  # Number of edges new nodes will connect to existing nodes
mu = 1.0  # Base intensity for edge formation
decay_rate = 0.5  # Decay rate for the incentive function

# Initialize a small complete graph for the network
G = nx.complete_graph(num_initial_nodes)

# Function to calculate Hawkes process intensity
def hawkes_intensity(t, t_events):
    if not t_events:
        return mu
    return mu + sum(np.exp(-decay_rate * (t - t_event)) for t_event in t_events)

# Function to add a new node based on the Hawkes process
def add_node_with_hawkes_process(G, t, t_events):
    intensities = np.array([hawkes_intensity(t, t_events[node]) for node in G.nodes()])
    probabilities = intensities / intensities.sum()  # Normalize to get probabilities
    chosen_nodes = np.random.choice(list(G.nodes), size=m, replace=False, p=probabilities)
    new_node = len(G.nodes)
    G.add_node(new_node)
    for node in chosen_nodes:
        G.add_edge(new_node, node)
        t_events[node].append(new_node)  # Update the event time list for chosen nodes
    t_events.append([])  # New event list for the new node

# Track the time of additions for each node
t_events = [[] for _ in range(num_initial_nodes)]

# Build the network
for _ in range(num_new_nodes):
    t = len(G.nodes)  # Use node index as a proxy for time
    add_node_with_hawkes_process(G, t, t_events)

# Degree distribution for Hawkes network
degrees = G.degree()
degree_values = [d for n, d in degrees]

# Create a BA network
BA_G = nx.barabasi_albert_graph(num_initial_nodes + num_new_nodes, m)
BA_degrees = BA_G.degree()
BA_degree_values = [d for n, d in BA_degrees]

# Plot degree distributions on a log-log scale
plt.figure(figsize=(12, 6))

# Hawkes Network
plt.subplot(1, 2, 1)
counts, bins, patches = plt.hist(degree_values, bins=np.logspace(np.log10(min(degree_values)), np.log10(max(degree_values)), 20), density=True, alpha=0.75, edgecolor='black' )
plt.xlabel('Degree')
plt.ylabel('Probability')
plt.title('Hawkes Network Degree Distribution (Log-Log)')
s
# Barabási-Albert Network
plt.subplot(1, 2, 2)
counts, bins, patches = plt.hist(BA_degree_values, bins=np.logspace(np.log10(min(BA_degree_values)), np.log10(max(BA_degree_values)), 20), density=True, alpha=0.75, edgecolor='black')
plt.xlabel('Degree')
plt.ylabel('Probability')
plt.title('BA Network Degree Distribution (Log-Log)')

plt.tight_layout()
plt.show()
