## cd "/Users/lukaspertl/Library/Mobile Documents/com~apple~CloudDocs/Cambridge part III/Project" ##
from GraphGeneration import *
from Model import *

if __name__ == '__main__':
    # 1. Simple chains (no correlations)
    generator_simple = SyntheticGraphDataGenerator(mode="simple", num_categories=3, p=0.25, num_nodes=20)
    data_simple = generator_simple.generate_data(num_samples=5)
    print("Simple mode sample:")
    for d in data_simple:
        print(d)

    # 2. Simple chains with correlations
    # Create a dummy candidate transition matrix for demonstration.
    dummy_matrix = torch.ones(3, 3) / 3
    generator_corr = SyntheticGraphDataGenerator(mode="correlated", num_categories=3, p=0.25, num_nodes=20,
                                                   candidate_matrices=[dummy_matrix])
    data_corr = generator_corr.generate_data(num_samples=5)
    print("\nCorrelated mode sample:")
    for d in data_corr:
        print(d)

    # 3. Graphs with motif topology
    generator_motif = SyntheticGraphDataGenerator(mode="motif", motif_dim=3)
    data_motif = generator_motif.generate_data(num_samples=5)
    print("\nMotif mode sample:")
    for d in data_motif:
        print(d)

    # 4. Combined features (motif + correlated embeddings)
    generator_combined = SyntheticGraphDataGenerator(mode="combined", num_categories=5, p=0.35, motif_dim=3,
                                                       chain_length_min=2, chain_length_max=7)
    data_combined = generator_combined.generate_data(num_samples=5)
    print("\nCombined mode sample:")
    for d in data_combined:
        print(d)



if __name__ == '__main__':
    # Example 1: Simple GCN-based model
    model_gcn = GNNModel(model_type="GCN", in_dim=3, hidden_dims=[4, 4], out_dim=3, freeze_final=True)
    print("GCN-based Model:")
    print(model_gcn)

    # Example 2: Simple GIN-based model
    model_gin = GNNModel(model_type="GIN", in_dim=3, hidden_dims=[4, 4], out_dim=3, freeze_final=True)
    print("\nGIN-based Model:")
    print(model_gin)

    # Create dummy inputs to test a forward pass
    # For demonstration, we create a fake batch with 6 nodes divided in 2 graphs.
    x_dummy = torch.randn(6, 3)  # 6 nodes, feature dimension = 3
    # Create a simple edge index (e.g. a chain)
    edge_index_dummy = torch.tensor([[0, 1, 2, 3, 4],
                                     [1, 2, 3, 4, 5]], dtype=torch.long)
    # Create a batch vector: first 3 nodes in graph 0, next 3 nodes in graph 1.
    batch_dummy = torch.tensor([0, 0, 0, 1, 1, 1], dtype=torch.long)

    # Run forward passes for both models
    out_gcn = model_gcn(x_dummy, edge_index_dummy, batch_dummy)
    out_gin = model_gin(x_dummy, edge_index_dummy, batch_dummy)
    print("\nGCN Model Output (logits):", out_gcn)
    print("GIN Model Output (logits):", out_gin)