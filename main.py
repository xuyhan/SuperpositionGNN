## cd "/Users/lukaspertl/Library/Mobile Documents/com~apple~CloudDocs/Cambridge part III/Project" ##
from GraphGeneration import *

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