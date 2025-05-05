## cd "/Users/lukaspertl/Library/Mobile Documents/com~apple~CloudDocs/Cambridge part III/Project" ##
from GraphGeneration import *
from Model import *
from Trainer import *
from Visualizer import *
from torch_geometric.data import DataLoader
from Runner import run_multiple_experiments
from ExperimentalPipeline1 import main

if __name__ == '__main__':
    specific_rows = [202]
    Mode = "simple"
    main(specific_rows, Mode)


# =============================================================================
# Main Execution: Use Simple Mode Data for Training
# =============================================================================
if __name__ == '__main__(SIMPLE)':
    # Set device.
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    
    # Generate training and testing data using "simple" mode.
    generator_simple = SyntheticGraphDataGenerator(mode="simple", num_categories=3, p=0.25, num_nodes=20)
    train_data = generator_simple.generate_data(num_samples=10000)
    test_data = generator_simple.generate_data(num_samples=3000)
    print("Generated Simple Mode Data Samples:")
    for d in train_data[:2]:
        print(d)
    
    # Create DataLoaders.
    train_loader = DataLoader(train_data, batch_size=4, shuffle=True)
    test_loader = DataLoader(test_data, batch_size=4, shuffle=False)
    
    # Instantiate a GNN model. For the simple case, we use a GCN-based model.
    model = GNNModel(model_type="GCN", in_dim=3, hidden_dims=[3,3], out_dim=3, freeze_final=True, pooling="mean").to(device)
    print("\nModel Architecture:")
    print(model)
    
    # Set up optimizer and loss function.
    optimizer = torch.optim.Adam(model.parameters(), lr=0.01)
    criterion = nn.BCEWithLogitsLoss(reduction='none')
    
    # Configuration for Trainer.
    # Since the simple mode has no motif features, set motif_dim=0 and feature_dim=3.
    config = {
        "use_weighting": True,
        "feature_dim": 3,
        "motif_dim": 0,
        "importance": (15.0, 10.0),
        "num_epochs": 5
    }
    
    # Create the Trainer instance.
    trainer = Trainer(model, train_loader, test_loader, optimizer, criterion, device, config)
    
    # Train the model.
    print("\nStarting Training:")
    trainer.train(num_epochs=config["num_epochs"])
    
    # Evaluate the model.
    print("\nEvaluating Model:")
    avg_loss, avg_accuracy, preds_dict, avg_embeddings, avg_predictions = trainer.evaluate()

    # Visualization:
    # In the simple mode, our targets (and hence average predictions) are 3-dimensional one-hot vectors.
    # We use the 3D visualization function.
    colors = ['red', 'green', 'blue']       # one color for each possible target tuple (e.g., (1,0,0), (0,1,0), (0,0,1))
    markers = ['o', 's', '^']                # marker styles for each type
    # Use the keys from avg_predictions (which are pure target tuples) as the keys to plot.
    keys_to_plot = list(avg_embeddings.keys())
    print("\nVisualizing Average Predictions (as 3D Embeddings):")
    Visualizer.plot_avg_hidden_embeddings_3d(avg_embeddings, colors, markers, keys_to_plot)


if __name__ == '__main__(DELETE FOR MULTIPLE RUNS)':
    # Define experiment configuration.
    experiment_config = {
        "mode": "simple",           # Options: "simple", "motif", "correlated", "combined"
        "num_categories": 3,
        "p": 0.25,
        "num_nodes": 20,
        "motif_dim": 0,             # 0 for simple experiments
        "chain_length_min": 2,
        "chain_length_max": 7,
        "num_train_samples": 10000,
        "num_test_samples": 3000,
        "batch_size": 4,
        "in_dim": 3,
        "hidden_dims": [6, 2],
        "lr": 0.01,
        "use_weighting": True,
        "importance": (15.0, 10.0),
        "phase1_epochs": 5,
        "phase2_epochs": 10,
        "num_epochs": 5,
        "device": torch.device('cuda' if torch.cuda.is_available() else 'cpu'),
        "model_type": "GCN"
    }
    
    results = run_multiple_experiments(experiment_config, num_experiments=3)
    
    # Perform geometry analysis on the results.
    config_losses = Trainer.geometry_analysis(results)
    summary = Trainer.summarize_config_losses(config_losses)

    print("\nExperiment Loss Summary:")
    for config, stats in summary.items():
        print(f"{config} : {stats}")