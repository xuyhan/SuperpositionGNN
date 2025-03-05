## cd "/Users/lukaspertl/Library/Mobile Documents/com~apple~CloudDocs/Cambridge part III/Project" ##
from GraphGeneration import *
from Model import *
from Trainer import *
from torch_geometric.data import DataLoader

# =============================================================================
# Main Execution: Use Simple Mode Data for Training
# =============================================================================
if __name__ == '__main__':
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
    model = GNNModel(model_type="GCN", in_dim=3, hidden_dims=[4, 4], out_dim=3, freeze_final=True).to(device)
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
    trainer.evaluate()