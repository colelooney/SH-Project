import numpy as np
import matplotlib.pyplot as plt

data = np.load('../graphdata/gnn_test_results_module.npz')
discriminant_scores = data['discriminant_scores']
Lumi_weights = data['lumi_weights']

# test = np.load('../graphdata/gnn_discriminant_scores_29th_september_low_lr.npz')
# test_discriminant_scores = test['discriminant_scores']
# Lumi_weights_test = test['lumi_weights']

# loss = np.load("../graphdata/training_loss_large_linear.npz")
# training_loss = loss['training_loss']
# epoch = loss['epoch']

def main():
    # Create histogram
    plt.figure(figsize=(10,6))
    plt.hist(discriminant_scores, bins=75, weights=Lumi_weights, alpha=0.5, color = 'blue', edgecolor  = 'black')
    plt.title('GNN Discriminant Scores Weighted by Lumi Weight')
    plt.xlabel('Discriminant Score (p(+) - p(-))')
    plt.ylabel('Weighted Event Count')
    plt.xlim(-1,1)
    plt.grid(False)
    plt.savefig('../plots/GNN_Discriminant_vs_LumiWeight_validation_20th_October_Module.png')
    plt.show()

    #create test histogram
    # plt.figure(figsize=(10,6))
    # plt.hist(test_discriminant_scores, bins=75, weights=Lumi_weights_test, alpha=0.5, color = 'blue', edgecolor  = 'black')
    # plt.title('GNN Test Discriminant Scores Weighted by Lumi Weight')
    # plt.xlabel('Discriminant Score (p(+) - p(-))')
    # plt.ylabel('Weighted Event Count')
    # plt.xlim(-1,1)
    # plt.grid(False)
    # plt.savefig('../plots/GNN_Discriminant_vs_LumiWeight_29th_September_lowlr.png')
    # plt.show()

    # #plot training loss over time
    # plt.figure(figsize = (10,6))
    # plt.plot(epoch, training_loss, linestyle = '-')
    # plt.title('GNN Training loss over time')
    # plt.xlabel('Epoch Number')
    # plt.ylabel('Training Loss')
    # plt.grid(False)
    # plt.savefig('../plots/GNN_training_loss_large_linear')


if __name__ == '__main__':
    main()