import numpy as np
import matplotlib.pyplot as plt

data = np.load('../graphdata/gnn_discriminant_score_validation_overfit.npz')
discriminant_scores = data['discriminant_scores']
Lumi_weights = data['lumi_weights']

test = np.load('../graphdata/gnn_discriminant_score_overfit.npz')
test_discriminant_scores = test['discriminant_scores']
Lumi_weights_test = test['lumi_weights']

def main():
    # Create histogram
    plt.figure(figsize=(10,6))
    plt.hist(discriminant_scores, bins=75, weights=Lumi_weights, alpha=0.5, color = 'blue', edgecolor  = 'black')
    plt.title('GNN Overfitted Validation Discriminant Scores Weighted by Lumi Weight')
    plt.xlabel('Discriminant Score (p(+) - p(-))')
    plt.ylabel('Weighted Event Count')
    plt.xlim(-1,1)
    plt.grid(False)
    plt.savefig('../plots/GNN_Discriminant_vs_LumiWeight_validation_overfit.png')
    plt.show()

    #create test histogram
    plt.figure(figsize=(10,6))
    plt.hist(test_discriminant_scores, bins=75, weights=Lumi_weights_test, alpha=0.5, color = 'blue', edgecolor  = 'black')
    plt.title('GNN Overfitted Test Discriminant Scores Weighted by Lumi Weight')
    plt.xlabel('Discriminant Score (p(+) - p(-))')
    plt.ylabel('Weighted Event Count')
    plt.xlim(-1,1)
    plt.grid(False)
    plt.savefig('../plots/GNN_Discriminant_vs_LumiWeight_overfit.png')
    plt.show()


if __name__ == '__main__':
    main()