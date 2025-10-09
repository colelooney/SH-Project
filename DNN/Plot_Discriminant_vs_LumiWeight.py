import numpy as np
import matplotlib.pyplot as plt

data = np.load('./data/dnn_discriminant_scores_and_lumi_weights.npz')
discriminant_scores = data['discriminant_scores']
Lumi_weights = data['Lumi_weights']

def main():
    # Create histogram
    plt.figure(figsize=(10,6))
    plt.hist(discriminant_scores, bins=75, weights=Lumi_weights, alpha=0.5, color = 'blue', edgecolor  = 'black')
    plt.title('DNN Discriminant Scores Weighted by Lumi Weight')
    plt.xlabel('Discriminant Score (p(+) - p(-))')
    plt.ylabel('Weighted Event Count')
    plt.xlim(-1,1)
    plt.grid(False)
    plt.savefig('./plots/DNN_Discriminant_vs_LumiWeight.png')
    plt.show()


if __name__ == '__main__':
    main()