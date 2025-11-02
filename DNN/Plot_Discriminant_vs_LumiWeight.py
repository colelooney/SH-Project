import numpy as np
import matplotlib.pyplot as plt

data = np.load('../data/dnn_discriminant_scores_and_lumi_weights_10272025.npz')
discriminant_scores = data['discriminant_scores']
Lumi_weights = data['Lumi_weights']

def main():
    # Create histogram
    # data = np.load('../data/dnn_discriminant_scores_and_lumi_weights_10302025_even_val.npz')
    # discriminant_scores = data['discriminant_scores']
    # Lumi_weights = data['Lumi_weights']

    # plt.figure(figsize=(10,6))
    # plt.hist(discriminant_scores, bins=75, weights=Lumi_weights, alpha=0.5, color = 'blue', edgecolor  = 'black')
    # plt.title('Even DNN Discriminant Scores on Even Validation Set Weighted by Lumi Weight')
    # plt.xlabel('Discriminant Score (p(+) - p(-))')
    # plt.ylabel('Weighted Event Count')
    # plt.xlim(-1,1)
    # plt.grid(False)
    # plt.savefig('../plots/DNN_Discriminant_vs_LumiWeight_10302025_even_val.png')
    # plt.show()

    loss = np.load('../data/10312025_even_loss.npz')
    losses = loss['losses']
    epochs = loss['epochs']

    plt.figure(figsize=(10,6))
    plt.plot(epochs,losses)
    plt.title('Loss over num_epochs for Model training on even')
    plt.savefig('../plots/Even_Model_Losses.png')
    plt.show()

    train = np.load('../data/dnn_discriminant_scores_and_lumi_weights_10302025_even_train.npz')
    discriminant_scores_train = train['discriminant_scores']
    Lumi_weights_train = train['Lumi_weights']

    plt.figure(figsize=(10,6))
    plt.hist(discriminant_scores_train, bins=75, weights=Lumi_weights_train, alpha=0.5, color = 'blue', edgecolor  = 'black')
    plt.title('Even DNN Discriminant Scores on Even Train Set Weighted by Lumi Weight')
    plt.xlabel('Discriminant Score (p(+) - p(-))')
    plt.ylabel('Weighted Event Count')
    plt.xlim(-1,1)
    plt.grid(False)
    plt.savefig('../plots/DNN_Discriminant_vs_LumiWeight_10302025_even_train.png')
    plt.show()

    test = np.load('../data/dnn_discriminant_scores_and_lumi_weights_10302025_eventest.npz')
    discriminant_scores_test = test['discriminant_scores']
    Lumi_weights_test = test['Lumi_weights']

    plt.figure(figsize=(10,6))
    plt.hist(discriminant_scores_test, bins=75, weights=Lumi_weights_test, alpha=0.5, color = 'blue', edgecolor  = 'black')
    plt.title('Even DNN Discriminant Scores on Odd Dataset Weighted by Lumi Weight')
    plt.xlabel('Discriminant Score (p(+) - p(-))')
    plt.ylabel('Weighted Event Count')
    plt.xlim(-1,1)
    plt.grid(False)
    plt.savefig('../plots/DNN_Discriminant_vs_LumiWeight_10302025_even.png')
    plt.show()

    discriminant_scores = np.concatenate((discriminant_scores_train,discriminant_scores_test))
    Lumi_weights = np.concatenate((Lumi_weights_train,Lumi_weights_test))

    plt.figure(figsize=(10,6))
    plt.hist(discriminant_scores, bins=75, weights=Lumi_weights, alpha=0.5, color = 'blue', edgecolor  = 'black')
    plt.title('Even DNN Discriminant Scores on Full Dataset Weighted by Lumi Weight')
    plt.xlabel('Discriminant Score (p(+) - p(-))')
    plt.ylabel('Weighted Event Count')
    plt.xlim(-1,1)
    plt.grid(False)
    plt.savefig('../plots/DNN_Discriminant_vs_LumiWeight_10302025_even_full.png')
    plt.show()

    data = np.load('../data/dnn_discriminant_scores_and_lumi_weights_10302025_even_quad.npz')
    discriminant_scores = data['discriminant_scores']
    Lumi_weights = data['Lumi_weights']

    plt.figure(figsize=(10,6))
    plt.hist(discriminant_scores, bins=75, weights=Lumi_weights, alpha=0.5, color = 'blue', edgecolor  = 'black')
    plt.title('Even DNN Discriminant Scores on Quadratic Term Set Weighted by Lumi Weight')
    plt.xlabel('Discriminant Score (p(+) - p(-))')
    plt.ylabel('Weighted Event Count')
    plt.xlim(-1,1)
    plt.grid(False)
    plt.savefig('../plots/DNN_Discriminant_vs_LumiWeight_10302025_even_quad.png')
    plt.show()


if __name__ == '__main__':
    main()