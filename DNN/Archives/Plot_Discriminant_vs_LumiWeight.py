import numpy as np
import matplotlib.pyplot as plt

# data = np.load('../data/dnn_discriminant_scores_and_lumi_weights_10272025.npz')
# discriminant_scores = data['discriminant_scores']
# Lumi_weights = data['Lumi_weights']

plt.rcParams.update({
    'axes.linewidth': 1.2,
    'font.family': 'DejaVu Sans',
    'font.size': 13,
    'axes.labelsize': 13,
    'axes.titlesize': 14,
    'xtick.direction': 'in',
    'ytick.direction': 'in'
})

def main():
    # Create histogram
    # data_odd = np.load('../data/dnn_discriminant_scores_20251108_odd.npz')
    # discriminant_scores_odd = data_odd['discriminant_scores']
    # Lumi_weights_odd = data_odd['Lumi_weights']

    data_even = np.load('../../data/dnn_discriminant_scores_and_lumi_weights_archive.npz')
    discriminant_scores_even = data_even['discriminant_scores']
    Lumi_weights_even = data_even['Lumi_weights']

    # discriminant_scores = np.concatenate((discriminant_scores_odd,discriminant_scores_even))
    # Lumi_weights = np.concatenate((Lumi_weights_odd,Lumi_weights_even))
    
    norm = np.sum(np.abs(Lumi_weights_even)) 

    plt.figure(figsize=(10,6))
    plt.hist(discriminant_scores_even, bins=75, weights=Lumi_weights_even, alpha=0.5, color = 'blue', edgecolor  = 'black')
    plt.title('Even DNN Discriminant Scores on Odd DataSet Weighted by Lumi Weight')
    plt.xlabel('Discriminant Score (p(+) - p(-))')
    plt.ylabel('Weighted Event Count')
    plt.xlim(-1,1)
    plt.grid(False)
    plt.savefig('../../plots/DNN_Discriminant_vs_LumiWeight_archive.png')
    plt.show()

    data_even = np.load(r"C:\Users\colel\OneDrive\Documents\UoE_UG\Y4\SH_Project\SH-Project\data\dnn_discriminant_scores_and_lumi_weights_archive_quad  .npz")
    discriminant_scores_even = data_even['discriminant_scores']
    Lumi_weights_even = data_even['Lumi_weights']

    # discriminant_scores = np.concatenate((discriminant_scores_odd,discriminant_scores_even))
    # Lumi_weights = np.concatenate((Lumi_weights_odd,Lumi_weights_even))
    
    norm = np.sum(np.abs(Lumi_weights_even)) 

    plt.figure(figsize=(10,6))
    plt.hist(discriminant_scores_even, bins=75, weights=Lumi_weights_even, alpha=0.5, color = 'blue', edgecolor  = 'black')
    plt.title('Even DNN Discriminant Scores on Odd DataSet Weighted by Lumi Weight')
    plt.xlabel('Discriminant Score (p(+) - p(-))')
    plt.ylabel('Weighted Event Count')
    plt.xlim(-1,1)
    plt.grid(False)
    plt.savefig('../../plots/DNN_Discriminant_vs_LumiWeight_archive_quad.png')
    plt.show()


    # plt.figure(figsize=(10,6))
    # plt.hist(discriminant_scores_odd, bins=75, weights=Lumi_weights_odd, alpha=0.5, color = 'blue', edgecolor  = 'black')
    # plt.title('Odd DNN Discriminant Scores on Even DataSet Weighted by Lumi Weight')
    # plt.xlabel('Discriminant Score (p(+) - p(-))')
    # plt.ylabel('Weighted Event Count')
    # plt.xlim(-1,1)
    # plt.grid(False)
    # plt.savefig('../plots/DNN_Discriminant_vs_LumiWeight_11082025_odd.png')
    # plt.show()

    # plt.figure(figsize=(10,6))
    # plt.hist(discriminant_scores, bins=75, weights=Lumi_weights, alpha=0.5, color = 'blue', edgecolor  = 'black')
    # plt.title('DNN Discriminant Scores on Full DataSet Weighted by Lumi Weight')
    # plt.xlabel('Discriminant Score (p(+) - p(-))')
    # plt.ylabel('Weighted Event Count')
    # plt.xlim(-1,1)
    # plt.grid(False)
    # plt.savefig('../plots/DNN_Discriminant_vs_LumiWeight_11072025_full.png')
    # plt.show()

    # loss = np.load('../data/10312025_even_loss.npz')
    # losses = loss['losses']
    # epochs = loss['epochs']

    # plt.figure(figsize=(10,6))
    # plt.plot(epochs,losses)
    # plt.title('Loss over num_epochs for Model training on even')
    # plt.savefig('../plots/Even_Model_Losses.png')
    # plt.show()

    # train = np.load('../data/dnn_discriminant_scores_and_lumi_weights_10302025_even_train.npz')
    # discriminant_scores_train = train['discriminant_scores']
    # Lumi_weights_train = train['Lumi_weights']

    # plt.figure(figsize=(10,6))
    # plt.hist(discriminant_scores_train, bins=75, weights=Lumi_weights_train, alpha=0.5, color = 'blue', edgecolor  = 'black')
    # plt.title('Even DNN Discriminant Scores on Even Train Set Weighted by Lumi Weight')
    # plt.xlabel('Discriminant Score (p(+) - p(-))')
    # plt.ylabel('Weighted Event Count')
    # plt.xlim(-1,1)
    # plt.grid(False)
    # plt.savefig('../plots/DNN_Discriminant_vs_LumiWeight_10302025_even_train.png')
    # plt.show()

    # test = np.load('../data/dnn_discriminant_scores_and_lumi_weights_10302025_eventest.npz')
    # discriminant_scores_test = test['discriminant_scores']
    # Lumi_weights_test = test['Lumi_weights']

    # plt.figure(figsize=(10,6))
    # plt.hist(discriminant_scores_test, bins=75, weights=Lumi_weights_test, alpha=0.5, color = 'blue', edgecolor  = 'black')
    # plt.title('Even DNN Discriminant Scores on Odd Dataset Weighted by Lumi Weight')
    # plt.xlabel('Discriminant Score (p(+) - p(-))')
    # plt.ylabel('Weighted Event Count')
    # plt.xlim(-1,1)
    # plt.grid(False)
    # plt.savefig('../plots/DNN_Discriminant_vs_LumiWeight_10302025_even.png')
    # plt.show()

    # discriminant_scores = np.concatenate((discriminant_scores_train,discriminant_scores_test))
    # Lumi_weights = np.concatenate((Lumi_weights_train,Lumi_weights_test))

    # plt.figure(figsize=(10,6))
    # plt.hist(discriminant_scores, bins=75, weights=Lumi_weights, alpha=0.5, color = 'blue', edgecolor  = 'black')
    # plt.title('Even DNN Discriminant Scores on Full Dataset Weighted by Lumi Weight')
    # plt.xlabel('Discriminant Score (p(+) - p(-))')
    # plt.ylabel('Weighted Event Count')
    # plt.xlim(-1,1)
    # plt.grid(False)
    # plt.savefig('../plots/DNN_Discriminant_vs_LumiWeight_10302025_even_full.png')
    # plt.show()

    # data = np.load('../data/dnn_discriminant_scores_and_lumi_weights_10302025_even_quad.npz')
    # discriminant_scores = data['discriminant_scores']
    # Lumi_weights = data['Lumi_weights']

    # plt.figure(figsize=(10,6))
    # plt.hist(discriminant_scores, bins=75, weights=Lumi_weights, alpha=0.5, color = 'blue', edgecolor  = 'black')
    # plt.title('Even DNN Discriminant Scores on Quadratic Term Set Weighted by Lumi Weight')
    # plt.xlabel('Discriminant Score (p(+) - p(-))')
    # plt.ylabel('Weighted Event Count')
    # plt.xlim(-1,1)
    # plt.grid(False)
    # plt.savefig('../plots/DNN_Discriminant_vs_LumiWeight_10302025_even_quad.png')
    # plt.show()

    # odd_test = np.load('../data/dnn_discriminant_scores_and_lumi_weights_10302025_odd_test.npz')
    # odd_discriminant_scores_test = odd_test['discriminant_scores']
    # odd_Lumi_weights_test = odd_test['Lumi_weights']

    # discriminant_scores = np.concatenate((discriminant_scores_test,odd_discriminant_scores_test))
    # lumi_weights = np.concatenate((Lumi_weights_test,odd_Lumi_weights_test))

    # plt.figure(figsize=(10,6))
    # plt.hist(discriminant_scores, bins=75, weights=lumi_weights, alpha=0.5, color = 'blue', edgecolor  = 'black')
    # plt.title('DNN Discriminant Scores on Full Dataset Weighted by Lumi Weight')
    # plt.xlabel('Discriminant Score (p(+) - p(-))')
    # plt.ylabel('Weighted Event Count')
    # plt.xlim(-1,1)
    # plt.grid(False)
    # plt.savefig('../plots/DNN_Discriminant_vs_LumiWeight_11032025_full.png')
    # plt.show()


if __name__ == '__main__':
    main()