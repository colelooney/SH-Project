"""
plot_metrics.py
Plot ROC curves and confusion matrices for DNN evaluation in academic quality.
"""

import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from matplotlib.ticker import MaxNLocator
import h5py
import pandas as pd

plt.rcParams.update({
    'font.size': 13,
    'axes.labelsize': 13,
    'axes.titlesize': 14,
    'legend.fontsize': 12,
    'axes.linewidth': 1.2,
    'xtick.direction': 'in',
    'ytick.direction': 'in',
    'xtick.top': True,
    'ytick.right': True,
})

def plot_roc(npz_path, save_path):
    data = np.load(npz_path)
    fpr = data['fpr']
    tpr = data['tpr']
    auc = data['roc_auc']

    plt.figure(figsize=(7, 6))
    plt.plot(fpr, tpr, color='blue', lw=2, label=f'DNN (AUC = {auc:.3f})')
    plt.plot([0, 1], [0, 1], color='gray', lw=1.2, linestyle='--', label='Random Guess')

    plt.xlabel('False Positive Rate')
    plt.ylabel('True Positive Rate')
    plt.title('Receiver Operating Characteristic (ROC) Curve')
    plt.legend(frameon=False, loc='lower right')
    plt.grid(False)
    plt.tight_layout()
    plt.savefig(save_path, dpi=300, bbox_inches='tight')
    plt.close()

def plot_confusion_matrix(npz_path, save_path):
    data = np.load(npz_path)
    cm = data['confusion_matrix']
    auc = data['roc_auc']

    # Normalize to show percentages
    cm_norm = cm.astype('float') / cm.sum(axis=1)[:, np.newaxis]
    labels = ['Destructive (0)', 'Constructive (1)']

    plt.figure(figsize=(6, 5))
    sns.heatmap(cm_norm, annot=True, fmt=".2f", cmap='Blues', cbar=False,
                xticklabels=labels, yticklabels=labels, square=True)
    plt.xlabel('Predicted Label')
    plt.ylabel('True Label')
    plt.title(f'Confusion Matrix (AUC = {auc:.3f})')
    plt.tight_layout()
    plt.savefig(save_path, dpi=300, bbox_inches='tight')
    plt.close()

def plot_histogram():
    # Create histogram
    #Hardcoded paths for now
    # data_odd = np.load('results/18_feature_odd_noweightdecay.npz')
    # discriminant_scores_odd = data_odd['discriminant_scores']
    # Lumi_weights_odd = data_odd['Lumi_weights']

    data_even = np.load('results/gnn_noweightdecay.npz')
    discriminant_scores = data_even['discriminant_scores']
    Lumi_weights = data_even['Lumi_weights']

    # discriminant_scores = np.concatenate((discriminant_scores_odd,discriminant_scores_even))
    # Lumi_weights = np.concatenate((Lumi_weights_odd,Lumi_weights_even))
    
    norm = np.sum(np.abs(Lumi_weights)) 
    # norm = 1

    plt.figure(figsize=(10,6))
    plt.hist(discriminant_scores, bins=75, weights=Lumi_weights/norm, alpha=0.5, color = 'blue', edgecolor  = 'black',histtype='step')
    plt.title('GNN Discriminant Scores on Linear Interference Term DataSet Weighted by Lumi Weight')
    plt.xlabel('Discriminant Score (p(+) - p(-))')
    plt.ylabel('Event Fraction')
    plt.xlim(-1,1)
    plt.grid(False)
    plt.savefig('plots/GNN_Discriminant_Histogram_Final.png')
    # plt.show()

    # quad_odd = np.load('results/quad_odd_final_nwd.npz')
    # discriminant_scores_odd = quad_odd['discriminant_scores']
    # Lumi_weights_odd = quad_odd['Lumi_weights']

    # quad_even = np.load('results/quad_even_final_nwd.npz')
    # discriminant_scores_even = quad_even['discriminant_scores']
    # Lumi_weights_even = quad_even['Lumi_weights']

    # discriminant_scores = np.concatenate((discriminant_scores_odd,discriminant_scores_even))
    # Lumi_weights = np.concatenate((Lumi_weights_odd,Lumi_weights_even))

    # norm = np.sum(np.abs(Lumi_weights))

    # plt.figure(figsize=(10,6))
    # plt.hist(discriminant_scores, bins=75, weights=Lumi_weights/norm, alpha=0.5, color = 'blue', edgecolor  = 'black',histtype='step')
    # plt.title('DNN Discriminant Scores on Quadratic Term DataSet Weighted by Lumi Weight')
    # plt.xlabel('Discriminant Score (p(+) - p(-))')
    # plt.ylabel('Event Fraction')
    # plt.xlim(-1,1)
    # plt.grid(False)
    # plt.savefig('plots/DNN_Discriminant_vs_LumiWeight_final_quadratic_noweightdecay.png')

def plot_feature_histograms():
    with h5py.File('data/s2286706/new_Input_CP_Studies_llqq_LinearTerm_20th_October2025.h5') as f:
        df = pd.DataFrame(f['LargeRJet']['1d'][:])

    with h5py.File('data/s2286706/new_Input_CP_Studies_llqq_QuadraticTerm_20th_October2025.h5') as f:
        df_quad = pd.DataFrame(f['LargeRJet']['1d'][:])

    # features_to_drop = ['EventNumber', 'FJ_flavour','Type','NegLep_Eta',
    #                 'Lep_pT_balance','FJ_mass','FJ_E','FJ_phi','LeadingSubJet_Phi',
    #                 'NegLep_E','PosLep_Eta','SubLeadingSubJet_pT','Vlep_E','Vlep_mass',
    #                 'Vlep_phi','cosThetaStar','costheta1']
    
    features_to_include = ['Lumi_weight','LeadingSubJet_E', 'SubLeadingSubJet_E', 'PosLep_Phi', 'NegLep_Phi', 'LeadingSubJet_Eta', 'NegLep_pT', 'PosLep_E', 'SubLeadingSubJet_Eta', 'FJ_pT', 'LeadingSubJet_pT', 'costheta2', 'PosLep_pT', 'Vlep_pT', 'Phi', 'FJ_eta', 'Phi1', 'SubLeadingSubJet_Phi', 'Vlep_eta']
    features_to_drop = [col for col in df.columns if col not in features_to_include]
    df = df.drop(columns = features_to_drop) #get final testing dataset
    df_quad = df_quad.drop(columns=features_to_drop)

    print(df.columns)

    df_constructive = df[df['Lumi_weight'] > 0]
    df_destructive = df[df['Lumi_weight'] < 0]

    df_constructive = df_constructive.drop(columns=['Lumi_weight'])
    df_destructive = df_destructive.drop(columns=['Lumi_weight'])
    df = df.drop(columns=['Lumi_weight'])
    df_quad = df_quad.drop(columns=['Lumi_weight'])
    for feature in df.columns:
        plt.figure(figsize=(10,6))
        plt.hist(df_constructive[feature], bins=75, alpha=0.5, color = 'blue', edgecolor  = 'blue',histtype='step',density = True)
        plt.hist(df_destructive[feature], bins=75, alpha=0.5, color = 'red', edgecolor  = 'red',histtype='step',density = True)
        plt.hist(df_quad[feature], bins=75, alpha=0.5, color = 'green', edgecolor  = 'green',histtype='step',density = True)
        plt.title(f'Histogram of Feature: {feature}')
        plt.legend(['Constructive Interference','Destructive Interference','Quadratic Term'])
        plt.xlabel(f'{feature} Value')
        plt.ylabel('Event Fraction')
        plt.grid(False)
        plt.savefig(f'plots/Feature_Histograms/{feature}_histogram.png')
        plt.close()



if __name__ == '__main__':
    # plot_roc('results/gnn_noweightdecay.npz', 'plots/GNN_ROC_Curve_final.png')
    # plot_confusion_matrix('results/gnn_noweightdecay.npz', 'plots/GNN_Confusion_matric_final.png')
    # # plot_roc('results/18_feature_odd_noweightdecay.npz', 'plots/DNN_ROC_Curve_final_odd_18_features_noweightdecay.png') 
    # # plot_confusion_matrix('results/18_feature_odd_noweightdecay.npz', 'plots/DNN_Confusion_Matrix_final_odd_18_features_noweightdecay.png') 
    # plot_histogram()
    plot_feature_histograms()