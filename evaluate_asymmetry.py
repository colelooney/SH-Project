import numpy as np
import matplotlib.pyplot as plt

data_odd = np.load('results/18_feature_odd_noweightdecay.npz')
discriminant_scores_odd = data_odd['discriminant_scores']
Lumi_weights_odd = data_odd['Lumi_weights']

data_even = np.load('results/20_feature_noweightdecay.npz')
discriminant_scores_even = data_even['discriminant_scores']
Lumi_weights_even = data_even['Lumi_weights']

discriminant_scores = np.concatenate((discriminant_scores_odd,discriminant_scores_even))
lumi_weights = np.concatenate((Lumi_weights_odd,Lumi_weights_even))

norm = np.sum(np.abs(lumi_weights))

D = discriminant_scores
w = lumi_weights.squeeze()   # signed lumi weight

hist, edges = np.histogram(discriminant_scores,
                           bins=np.linspace(-1, 1, 80),
                           weights=lumi_weights/norm,
                           density=False)

bin_centers = 0.5 * (edges[1:] + edges[:-1])
bin_widths  = np.diff(edges)

# masks for y>0 and y<0 (i.e. peak vs trough)
pos_mask = hist > 0
neg_mask = hist < 0

# integrate area of positive part (peak)
area_pos = np.sum(hist[pos_mask] * bin_widths[pos_mask])

# integrate area of negative part (trough)
area_neg = np.sum(hist[neg_mask] * bin_widths[neg_mask])

# your desired "peak minus trough" quantity
asymmetry = area_pos - abs(area_neg)

print("Area (positive part) =", area_pos)
print("Area (negative part) =", area_neg)
print("Asymmetry (peak - |trough|) =", asymmetry)