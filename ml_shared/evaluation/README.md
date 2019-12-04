# Utilities for model evaluation

モデル評価に使える便利関数. 主にカリブレーション関係:

## Usage

```
from ml-shared.evaluation import *
```

## Featured Functions


### calibrate_imbalanceness
不均衡データで分類モデルを扱う際のバランス調整用
calibrate probability estimates for imbalanced data, suggested by the followings:

* Yui, T. 『[Over-/Under-samplingをして学習した2クラス分類器の予測確率を調整する式](https://takuti.me/note/adjusting-for-oversampling-and-undersampling/)』, 2017
* He, X. et al., “Practical Lessons from Predicting Clicks on Ads at Facebook,” in Proceedings of 20th ACM SIGKDD Conference on Knowledge Discovery and Data Mining - ADKDD’14, 2014, pp. 1–9. DOI: [10.1145/2648584.2648589](https://doi.org/10.1145/2648584.2648589)
* Dal Pozzolo, A., O. Caelen, and G. Bontempi, “When is Undersampling Effective in Unbalanced Classification Tasks?,” in Proceedings of the 2015th European Conference on Machine Learning and Knowledge Discovery in Databases, Porto, Portugal, 2015, vol. 9284, pp. 200–215. DOI: [10.1007/978-3-319-23528-8_13](https://doi.org/10.1007/978-3-319-23528-8_13)


### normalized_entropy 
正規化エントロピー (NE; Normalized cross Entropy) の計算
calculate the normalized (cross) entropy suggested by:

* He, X. et al., “Practical Lessons from Predicting Clicks on Ads at Facebook,” in Proceedings of 20th ACM SIGKDD Conference on Knowledge Discovery and Data Mining - ADKDD’14, 2014, pp. 1–9. DOI: [10.1145/2648584.2648589](https://doi.org/10.1145/2648584.2648589)
* Yi, J. , Y. Chen, J. Li, S. Sett, and T. W. Yan (2013) “_Predictive model performance: offline and online evaluations_,” in Proceedings of the 19th ACM SIGKDD international conference on Knowledge discovery and data mining - KDD ’13, New York, New York, USA, p. 1294. DOi: [10.1145/2487575.2488215](https://doi.org/10.1145/2487575.2488215)


### normalized_log_loss
正規化対数損失の計算
Lefortier, Damien, Anthony Truchet, and Maarten de Rijke. 2015. “Sources of Variability in Large-Scale Machine Learning Systems.” In Machine Learning Systems (NIPS 2015 Workshop). http://learningsys.org/2015/papers.html.


### expected_calibration_error
期待カリブレーション誤差 (ECE) の計算
calculate Hozmer-Lemeshaw-statistics (calibration curve) -based metric, inspired by:

* Guo, Chuan, Geoff Pleiss, Yu Sun, and Kilian Q. Weinberger. (2017) “_On Calibration of Modern Neural Networks_.” In Proceedings of the 34th International Conference on Machine Learning, 70:1321–30. Sydney, NSW, Australia. URL: https://dl.acm.org/citation.cfm?id=3305518, arXiv: [1706.04599](http://arxiv.org/abs/1706.04599).
* Naeini, Mahdi Pakdaman, Gregory F. Cooper, and Milos Hauskrecht. (2015) “_Obtaining Well Calibrated Probabilities Using Bayesian Binning_.” in Proceedings of the AAAI Conference on Artificial Intelligence, 2901–7, PMID: [25927013](https://europepmc.org/abstract/med/25927013)

### integrated_calibration_index
積分カリブレーション指数 (ICI) の計算
calculate integrated calibration index (ICI) suggested by:

* P. C. Austin and E. W. Steyerberg. (2019) “_The Integrated Calibration Index (ICI) and related metrics for quantifying the calibration of logistic regression models_,” Statistics in Medicine. DOI: [10.1002/sim.8281](https://doi.org/10.1002/sim.8281)


### print_metrics
display glavnie metrics for multiple inputs (e.g., train/test data).


### plot_ROC
plot ROC curves for multiple inputs.

### plot_calibration
plot calibration curves for multiple inputs.


### plot_pred_hist
plot histograms of prediction values for multiple inputs.
