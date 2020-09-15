## Model module

scikit-learn の estimator 準拠, もしくは修正版クラス

### FastFMClassifier

`fasFM.asl.FMClassification` に対応.

fastFM (`https://github.com/ibayer/fastFM`) の scikit-learn 風インターフェイスのラッパ. 本家でも scikit-kearn API を提供すると言っているが微妙に違うので修正.
例えば:
* `y` を [1, -1] しか受け付けない
* `.predixct_proba()` が1列しか返さない

引数は全く同じ.

### FastFMSgdClassifier

同上.

`fastFM.sgd.FMClassification` に対応
引数は全く同じ.

### WeightMixtureClassifier

weight mixing 用クラス. 現在の実装では2値分類にしか対応していないし, 分散処理によるパフォーマンス向上も見込めない, 動作確認用.

参考: Mann, G. S., McDonald, R., Mohri, M., Silberman, N., & Walker, D. (2009). "_Efficient large-scale distributed training of conditional maximum entropy models_," Advances in neural information processing systems (NIPS) 22 (pp. 1231–1239). available at [here](https://papers.nips.cc/paper/3881-efficient-large-scale-distributed-training-of-conditional-maximum-entropy-models)

### ImblernRecalibrator

不均衡リサンプリングの事後補正. ただし分類以外に役に立つのかは自明でない

* Kitazawa, T. 『[Over-/Under-samplingをして学習した2クラス分類器の予測確率を調整する式](https://takuti.me/note/adjusting-for-oversampling-and-undersampling/)』, 2017
* He, X. et al., “Practical Lessons from Predicting Clicks on Ads at Facebook,” in Proceedings of 20th ACM SIGKDD Conference on Knowledge Discovery and Data Mining - ADKDD’14, 2014, pp. 1–9. DOI: [10.1145/2648584.2648589](https://doi.org/10.1145/2648584.2648589)
* Dal Pozzolo, A., O. Caelen, and G. Bontempi, “When is Undersampling Effective in Unbalanced Classification Tasks?,” in Proceedings of the 2015th European Conference on Machine Learning and Knowledge Discovery in Databases, Porto, Portugal, 2015, vol. 9284, pp. 200–215. DOI: [10.1007/978-3-319-23528-8_13](https://doi.org/10.1007/978-3-319-23528-8_13)
