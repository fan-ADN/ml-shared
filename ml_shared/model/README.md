fastFM (`https://github.com/ibayer/fastFM`) の scikit-learn 風インターフェイスのラッパ. scikit-kearn API を提供すると言っているが微妙に違うので修正.
例えば:
* `y` を [1, -1] しか受け付けない
* `.predixct_proba()` が1列しか返さない

## FastFMClassifier
`fasFM.asl.FMClassification` に対応.
引数は全く同じ.

## FastFMSgdClassifier
`fastFM.sgd.FMClassification` に対応
引数は全く同じ.

## WeightMixtureClassifier

weight mixing 用クラス. 現在の実装では2値分類にしか対応していないし, 分散処理によるパフォーマンス向上も見込めない