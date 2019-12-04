# なにこれ
機械学習関係の便利スクリプト共有用モジュール

# インストール方法

```bash
git clone hogehoge; cd hogehoge
pip install -U dist/ml_shared-*.zip
```

# 使い方

大まかに分けて, 以下の4つ

* [`data`](ml_shared/evaluation/README.md): データの取得関係. ほとんどは TD API のラッパーになりそう

* [`feature`](ml_shared/evaluation/README.md):  データの特徴量行列への変換, あるいは特徴量変換
* [`model`](ml_shared/model/README.md): 予測モデル関係
* [`evaluation`](ml_shared/evaluation/README.md): 予測モデルの事後評価. 損失の計算やグラフ描画など

```python
from ml_shared.data import *
from ml_shared.feature import *
from ml_shared.model import *
from ml_shared.evalutaion import *
```

# メンテ
新しい `FUGA.py` ファイル追加した場合, 配置ディレクトリ `HOGE` の `__init__.py` に

```python
from HOGE.FUGA import (...)
```

と書いておく.
