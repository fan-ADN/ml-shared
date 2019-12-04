# なにこれ
機械学習関係の便利スクリプト共有用モジュール

# インストール方法

```bash
git clone hogehoge; cd hogehoge
pip install -U sdist/ml-shared-*.zip
```

# 使い方

大まかに分けて, 以下の4つ

* `data`: データの取得関係. ほとんどは TD API のラッパーになりそう
* `feature`:  データの特徴量行列への変換, あるいは特徴量変換
* `model`: 予測モデル関係
* `evaluation`: 予測モデルの事後評価. 損失の計算やグラフ描画など

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