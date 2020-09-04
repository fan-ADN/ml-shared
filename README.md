# なにこれ
機械学習関係の便利スクリプト共有用モジュール... 

scikit-learn 準拠でモジュール化して使いやすくした各種機械学習アルゴリズムの実装集 実際には検証用に作っただけのものもあるため「使いやすい」とは限らない...

# インストール方法

* 特定のリリースをインストール:

適当なバージョンをダウンロードして

```bash
pip install -U dist/ml_shared-<バージョン>.zip
```

* リポジトリの最新版をインストール:

```bash
pip install -U git+https://github.com/fan-ADN/ml-shared.git@master
```

(`pip` で Git 扱う方法は https://pip.pypa.io/en/stable/reference/pip_install/#git を参考に)


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
