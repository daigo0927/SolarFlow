# 全天日射量の時間軸補間
- 全天日射量は時間に応じて変化する．画像処理で用いられるOpticalFlowの技術に基づいて時間軸の補完を試みる．
  - OpticalFlow
  	- 動画はフレームの連続であり，パラパラ漫画と同じ原理である．OpticalFlowではフレームのある特徴点について，ピクセルレベルで次のフレームと比較し画像の動きを捉えることを目的とする．
	- ピクセルの移動を定式化したモデルを解くと，一意に解を求められないことが分かる．これに対処するためにはピクセルの移動になんらかの制約を与える必要がある．本研究においては近くの隣接する場所：ピクセルの日射量は同じような動きをするという仮定をおき，これを制約式とする．これはOpticalFlownにおいてはGunnar Farneback法と呼ばれている．
- main_forall.py usage example

```
python main_forall.py --data_dir /path/to/row_data --date 2016-08-02 2016-12-27 --region_name Tokyo --method bi
```