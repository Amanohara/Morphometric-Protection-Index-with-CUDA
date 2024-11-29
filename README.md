#  Morphometric Protection Index with CUDA

## Abstract for EN/ES
This program is a Python wrapper for the Morphometric Protection Index previously implemented in SAGA GIS and QGIS. It supports CUDA, enabling potential performance improvements through acceleration. The program is released under the GPL v3 license.

Este programa es un envoltorio en Python para el índice de protección morfométrico previamente implementado en SAGA GIS y QGIS. Es compatible con CUDA, lo que permite mejorar el rendimiento gracias a la aceleración. El programa se publica bajo la licencia GPL v3.

### Acknowledgments
Special thanks and heartfelt appreciation to [Victor Olaya](https://x.com/volayaf) for developing the original `C++` program that serves as the foundation for this implementation.

### Agradecimientos
Nuestro más sincero agradecimiento y reconocimiento a [Victor Olaya](https://x.com/volayaf)  por desarrollar el programa original en `C++` que sirve como base para esta implementación.


## このプログラムは？
**SAGA GIS**のta_morphometry内にある、**Morphometric Protection Index**のPythonラッパーです。<br>
**なんとCUDAで動きます**。ここ重要。

逆にCUDAを使うことが前提ですので、<u>CUDAが動かせない環境では現時点は動きません。</u>

**GPL v3**で配布します。

## 動作環境
+ **Python=3.11, CUDA 12.4**で動作を確認しています。
+ `PyTorch`, `rasterio`,`numba`,`tqdm`が必要です。 

## 動作方法
`python3 index.py <input_GeoTiff> <output_GeoTiff> --radius <RADIUS>`

[詳しくはQGIS2のマニュアルへ](https://docs.qgis.org/2.18/en/docs/user_manual/processing_algs/saga/terrain_analysis_morphometry.html#morphometric-protection-index)

## 注意事項
**GeoTiffは座標系**でなければなりません。測地系はダメです。<br>日本のDEMを用いる場合は**平面直角座標系**にするのがよいです。

出力したGeoTiffの座標系は入力したCRSを受け継ぎます。