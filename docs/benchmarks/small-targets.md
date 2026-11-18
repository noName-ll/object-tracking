# 小截圖、背景與轉面測試

資料來自 Columbia COIL-100 的物件 1、5、20，使用各自真實拍攝的角度，並以亮度閾值擷取非黑背景區域。三個目標搭配相同干擾物（物件 2）。每次只給一張正面截圖，長邊佔截圖 95% 或 60%，不提供 ROI。這是長度比例，實際背景面積隨物件形狀而變。截圖背景、影片背景取自不同影像區域。

影片為 320×240，物體在串流中的長邊為 70 像素，位置隨幀移動、背景平移。每條序列依序是 0、5、15、30、60、90、135、180 度，消失一幀，再以 180 度、0 度出現。11 個影格以 MJPEG 24 FPS 編碼。真值來自合成位置與各角度的物體外接矩形。每個尺寸與背景組合有 30 個目標可見影格、3 個無目標影格，角度和時間順序相關，這些影格不是獨立統計樣本。

Rust 欄走現有執行檔的預設 SIFT + NanoTrack 路徑，包含初始化失敗。AI 欄是 SuperPoint + LightGlue 全圖匹配加既有 Python 原型的幾何驗證，每幀獨立定位；沒有接 NanoTrack。兩欄各自回答「現有產品能否持續出框」與「AI 搜尋器能否憑這張截圖定位」。位置正確定義為預測框與物體真值框 IoU ≥ 0.5。匹配到截圖背景或框太大都會算錯。

| 方法 | 截圖 | 物體長邊佔比 | 正確框 | 有框但 IoU 不足 | 漏框 | 無目標時誤報 |
|---|---:|---:|---:|---:|---:|---:|
| rust | 40×40 | 95% | 0/30 | 0 | 30 | 0/3 |
| rust | 40×40 | 60% | 0/30 | 0 | 30 | 0/3 |
| rust | 90×90 | 95% | 28/30 | 0 | 2 | 1/3 |
| rust | 90×90 | 60% | 0/30 | 0 | 30 | 0/3 |
| lightglue | 40×40 | 95% | 2/30 | 0 | 28 | 0/3 |
| lightglue | 40×40 | 60% | 0/30 | 4 | 26 | 0/3 |
| lightglue | 90×90 | 95% | 2/30 | 1 | 27 | 0/3 |
| lightglue | 90×90 | 60% | 0/30 | 1 | 29 | 0/3 |

| 角度 | Rust 正確框 | AI 全圖搜尋正確框 |
|---:|---:|---:|
| 0° | 6/24 | 0/24 |
| 5° | 3/12 | 0/12 |
| 15° | 3/12 | 0/12 |
| 30° | 3/12 | 1/12 |
| 60° | 3/12 | 0/12 |
| 90° | 3/12 | 1/12 |
| 135° | 3/12 | 0/12 |
| 180° | 4/24 | 2/24 |

## 重現

在專案根目錄建置 debug 執行檔，下載 NanoTrack 權重及測試背景，安裝 PyTorch、OpenCV、NumPy 以及 LightGlue。測試使用 CPU PyTorch 2.15.0、LightGlue commit 5a650c444718caffc2669dc0094cde908f42dd73，兩個 Torch CPU threads。本環境無 CUDA。腳本可在 CUDA 主機上執行；會重用已計算的幀特徵以測準確度，不宜用此腳本的總耗時換算串流 FPS。

```bash
cargo build --locked
bash scripts/download_models.sh
bash scripts/download_test_assets.sh
pip install 'git+https://github.com/cvg/LightGlue.git@5a650c444718caffc2669dc0094cde908f42dd73'
curl -fL https://www.cs.columbia.edu/CAVE/databases/SLAM_coil-20_coil-100/coil-100/coil-100.zip -o /tmp/coil.zip
python scripts/validate_small_targets.py /tmp/coil.zip > results.csv
```

COIL-100 ZIP SHA-256 為 `fcb4b8b01c0778f416d8d7a09f45082c4f7075cc74743dcddd5c44307bd4afe7`。本測例使用合成動作、固定串流目標尺度與三個物體，未測真實運動模糊、鏡頭曝光、DDS 或底盤端到端延遲。

[逐幀紀錄](small-targets.csv) · [COIL-100 資料來源](https://www.cs.columbia.edu/CAVE/software/softlib/coil-100.php)
