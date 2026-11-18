# 小截圖的前景搜尋與追蹤比較

以每次換目標只給一張 40×40 或 90×90 正面截圖的條件測試 Rust 影片執行檔。A 到 C 組用來比較版本，固定選擇後才跑 D 組。衡量標準為框 IoU ≥ 0.5。F1 同時處罰漏框及錯框，計算方式是「2 × 正確框數 ÷（輸出框數＋目標可見幀數）」。同一序列的影格彼此相關。

使用 COIL-100 的實拍轉面照片，物件和干擾物合成到平移的照片背景。每個物件測試四次，涵蓋兩種截圖尺寸及物體長邊佔圖寬 95%、60%。串流為 320×240 MJPEG。每條序列以 0、5、15、30、60、90、135、180 度移動，再消失，接著以背面及正面重現。所有組別使用相同的干擾物和同一張背景照片的不同區域。未測多個相同實例、實際攝影機雜訊或 ROS 收發延遲。

| 組 | 物件 ID | 場景 |
|---|---|---|
| A | 1、5、20 | 重現前次測試。串流物體長邊 70 px，暫時消失 1 幀 |
| B | 9、32、57、81、96 | 反轉背景、改變截圖背景與水平位置、改變移動方向，尺度由 54 px 逐漸增加 |
| C | 13、26、48、72、88 | B 的場景加上亮度變化，三幀套用 7 px 水平運動模糊，消失 3 幀後移到另一位置 |
| D | 4、17、36、52、69、99 | 與 C 相同的壓力條件，固定版本後更換全部物件 |

## 選擇版本

| 方法（腳本參數） | A F1 | B F1 | C F1 | 三組等權平均 |
|---|---:|---:|---:|---:|
| 原版 SIFT（`original`） | 0.376 | 0.120 | 0.160 | 0.219 |
| 放大後 SIFT（`upscale`） | 0.440 | — | — | — |
| GrabCut 裁圖後 SIFT（`grabcut`） | 0.261 | — | — | — |
| GrabCut 裁圖放大後 SIFT（`grabcut_upscale`） | 0.657 | 0.446 | — | — |
| 遮罩模板（`integrated`） | 0.884 | 0.550 | 0.357 | 0.597 |
| 遮罩模板＋外觀門檻 42（`guarded`） | 0.835 | 0.421 | 0.438 | 0.565 |
| 遮罩模板＋外觀門檻 60（`guarded60`） | 0.928 | 0.517 | 0.409 | 0.618 |
| 混合搜尋＋外觀門檻 42（`hybrid42`） | 0.852 | 0.458 | 0.512 | 0.607 |
| 混合搜尋＋外觀門檻 60（`hybrid`） | 0.942 | 0.551 | 0.476 | 0.656 |

混合搜尋先做 GrabCut 遮罩多尺度彩色模板搜尋，找不到時才以放大裁圖的 SIFT＋MAGSAC 再定位。所有版本以 NanoTrack 追蹤。外觀門檻表示相鄰追蹤框中央區域重採樣至 8×8 後每通道的平均絕對色差，超過門檻便丟棄預測並重新搜尋。A、B、C 等權平均的 F1 以混合搜尋、門檻 60 最高。C 組單獨以門檻 42 較高。

## 固定版本後驗證

每格有 60 個可見影格、18 個無目標影格、6 次首次定位與 6 次背面重現。背面重現統計的是該幀框對的次數，包含持續追蹤和重新搜尋。逐幀狀態可區分兩者。

| 截圖 | 長邊佔比 | 原版正確 | 新版正確 | 新版首次框對 | 新版背面重現框對 | 新版無目標誤報 |
|---|---:|---:|---:|---:|---:|---:|
| 40×40 | 95% | 7/60 | 16/60 | 2/6 | 1/6 | 0/18 |
| 40×40 | 60% | 0/60 | 23/60 | 3/6 | 1/6 | 0/18 |
| 90×90 | 95% | 9/60 | 59/60 | 6/6 | 5/6 | 0/18 |
| 90×90 | 60% | 0/60 | 50/60 | 6/6 | 0/6 | 6/18 |

D 組新版 F1 為 0.700，舊版為 0.115。新版可見時共 148 個正確框、29 個位置錯誤、63 次漏框。無目標的 72 幀中誤報 6 次。CPU（AMD EPYC 9V45，debug build，OpenCV 設定兩個執行緒）上，D 組新版輸出 tracking_model 的影格處理時間中位數 5.4 ms、P95 9.1 ms；tracking_verified 的中位數 22.9 ms、P95 35.4 ms。lost 或 ambiguous 的 P95 為 46.6 ms。時間為執行檔的 processing_ms，未含影像傳輸、解碼、換目標前處理及發布結果。

## 重現

從專案根目錄建置。Python 需要 OpenCV 和 NumPy。COIL-100 下載位置與雜湊見[資料集說明](small-targets.md)。每個命令的標準輸出為逐幀 CSV。D 組的 auto 使用預設 CLI 路徑。

```bash
cargo build --locked
bash scripts/download_models.sh
bash scripts/download_test_assets.sh
export OPENCV_FOR_THREADS_NUM=2
python scripts/compare_reference_crops.py /tmp/coil.zip --methods original upscale grabcut grabcut_upscale integrated guarded guarded60 hybrid42 hybrid
python scripts/compare_reference_crops.py /tmp/coil.zip --holdout --objects 9 32 57 81 96 --methods original grabcut_upscale integrated guarded guarded60 hybrid42 hybrid
python scripts/compare_reference_crops.py /tmp/coil.zip --holdout --stress --objects 13 26 48 72 88 --methods original integrated guarded guarded60 hybrid42 hybrid
python scripts/compare_reference_crops.py /tmp/coil.zip --holdout --stress --objects 4 17 36 52 69 99 --methods original auto
```

[逐幀結果與處理時間](foreground-search.csv)
