# v0.3 YOLOE 驗證紀錄

日期：2026-11-18。全局搜尋已由 SIFT/MAGSAC 改為視覺提示 YOLOE；舊紀錄保留在 [VALIDATION-0.2.md](VALIDATION-0.2.md)，不能用舊版測試推論新搜尋器的效果。

## 實際環境與模型

- Debian 13 x86_64，Rust 1.100.0，原生 OpenCV 4.10.0。
- Rust `ort=2.0.0-rc.10` 動態載入 CPU ONNX Runtime 1.23.2。
- YOLOE-v8s-seg 官方 checkpoint，由本專案 `scripts/export_yoloe.py` 匯出，固定 640×640、單一 visual prompt、512 維 embedding。
- 匯出工具：Ultralytics 8.3.203、PyTorch 2.15.0+cpu、torchvision 0.29.2+cpu、ONNX 1.19.1、ONNX Runtime 1.23.2。
- NanoTrack v2 的兩個 ONNX 均沿用並校驗 SHA-256。
- 自然圖片測例為 Ultralytics `bus.jpg`，由下載腳本校驗 SHA-256，不放進 Git。

## 已實際通過

| 檢查 | 結果 |
|---|---|
| YOLOE 雙模型匯出 | 兩個 ONNX checker 通過；PyTorch/ORT 數值比對通過；替換 embedding 後 detector 分數改變，確認不是固定匯出類別 |
| `cargo fmt --check` | 通過 |
| `cargo clippy --locked --all-targets --features ros2-doc -- -D warnings` | 通過；包含 ROS adapter、目標發布工具、ROS 整合測試的 Rust API 型別檢查 |
| `cargo test --locked --no-default-features` | 5 個純 Rust 測試通過 |
| `cargo test --locked` | 8 個單元測試通過；需要實際模型的 4 個測試預設 ignored，另行顯式執行如下 |
| `cargo test --locked --test hybrid -- --ignored --nocapture` | 4 個實際模型／CLI 整合測試通過 |
| shell 語法、模型／測例校驗 | 下載腳本可重用現有正確檔案；exporter Python 語法檢查通過 |

### 單元測試

- 目標替換成功增加版本；驗證失敗保留舊目標。
- bbox 裁切與 NaN／非法位置拒絕。
- 無效 ROS 輸出為 found=false、零 bbox，且發布欄位與 TrackingResult.msg 一致。
- ROS raw RGB/BGR 轉換、step padding、截斷／不支援編碼拒絕。
- NanoTrack decoder 的非有限值與前景分數方向。
- YOLOE letterbox 還原、參考 mask 排除 padding。
- YOLOE NMS 消除重複框，分數相近的不同位置候選標為 ambiguous；拒絕非法數值。

### 真正模型與 CLI 測試

1. **NanoTrack forward**：真正執行 backbone/head，驗證 shape、框解碼。原合成測例同幀 IoU 約 0.973；不是實景準確率。
2. **混合流程**：一次接受巴士截圖；後續不重傳仍保留 embedding，能完成 YOLOE 初次定位與 NanoTrack 連續追蹤；畫面改變／模糊時不輸出舊框；巴士大幅移位後 YOLOE 全圖重新找到；非法新圖保留版本；有效新圖增加版本。
3. **影片 CLI**：Rust 寫入六幀 MJPEG，啟動真正 Rust 執行檔，驗證每幀 JSON、原始來源寬度 800／推論工作寬度 640 的座標還原、blurred 無框、重新搜尋以及 stdout/JSONL 檔案一致。
4. **更換視覺提示**：同一對已載入模型，使用巴士與人物兩個 crop，產生不同 embedding；不重新匯出即得到不同位置／尺度的候選。這證明動態 visual prompt 接通，不證明能識別同類中的特定實體；人物 prompt 可能匹配畫面中的另一位人物。

自然圖測例的初次 YOLOE 巴士分數約 0.877，重新定位約 0.852。這些是個別測例的模型分數，不是機率校準或 benchmark，也不構成真實底盤運動／遮擋／轉面的準確率結論。

## 尚未驗證

- 無實際 ROS 2 安裝：未執行 colcon、自訂訊息生成、真實 DDS 傳輸、壓縮 topic、transient-local 晚加入者或 ROS source timeout 的執行測試。
- `tests/ros_transport.rs` 已更新使用自然圖提示，且 Rust 型別檢查通過；需在正常 source ROS/interfaces 的主機實際執行。新 msg 增加 search_score，發布端／訂閱端都需重建。
- 未在 Ubuntu 22.04/Humble、Ubuntu 24.04/Jazzy 或 ARM64 真正部署；README 的支援目標不是已測平台聲明。
- 未用使用者的影片／相機／移動底盤測試，未測量實景誤追率、丟失率、遮擋恢復率、轉面效果、端到端延遲或 FPS。
- 未驗證 GPU backend；本版使用 CPU。

## 重現方式

先完成 README 的 NanoTrack 下載、YOLOE 匯出與原生 Runtime 設定：

```bash
bash scripts/download_models.sh
bash scripts/download_test_assets.sh
cargo test --locked
cargo test --locked --test hybrid -- --ignored --nocapture
```

在 ROS 主機上重新建置訊息與 Rust（不能用 ros2-doc）：

```bash
source /opt/ros/humble/setup.bash  # 或 jazzy
bash scripts/build_ros2.sh
source install/interfaces/setup.bash
export IDL_PACKAGE_FILTER='builtin_interfaces;std_msgs;sensor_msgs;geometry_msgs;unique_identifier_msgs;action_msgs;mobile_tracker_interfaces'
# 設定 ORT_DYLIB_PATH 後執行
cargo test --locked --features ros2 --test ros_transport -- --ignored --nocapture
```
