# 驗證紀錄

日期：2026-11-15。此紀錄區分本次開發環境的實際執行結果與需要 ROS 主機才能執行的檢查，不代表實機部署驗收。

## 已執行並通過

環境：Debian 13 / x86_64、Rust 1.100.0、原生 OpenCV 4.10.0、CPU DNN backend。兩個 NanoTrack v2 ONNX 檔案均經下載腳本中的 SHA-256 驗證。

| 檢查 | 結果 |
|---|---|
| `cargo fmt --check` | 通過 |
| `cargo clippy --locked --all-targets --features ros2-doc -- -D warnings` | 通過，包括 ROS adapter、一次性發布工具與 ROS 整合測試的 Rust 型別檢查 |
| `cargo test --locked --no-default-features` | 5 個純 Rust 單元測試通過 |
| `cargo test --locked` | 6 個單元測試通過；需要模型的 3 個測試預設略過，另行顯式執行如下 |
| `cargo test --locked --test hybrid -- --ignored --nocapture` | 3 個真正使用模型／OpenCV 的整合測試通過 |
| `bash -n scripts/build_ros2.sh scripts/download_models.sh` | shell 語法檢查通過 |
| `bash scripts/download_models.sh` | 已存在模型的校驗／重用流程通過；本次也實際下載並使用兩個 ONNX 檔案 |

### 單元測試涵蓋

- 有效目標替換、無效替換保留舊目標與版本。
- bbox 畫面裁切與 NaN／無效座標拒絕。
- ROS 無效結果的 `found=false`、零 bbox、JSON null，以及發布欄位與 `.msg` schema 一致。
- ROS RGB/BGR 編碼轉換、每行 padding、資料不足與不支援的深度編碼拒絕。
- NanoTrack decoder 的非有限值拒絕與前景 softmax 方向。

### 原生整合測試涵蓋

1. **真正 ONNX forward**：模板 127×127、搜尋 255×255、head 輸出 shape 驗證、框解碼。在合成同一影格測試中，原始前景分數約 1.0、bbox IoU 約 0.973。這只驗證這個測例，不是實際場景準確率。
2. **混合追蹤流程**：沒有目標時 waiting_target；一次接受目標後，後續不重傳仍保留；SIFT 全圖定位後由模型連續追蹤；更換背景；空白／模糊時不輸出舊框；大幅移位後重新定位；拒絕無效新截圖；成功替換時版本增加。
3. **影片 CLI 端到端**：以 Rust/OpenCV 寫入六幀 MJPEG 影片和目標圖，啟動真正 Rust 執行檔，驗證每幀 JSON、原始解析度 bbox、模糊無框、重新定位，以及 stdout 與 JSONL 檔案一致。推論寬度設為 640、來源寬度 800，檢查座標有正確轉回來源解析度。

## 尚未執行

- 本環境未安裝 ROS 2：未執行 `colcon build`、真實 DDS graph、ROS 影像／壓縮影像傳輸、transient-local 晚加入者、ROS 時戳及 source timeout 的執行測試。
- 已附 `tests/ros_transport.rs`，在正常 source ROS 及訊息 overlay 後可顯式執行，驗證一次性目標、持續 bbox、目標切換、header 保留與串流停止後的無效結果。`ros2-doc` 型別檢查無法代替這項測試。
- Ubuntu 22.04/Humble 的 OpenCV 4.5.4 與 Ubuntu 24.04/Jazzy 的套件組合尚未在本環境實際建置；README 提供對應步驟，不能把設計相容性當成已驗證相容性。
- 未用使用者的真實影片、相機、移動底盤或機械手臂測試。
- 未建立實景 FPS、端到端延遲、誤追率、遮擋恢復率、轉面或模糊 benchmark。

## ROS 主機驗收步驟

```bash
source /opt/ros/humble/setup.bash  # 或 jazzy
bash scripts/build_ros2.sh
source install/interfaces/setup.bash
export IDL_PACKAGE_FILTER='builtin_interfaces;std_msgs;sensor_msgs;geometry_msgs;unique_identifier_msgs;action_msgs;mobile_tracker_interfaces'
cargo test --locked --features ros2 --test ros_transport -- --ignored --nocapture
```

實際部署前，再以自己的 topic 測試：啟動後只送一次目標、持續追蹤、換另一張目標、無效圖片、遮擋、底盤移動、轉面、模糊與串流停止。下游應在沒有有效新觀測時停止使用舊框。
