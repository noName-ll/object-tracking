# Rust 即時物件追蹤與 ROS 2 bounding box

相機持續送畫面，另一個 topic 在指定或更換目標時送一張截圖。預設依截圖（或 ROI）的長邊選擇搜尋方式。96 像素以內用 GrabCut 分離前景、多尺度彩色模板定位，必要時用 SIFT＋MAGSAC 再找一次。較大截圖直接用 SIFT＋MAGSAC。兩者都用 NanoTrack 連續追蹤。可追蹤未預先定義類別的物品。特徵模式只需下載 NanoTrack 權重，安裝時可跳過第 2.2、2.3 節。

ROS 影像接收與 RTSP／攝影機讀取獨立於推論。每個串流保留最新待處理影格，推論落後時跳過中間影格。處理期間若收到換目標事件，丟棄舊搜尋結果。全圖搜尋仍會佔用推論時間，輸出頻率取決於影像尺寸、物件與硬體。

## 1. 輸入與輸出

| 項目 | 選項／格式 | 行為 |
|---|---|---|
| 連續影像 | `--video demo.mp4` | 讀取影片，每幀處理；依 FPS 節奏讀取，運算慢時播放也較慢 |
| 連續影像 | `--video camera:0` | OpenCV 攝影機 index |
| 連續影像 | `--video rtsp://...` | 依本機 FFmpeg/GStreamer 支援的串流 |
| 連續影像 | `--image-topic /camera/image_raw` | ROS `sensor_msgs/msg/Image` |
| 連續影像 | 加 `--image-compressed` | 改為 `sensor_msgs/msg/CompressedImage` |
| 目標截圖 | `--target object.jpg` | 啟動時載入，保存目標特徵 |
| 目標截圖 | `--target-topic /tracker/target_image` | 每張有效新圖片作為換目標事件；平時不必重傳 |
| 目標截圖 | 加 `--target-compressed` | 改用 `CompressedImage` |
| ROS 結果 | `/tracker/bbox` | `mobile_tracker_interfaces/msg/TrackingResult` |
| 文字結果 | stdout／`--jsonl results.jsonl` | 一行一筆 JSON；診斷寫 stderr |

影片／ROS 串流 × 圖片／ROS 截圖四種組合皆可。連續影像來源二選一；`--target` 與 `--target-topic` 可以同時指定，先用檔案初始化再接收換目標事件。影片模式加 `--ros` 也可發布 ROS bbox。

**截圖中有背景時**，小圖模式會把一圈邊界當成背景，估計前景遮罩。背景和物體顏色接近或同一截圖有好幾個物體時可能選錯。若可指定目標範圍，可使用 `--target-roi X,Y,W,H`。ROI 位於截圖原始座標，不是串流座標，會套用到每張新截圖。YOLOE prompt mask 排除 letterbox padding，但不能排除裁圖內的背景。

## 2. 安裝與模型準備

目標平台為 Ubuntu 22.04 / ROS 2 Humble、Ubuntu 24.04 / ROS 2 Jazzy。單純影片模式不需要 ROS。設計相容性與實際測試平台的差異見驗證紀錄。

需求：Rust stable **1.89+**、OpenCV **4.5.4+**（含 DNN）、libclang、C++ 編譯工具。預設搜尋使用 OpenCV，連續追蹤使用 NanoTrack CPU 推論。YOLOE 模式另需原生 ONNX Runtime。

```bash
sudo apt update
sudo apt install -y build-essential cmake pkg-config clang libclang-dev libopencv-dev curl git
# 尚未安裝 Rust 時，依 https://rustup.rs 安裝 stable toolchain
rustc --version
cargo --version
pkg-config --modversion opencv4

git clone https://github.com/noName-ll/object-tracking.git
cd object-tracking
```

### 2.1 NanoTrack 模型

```bash
bash scripts/download_models.sh
```

腳本只下載兩個 NanoTrack v2 ONNX，並做 SHA-256 校驗；它不下載／匯出 YOLOE。

| 模型 | SHA-256 |
|---|---|
| `models/nanotrack_backbone_sim.onnx` | `530bdd0cd00f19afab79a863e71ba71e3312395a5dc9151af675082bdaaa2fc4` |
| `models/nanotrack_head_sim.onnx` | `0d8c0637be849f092cc7236cae02e55c8b9455ebe37ba50601d6115db4247cd9` |

### 2.2 選用 YOLOE 時才需要一次性匯出 YOLOE 視覺提示模型

必須使用本專案的兩輸入匯出方式。一般 `yolo export` 產生的固定類別模型，或 prompt-free YOLOE，不能動態接收新目標截圖。

以下可以在開發電腦執行，建議使用 Python 3.10–3.12 的獨立環境：

```bash
python3 -m venv .export-venv
source .export-venv/bin/activate
pip install torch torchvision --index-url https://download.pytorch.org/whl/cpu
pip install -r scripts/requirements-yoloe-export.txt
python scripts/export_yoloe.py --model yoloe-v8s-seg.pt --size 640 --output models/yoloe
```

匯出工具固定 `ultralytics==8.3.203`，會下載官方 checkpoint，並輸出：

- `models/yoloe/encoder.onnx`：截圖＋單一物件提示遮罩 → `[1,1,512]` visual embedding。
- `models/yoloe/detector.onnx`：串流影像＋該 embedding → `[1,N,5]` 候選，欄位為 `cx,cy,w,h,score`。
- `models/yoloe/manifest.json`：匯出尺寸、格式、版本及兩個 ONNX 的 SHA-256。

工具會實際執行 ONNX checker、PyTorch／ONNX Runtime 數值比對，以及「替換 embedding 後分數是否改變」的檢查。這兩個模型**不包含任何使用者目標截圖**；日後換圖只需執行 encoder，不需重新匯出。

模型有固定 batch=1、單一視覺類別、固定影像尺寸。`--yoloe-size` 必須與匯出的 `--size` 一致。輸出只保留 bbox／score，不輸出分割 mask。預設為 YOLOE-v8s-seg 的視覺提示模式，不支援任意 YOLO/YOLOE 檔案直接替換。

### 2.3 選用 YOLOE 時才需要設定原生 ONNX Runtime

Rust 使用 `ort` 動態載入原生 Runtime，透過 `ORT_DYLIB_PATH` 指定**共享函式庫檔案**，不是資料夾。

在上面的匯出電腦，可直接使用已安裝的 CPU wheel 內的原生函式庫：

```bash
export ORT_DYLIB_PATH="$(find "$PWD/.export-venv" -type f -name 'libonnxruntime.so.*' -print -quit)"
test -f "$ORT_DYLIB_PATH"
deactivate
```

這只是使用 wheel 內的原生 `.so`，Rust 執行期不會啟動 Python。正式部署可改用 [ONNX Runtime 官方發行包](https://github.com/microsoft/onnxruntime/releases)中的原生函式庫：

```bash
# 範例：依你的實際安裝位置調整
export ORT_DYLIB_PATH=/opt/onnxruntime/lib/libonnxruntime.so.1.23.2
```

將兩個 ONNX 和 `manifest.json` 複製到機器人的 `models/yoloe/`。原生 Runtime 必須符合機器人的 CPU 架構／作業系統；不要將 x86_64 的 `.so` 複製給 ARM64。必要時將 Runtime 的相依函式庫目錄加入 `LD_LIBRARY_PATH`。新終端也要設定 `ORT_DYLIB_PATH`。

模型與 Runtime 二進位不放進 Git；Ultralytics／YOLOE 的授權與本專案 Apache-2.0 不同，見 [THIRD_PARTY.md](THIRD_PARTY.md)。

### 2.4 編譯 Rust

```bash
cargo build --release --locked
./target/release/mobile-object-tracker --help
```

### 2.5 編譯 ROS 2

先安裝 ROS 2，再 source 單一發行版，不要混用 Humble/Jazzy：

```bash
source /opt/ros/humble/setup.bash  # Jazzy 改成 /opt/ros/jazzy/setup.bash
sudo apt install -y python3-colcon-common-extensions \
  ros-${ROS_DISTRO}-rosidl-default-generators \
  ros-${ROS_DISTRO}-sensor-msgs ros-${ROS_DISTRO}-std-msgs
bash scripts/build_ros2.sh
source install/interfaces/setup.bash
```

`colcon` 是 ROS 建置工具；追蹤程式仍以 Rust 執行。腳本先建置 custom message，再 source overlay，最後 `cargo build --release --locked --features ros2`。

**從 v0.2 升級必須重新建置 custom message 和 Rust**：TrackingResult 增加 `search_score`，請讓發布端與訂閱端使用相同的新 interface。切換 ROS 發行版或 message schema 時，重新建置 interfaces、source overlay，再 `cargo clean` 並重新編譯 Rust。

每個 ROS 執行終端均需：

```bash
source /opt/ros/humble/setup.bash  # 或 jazzy
source install/interfaces/setup.bash
export ROS_DOMAIN_ID=0             # 與其他 node 保持一致
# 選用 YOLOE 時設定 ORT_DYLIB_PATH
```

Rust 執行檔直接執行，不透過 `ros2 run`。不要用 `--all-features` 部署：`ros2-doc` 僅供無 ROS 環境的型別檢查，無法建立可執行的 ROS node。

## 3. 使用方式

### 影片 + 圖片

```bash
./target/release/mobile-object-tracker \
  --video demo.mp4 --target object.jpg \
  --target-roi 120,80,240,300 --jsonl results.jsonl
```

已經裁好的圖片可省略 `--target-roi`。程式無 GUI，不需要桌面。

### ROS 相機 + 圖片

```bash
./target/release/mobile-object-tracker \
  --image-topic /camera/color/image_raw --target cropped_object.jpg
```

### ROS 相機 + 一次性目標 topic

先啟動追蹤器：

```bash
./target/release/mobile-object-tracker \
  --image-topic /camera/color/image_raw --target-topic /tracker/target_image
```

另一個已 source ROS/overlay 的終端，只發一張截圖：

```bash
./target/release/send-target --file cropped_object.jpg
# 需要換目標時才再送一張
./target/release/send-target --file another_object.jpg
```

`send-target` 先等相容 subscriber，最多 10 秒，發送一次 BGR8 後保持 publisher 存活 2 秒。追蹤器收到後將目標特徵留在記憶體，不需要持續送截圖；尚未收到有效目標時輸出 `waiting_target`。

新圖先完成解碼、ROI、尺寸和 特徵檢查，成功後才增加 `target_generation` 並清除舊 NanoTrack 狀態。驗證失敗保留舊特徵／版本。每張有效新圖都視為換目標，即使圖片內容相同。新圖通過 特徵檢查不代表一定能在串流內找到物件。

### 影片 + 一次性目標 topic

```bash
./target/release/mobile-object-tracker \
  --video demo.mp4 --target-topic /tracker/target_image
```

### 影片 + 圖片，也發布 ROS bbox

```bash
./target/release/mobile-object-tracker --video demo.mp4 --target object.jpg --ros
```

### 壓縮 topic

```bash
./target/release/mobile-object-tracker \
  --image-topic /camera/image/compressed --image-compressed \
  --target-topic /tracker/target/compressed --target-compressed
```

直接訂閱 `CompressedImage`，不是 image_transport plugin。`send-target` 只發布 raw `Image`。

### 晚加入的訂閱者

預設目標 subscriber 為 reliable + volatile；應先開追蹤器，再送一次目標圖。若要接收仍存活的 publisher 保存的上一張圖片，兩端均使用 transient local：

```bash
./target/release/mobile-object-tracker \
  --image-topic /camera/image_raw --target-topic /tracker/target_image \
  --target-durability transient-local

./target/release/send-target --file object.jpg --transient-local \
  --wait-seconds 60 --keep-alive-seconds 3600
```

Transient local 不會在 publisher 結束後永久保存圖片。程式重啟也不會保留記憶體中的目標；要再次送圖或提供檔案。


## 追蹤流程與輸出

1. 在截圖中緊貼物件裁切，或提供原始截圖座標的 `--target-roi x,y,w,h`。新截圖驗證成功後才替換舊目標，版本加一。截圖中的背景也可能產生匹配點。
2. 小圖模式先跑四輪 GrabCut，預先快取遮罩與不同尺寸的前景模板。搜尋時縮小串流至半解析度，在各尺度比較遮罩內 RGB 通道的平方差，空間分離且得分接近的候選會回報 ambiguous。模板沒找到時，將截圖前景外框放大至長邊 180 像素取得 SIFT 特徵再全圖定位。NanoTrack 更新時比較前後框中央區域，顏色差異過大會觸發全圖重搜。可用 `--search-backend small-target` 強制啟用，或用 `--search-backend features` 重現舊版。
3. SIFT 路徑在整張工作圖上做 SIFT 雙向最近鄰匹配，ratio test 取 0.75。MAGSAC 估計單應性，需要至少 12 個內點、內點比例至少 40%，並檢查內點覆蓋面積和投影是否在畫面內。
4. 成功後用框初始化 NanoTrack。預設每 15 個處理影格用截圖重新驗證，模型低分時立即重新搜尋。可用 `--verify-every` 與 `--max-width` 調整速度／細節。平滑、反光物件可能缺少特徵，顯著轉面也可能追丟。若兩件物品有完全相同的圖案，幾何比對仍可能認錯。
5. 輸出 `/tracker/bbox`，型別為 `mobile_tracker_interfaces/msg/TrackingResult`。只在 `found=true` 時使用 `x,y,width,height`，原點在原始串流影像左上角。ROS header 取自該影格。`target_generation` 識別目標版本，`tracking_verified` 表示搜尋成功，`tracking_model` 表示當前由 NanoTrack 追蹤。遺失或模糊時輸出無效框。
6. `features` 模式輸出 `inliers` 和 `inlier_ratio`。若要比較 YOLOE，在相同啟動指令加上 `--search-backend yoloe`，這時全圖搜尋使用 visual embedding，輸出 `search_score`。完整選項見 `--help`。

```bash
ros2 topic echo /tracker/bbox
```

連續影像使用 best-effort、volatile、depth 1。目標圖片使用 reliable、depth 1。輸出使用 reliable、volatile、depth 1。raw 影像支援 bgr8、rgb8、mono8、rgba8、bgra8。ROS 接收會覆蓋尚未處理的影格，檔案播放則依順序處理。

測試特徵模式可執行以下指令。第二個測試使用自然照片內指定人物的截圖，檢查傳回框是否落在同一位人物上。

```bash
bash scripts/download_models.sh
bash scripts/download_test_assets.sh
cargo test --locked
cargo test --test hybrid feature_ -- --ignored
```

[先前 YOLOE 版本的技術細節](docs/YOLOE-0.3.md) · [YOLOE 驗證紀錄](docs/VALIDATION.md)

[40×40／90×90 截圖、不同背景與真實轉面角度的逐幀驗證](docs/benchmarks/small-targets.md)

[小截圖搜尋版本比較、遮擋與背面重現測試](docs/benchmarks/foreground-search.md)
