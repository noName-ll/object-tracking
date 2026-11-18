# Axis-aligned bounding box in ORIGINAL source image pixels.
# Consume only when found=true, and reject stale header timestamps downstream.
std_msgs/Header header
uint64 sequence
uint64 target_generation
# waiting_target | tracking_verified | tracking_model | lost | blurred
# ambiguous | source_timeout | invalid_frame | target_changed | source_ended
string status
bool found
# Top-left origin, x rightward, y downward; zero when found=false.
float32 x
float32 y
float32 width
float32 height
uint32 image_width
uint32 image_height
# Raw NanoTrack foreground softmax; not a calibrated identity probability.
float32 model_score
# YOLOE visual-prompt probability on global-search frames; not identity confidence.
float32 search_score
# SIFT geometric verification metrics; zero on frames without a feature match.
uint32 inliers
float32 inlier_ratio
float64 blur_variance
float64 processing_ms
bool has_source_position
float64 source_position_ms
