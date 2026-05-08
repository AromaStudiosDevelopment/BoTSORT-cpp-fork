#pragma once

#include <string>
#include <variant>

#include "GlobalMotionCompensation.h"
#include "GmcParams.h"
#include "ReID.h"
#include "ReIDParams.h"
#include "TrackerParams.h"
#include "track.h"

template<typename T>
using Config = std::variant<T, std::string, std::monostate>;

class BoTSORT
{
public:
    explicit BoTSORT(const Config<TrackerParams> &tracker_config,
                     const Config<GMC_Params> &gmc_config = {},
                     const Config<ReIDParams> &reid_config = {},
                     const std::string &reid_onnx_model_path = "");

    ~BoTSORT() = default;


    /**
     * @brief Track the objects in the frame
     *
     * @param detections Detections in the frame
     * @param frame Frame
     * @return std::vector<std::shared_ptr<Track>>
     */
    std::vector<std::shared_ptr<Track>>
    track(const std::vector<Detection> &detections, const cv::Mat &frame);


    /**
     * @brief Track the objects in the frame using pre-computed appearance
     *        embeddings supplied by an external Re-ID stage.
     *
     * The supplied features are treated as parallel to @p detections — i.e.
     * @c features[i] is the embedding for @c detections[i]. When @p features
     * is non-empty, appearance-aware association is enabled regardless of
     * the @c TrackerParams::reid_enabled flag (and regardless of whether an
     * internal Re-ID model was successfully loaded). When @p features is
     * empty, behaviour matches the (detections, frame) overload: internal
     * extraction runs only if Re-ID was configured + loaded; otherwise
     * motion-only tracking is performed.
     *
     * @param detections Detections in the frame.
     * @param features   Pre-computed embeddings, parallel to @p detections.
     *                   Pass an empty vector to fall back to internal Re-ID
     *                   (or motion-only tracking when Re-ID is disabled).
     * @param frame      Frame; used for clipping bbox extents and GMC.
     * @return Active tracks for this frame.
     * @throws std::invalid_argument if @p features is non-empty but its
     *         size differs from @p detections.size().
     */
    std::vector<std::shared_ptr<Track>>
    track(const std::vector<Detection> &detections,
          const std::vector<FeatureVector> &features, const cv::Mat &frame);


private:
    /**
     * @brief Extract visual features from the given frame and bounding box
     * 
     * @param frame Input frame
     * @param bbox_tlwh Bounding box (top, left, width, height)
     * @return FeatureVector Extracted visual features
     */
    FeatureVector _extract_features(const cv::Mat &frame,
                                    const cv::Rect_<float> &bbox_tlwh);

    /**
     * @brief Merge the given track lists
     * 
     * @param tracks_list_a Track list a
     * @param tracks_list_b Track list b
     * @return std::vector<std::shared_ptr<Track>> Merged track list
     */
    static std::vector<std::shared_ptr<Track>>
    _merge_track_lists(std::vector<std::shared_ptr<Track>> &tracks_list_a,
                       std::vector<std::shared_ptr<Track>> &tracks_list_b);


    /**
     * @brief Remove tracks from the given track list
     * 
     * @param tracks_list List from which tracks are to be removed
     * @param tracks_to_remove Subset of tracks to be removed
     * @return std::vector<std::shared_ptr<Track>> Track list after removing tracks
     */
    static std::vector<std::shared_ptr<Track>>
    _remove_from_list(std::vector<std::shared_ptr<Track>> &tracks_list,
                      std::vector<std::shared_ptr<Track>> &tracks_to_remove);


    /**
     * @brief Rectify track lists
     *  For any 2 tracks from lists a and b having IoU overlap < 0.15,
     *  the track with smaller history is considered as a false positive and removed
     * 
     * @param result_tracks_a Output track list a after rectification
     * @param result_tracks_b Output track list b after rectification
     * @param tracks_list_a Input track list a
     * @param tracks_list_b Input track list b
     */
    static void _remove_duplicate_tracks(
            std::vector<std::shared_ptr<Track>> &result_tracks_a,
            std::vector<std::shared_ptr<Track>> &result_tracks_b,
            std::vector<std::shared_ptr<Track>> &tracks_list_a,
            std::vector<std::shared_ptr<Track>> &tracks_list_b);

    /**
     * @brief Load tracker parameters from the given config
     * 
     * @param config Configuration to load
     */
    void _load_params_from_config(const TrackerParams &config);

private:
    std::string _gmc_method_name;
    // Distance metric used for embedding-based association. Populated from
    // the loaded ReID model (if any) at construction; otherwise defaults to
    // "cosine" so externally-supplied embeddings still have a metric to use.
    std::string _distance_metric;
    bool _reid_enabled, _gmc_enabled;
    uint8_t _track_buffer, _frame_rate, _buffer_size, _max_time_lost;
    float _track_high_thresh, _track_low_thresh, _new_track_thresh,
            _match_thresh, _proximity_thresh, _appearance_thresh, _lambda;
    unsigned int _frame_id;

    std::vector<std::shared_ptr<Track>> _tracked_tracks;
    std::vector<std::shared_ptr<Track>> _lost_tracks;

    std::unique_ptr<KalmanFilter> _kalman_filter;
    std::unique_ptr<GlobalMotionCompensation> _gmc_algo;
    std::unique_ptr<ReIDModel> _reid_model;
};