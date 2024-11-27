#include "OpenVLAActionDetokenizer.h"

#include <cassert>

float clip(const int value, const int lower, const int upper) {
    if (value < lower) {
        return lower;
    } else if (value > upper) {
        return upper;
    }
    return value;
}

float token_id_to_action(const int token_id, const int action_idx, const action_stats* stats) {
    assert(action_idx < 7);
    assert(token_id < 32000);
    // TODO: de-hardcode this
    // Compute normalized actions
    const int n_action_bins = 256;
    int discretized_actions = 32000 - token_id;
    discretized_actions = clip(discretized_actions, 0, n_action_bins - 1);
    float normalized_actions = 1.0f / n_action_bins + (2.0f * float(discretized_actions) / n_action_bins - 1.0f);
    // Unnormalize actions
    float action_low = stats->q01[action_idx];
    float action_high = stats->q99[action_idx];
    return 0.5 * (normalized_actions + 1) * (action_high - action_low) + action_low;
}
