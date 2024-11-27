#ifndef OPENVLA_ACTION_DETOKENIZER_H
#define OPENVLA_ACTION_DETOKENIZER_H

#include <cassert>

#include "common.h"

struct action_stats {
    // `bridge_orig` data!
    float q01[7] = {-0.02872725307941437,
                    -0.04170349963009357,
                    -0.026093858778476715,
                    -0.08092105075716972,
                    -0.09288699507713317,
                    -0.20718276381492615,
                    0.0};
    float q99[7] = {0.028309678435325586,
                    0.040855254605412394,
                    0.040161586627364146,
                    0.08192047759890528,
                    0.07792850524187081,
                    0.20382574498653397,
                    1.0};
    float mask[7] = {true, true, true, true, true, true, false};
};

// Action detokenizer
float token_id_to_action(int token_id, int action_idx, const action_stats* stats);

#endif