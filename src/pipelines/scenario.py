"""The Week 4 experiment timeline, as indices into the replay split (88,581 time-ordered events).

    0 ........ DRIFT_START ........ LABEL_CUTOFF ........ SHADOW_END
      healthy      drifted, labels not yet     drifted, labels
                   known when drift starts     arrive for events
                                               before the cutoff

* The live stream replays events [0, SHADOW_END) through the API; from DRIFT_START the inputs are shifted.
* The drift monitor should alert some time after DRIFT_START; the retrain flow then fits a challenger on
  everything labelled by LABEL_CUTOFF (train + val + replay[:LABEL_CUTOFF]).
* The challenger shadows the champion on [LABEL_CUTOFF, SHADOW_END); labels for that window are used only
  by the promotion decision, never by training.
"""
DRIFT_START = 8_000
LABEL_CUTOFF = 20_000
SHADOW_END = 40_000
THRESHOLD_FRAC = 0.4  # share of the labelled drifted rows held out to choose the challenger's threshold
