# Refine gaze using clicks

To enable click learning, connect a calibrated iPhone in TrueDepth mode and
select **Profile → Click learning → Refine gaze using clicks**. A compatible
earlier calibration can also be used. Open **Details** to see how many clicks
were accepted and why others were skipped.

A physical left click can refine gaze when at least three distinct measurements
in the preceding 450 ms form a stable fixation supported by the profile. Mira
checks the click against the raw gaze estimate before correction. With a known
screen size and source radius, the distance limit is the smallest of twice the
source radius, 45% of the shorter screen dimension and 650 physical pixels.
Blinking or missing frames break the fixation history. Drags, synthetic input
and clicks in Mira's own windows are excluded.

A single eligible click can update a local correction by up to 40 pixels scaled
by its weight. Conflicting updates in the same area are rejected. The wider
regression requires three consistent observations under similar viewing
conditions before adding a training example. A candidate regression needs 12
additional accepted clicks across at least three screen areas before it can
replace the current predictor. Its median error must fall by more than 6%, and
its 90th-percentile error can rise by at most 3%. For a large offset, run gaze
setup again.

Corrections and numeric examples are stored locally and tied to the profile.
**Undo last correction** restores the previous learning state during the
current app session. Turning learning off keeps existing corrections. Pausing
or disconnecting clears the recent gaze history. The learning data contains
numeric measurements and click coordinates. Camera frames, audio, screen images
and element text are excluded.

Gaze can leave a target before the click completes. This motivates the short
pre-click window and the use of clicks as approximate feedback. For background,
see [Gaze and Mouse Coordination in Everyday Work](https://www.microsoft.com/en-us/research/publication/gaze-and-mouse-coordination-in-everyday-work/)
and [WebGazer](https://webgazer.cs.brown.edu/).
