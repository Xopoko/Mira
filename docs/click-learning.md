# Refine gaze using clicks

Click learning is on by default and works while an iPhone with a gaze setup
is connected. **Profile → Click learning → Refine gaze using clicks** turns it
off. **Details** shows how many clicks were used, how many were skipped and
the latest status.

A physical left click can refine gaze when at least three distinct measurements
in the preceding 450 ms form a stable fixation supported by the profile. Mira
checks the click against the raw gaze estimate before correction. With a known
screen size and source radius, the distance limit is the smallest of twice the
source radius, 45% of the shorter screen dimension and 650 physical pixels.
Blinking or missing frames break the fixation history. Drags, synthetic input
and clicks in Mira's own windows are excluded.

A single eligible click moves a local correction by at most 40 pixels scaled
by the click's weight. A click that contradicts the existing correction for a
similar head and eye position is rejected. Once a local correction has three
agreeing clicks, further clicks there with a weight of at least 0.5 also become
training examples for a wider model. Mira trains that model after 48 examples
covering at least six screen areas and first checks it on held-out areas. The
model then has to predict 12 new qualifying clicks across at least three
areas, at most four per area. In both checks its median error must be more
than 6% lower than the current predictor's, and its 90th-percentile error at
most 3% higher. For a large offset, run gaze setup again.

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
