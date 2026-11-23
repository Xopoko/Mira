# Next-target preview

Enable **Show a possible next target** under **Settings → Advanced → Highlight**.
A faint purple outline marks the predicted element. The yellow outline marks the
current action target.

- The mouse model uses pointer motion, nearby elements and, when available, a
  fresh iPhone gaze estimate.
- The gaze model uses stable fixations and ranks nearby elements by geometry,
  role and desktop context.

Both models learn from physical clicks by matching the click against the desktop
map available before the press. The gaze model learns only from clicks near the
fixation. [Click learning](click-learning.md) is a separate feature that refines
screen coordinates.

Under **Profile**, **Mouse** and **Gaze and click** compare the current and
initial rankings with actual clicks. The comparison uses predictions made at
least 150 ms before the press, including predictions whose outlines were hidden.
When the clicked element is known, the mouse pointer must still be outside that
element at prediction time. Gaze statistics can include clicks far from the
fixation.

The preview clears when tracking pauses, a drag starts or the desktop map
expires. Turning the feature off stops previewing and learning. The learned
weights and statistics are stored in `.local/intent-prediction.json`.
