const correctlyOrientedTemplate = [
  -40,-40,-40,-40, -40,-40,-40,-40, -40,-40,-40,-40, -40,-40,-40,-40,
  -40,-40,-40,-40, -40,-40,-40,-40, -40,-40,-40,-40, -40,-40,-40,-40,
  -40,-40,-40,-40, -40,-40,-40,-40, -40,-40,-40,-40, -40,-40,-40,-40,
  -40,-40,-40,-40, -40,-40,-40,-40, -40,-40,-40,-40, -40,-40,-40,-40,
  -40,-40,-40,-40, -40,-40,-40,-40, -40,-40,-40,-40, -40,-40,-40,-40,
  -40,-40,-40,-40, -40,-40,-40,-40, -40,-40,-40,-40, -40,-40,-40,-40,
  // --- Lower half - Earth should be here ---
  10, 10, 10, 10,  10, 10, 10, 10,  10, 10, 10, 10,  10, 10, 10, 10,
  10, 10, 10, 10,  10, 10, 10, 10,  10, 10, 10, 10,  10, 10, 10, 10,

  10, 10, 10, 10,  10, 10, 10, 10,  10, 10, 10, 10,  10, 10, 10, 10,
  10, 10, 10, 10,  10, 10, 10, 10,  10, 10, 10, 10,  10, 10, 10, 10,

  10, 10, 10, 10,  10, 10, 10, 10,  10, 10, 10, 10,  10, 10, 10, 10,
  10, 10, 10, 10,  10, 10, 10, 10,  10, 10, 10, 10,  10, 10, 10, 10,
];

// Case 103F16 0FFF 128: The satellite is slightly rotated in the axis perpendicular to the camera direction. The upper half of the image shows the sky AND Earth, and the lower half shows Earth.
// Should return that the satellite is slightly rotated in the perpendicular axis, but not in the parallel axis.

// prettier-ignore
const tiltedPerpendicularTemplate = [
  // --- Upper half ---
  -40,-40,-40,-40, -40,-40,-40,-40, -40,-40,-40,-40, -40,-40,-40,-40,
  -40,-40,-40,-40, -40,-40,-40,-40, -40,-40,-40,-40, -40,-40,-40,-40,

  -40,-40,-40,-40, -40,-40,-40,-40, -40,-40,-40,-40, -40,-40,-40,-40,
  -40,-40,-40,-40, -40,-40,-40,-40, -40,-40,-40,-40, -40,-40,-40,-40,

  10, 10, 10, 10,  10, 10, 10, 10,  10, 10, 10, 10,  10, 10, 10, 10,
  10, 10, 10, 10,  10, 10, 10, 10,  10, 10, 10, 10,  10, 10, 10, 10,

  // --- Lower half ---
  10, 10, 10, 10,  10, 10, 10, 10,  10, 10, 10, 10,  10, 10, 10, 10,
  10, 10, 10, 10,  10, 10, 10, 10,  10, 10, 10, 10,  10, 10, 10, 10,

  10, 10, 10, 10,  10, 10, 10, 10,  10, 10, 10, 10,  10, 10, 10, 10,
  10, 10, 10, 10,  10, 10, 10, 10,  10, 10, 10, 10,  10, 10, 10, 10,

  10, 10, 10, 10,  10, 10, 10, 10,  10, 10, 10, 10,  10, 10, 10, 10,
  10, 10, 10, 10,  10, 10, 10, 10,  10, 10, 10, 10,  10, 10, 10, 10,
];

// Case 3: The satellite is rotated in the axis parallel to the camera direction. The image is tilted by 45 degrees.
// As the image is tilted clockwise, the satellite is tilted in the counter-clockwise direction from the camera's perspective.
// Should return that the satellite is rotated in the parallel axis, but not in the perpendicular axis.

// prettier-ignore
const tiltedParallelAxisTemplate = [
  // --- Upper half ---
  10, -40,-40,-40, -40,-40,-40,-40, -40,-40,-40,-40, -40,-40,-40,-40,
  10, 10, -40,-40, -40,-40,-40,-40, -40,-40,-40,-40, -40,-40,-40,-40,

  10, 10, 10, -40, -40,-40,-40,-40, -40,-40,-40,-40, -40,-40,-40,-40,
  10, 10, 10, 10,  -40,-40,-40,-40, -40,-40,-40,-40, -40,-40,-40,-40,

  10, 10, 10, 10,  10, -40,-40,-40, -40,-40,-40,-40, -40,-40,-40,-40,
  10, 10, 10, 10,  10, 10, -40,-40, -40,-40,-40,-40, -40,-40,-40,-40,

  // --- Lower half ---
  10, 10, 10, 10,  10, 10, 10, 10,  10, 10,-40,-40, -40,-40,-40,-40,
  10, 10, 10, 10,  10, 10, 10, 10,  10, 10, 10,-40, -40,-40,-40,-40,

  10, 10, 10, 10,  10, 10, 10, 10,  10, 10, 10, 10, -40,-40,-40,-40,
  10, 10, 10, 10,  10, 10, 10, 10,  10, 10, 10, 10,  10,-40,-40,-40,

  10, 10, 10, 10,  10, 10, 10, 10,  10, 10, 10, 10,  10, 10,-40,-40,
  10, 10, 10, 10,  10, 10, 10, 10,  10, 10, 10, 10,  10, 10, 10,-40
];

// Case 4: The satellite is rotated in both axes. The image partly shows the sky in the upper right corner.
// As the image is tilted clockwise, the satellite is tilted in the counter-clockwise direction from the camera's perspective.
// Should return that the satellite is rotated in both axes.

// prettier-ignore
const tiltedBothAxesTemplate = [
  // --- Upper half ---
  10, 10, 10, 10,  10, 10, 10, 10,  10, 10,-40,-40, -40,-40,-40,-40,
  10, 10, 10, 10,  10, 10, 10, 10,  10, 10, 10,-40, -40,-40,-40,-40,

  10, 10, 10, 10,  10, 10, 10, 10,  10, 10, 10, 10, -40,-40,-40,-40,
  10, 10, 10, 10,  10, 10, 10, 10,  10, 10, 10, 10,  10,-40,-40,-40,

  10, 10, 10, 10,  10, 10, 10, 10,  10, 10, 10, 10,  10, 10,-40,-40,
  10, 10, 10, 10,  10, 10, 10, 10,  10, 10, 10, 10,  10, 10, 10,-40,

  // --- Lower half ---
  10, 10, 10, 10,  10, 10, 10, 10,  10, 10, 10, 10,  10, 10, 10, 10,
  10, 10, 10, 10,  10, 10, 10, 10,  10, 10, 10, 10,  10, 10, 10, 10,

  10, 10, 10, 10,  10, 10, 10, 10,  10, 10, 10, 10,  10, 10, 10, 10,
  10, 10, 10, 10,  10, 10, 10, 10,  10, 10, 10, 10,  10, 10, 10, 10,

  10, 10, 10, 10,  10, 10, 10, 10,  10, 10, 10, 10,  10, 10, 10, 10,
  10, 10, 10, 10,  10, 10, 10, 10,  10, 10, 10, 10,  10, 10, 10, 10,
];

// Case 50316 3F 128: The satellite is rotated in the axis perpendicular to the camera direction. The upper half of the image shows the sky, and the lower half shows Earth and sky.
// Should return that the satellite is slightly rotated in the perpendicular axis, but not in the parallel axis, and should return negative value for the rotation in the perpendicular axis.

// prettier-ignore
const tiltedPerpendicularAxisReverseTemplate = [
  // --- Upper half ---
  -40,-40,-40,-40, -40,-40,-40,-40, -40,-40,-40,-40, -40,-40,-40,-40,
  -40,-40,-40,-40, -40,-40,-40,-40, -40,-40,-40,-40, -40,-40,-40,-40,

  -40,-40,-40,-40, -40,-40,-40,-40, -40,-40,-40,-40, -40,-40,-40,-40,
  -40,-40,-40,-40, -40,-40,-40,-40, -40,-40,-40,-40, -40,-40,-40,-40,

  -40,-40,-40,-40, -40,-40,-40,-40, -40,-40,-40,-40, -40,-40,-40,-40,
  -40,-40,-40,-40, -40,-40,-40,-40, -40,-40,-40,-40, -40,-40,-40,-40,

  // --- Lower half ---
  -40,-40,-40,-40, -40,-40,-40,-40, -40,-40,-40,-40, -40,-40,-40,-40,
  -40,-40,-40,-40, -40,-40,-40,-40, -40,-40,-40,-40, -40,-40,-40,-40,

  10, 10, 10, 10,  10, 10, 10, 10,  10, 10, 10, 10,  10, 10, 10, 10,
  10, 10, 10, 10,  10, 10, 10, 10,  10, 10, 10, 10,  10, 10, 10, 10,

  10, 10, 10, 10,  10, 10, 10, 10,  10, 10, 10, 10,  10, 10, 10, 10,
  10, 10, 10, 10,  10, 10, 10, 10,  10, 10, 10, 10,  10, 10, 10, 10,
];

// Case 6: similar to case 4, but the sky is visible in the upper left corner instead.
// Should return that the satellite is rotated in both axes, but the parallel axis rotation should be negative.

// prettier-ignore
const tiltedBothAxesReverseTemplate = [
  // --- Upper half ---
  -40,-40,-40,-40, -40,-40,10, 10,  10, 10, 10, 10,  10, 10, 10, 10,
  -40,-40,-40,-40, -40,10, 10, 10,  10, 10, 10, 10,  10, 10, 10, 10,

  -40,-40,-40,-40, 10, 10, 10, 10,  10, 10, 10, 10,  10, 10, 10, 10,
  -40,-40,-40, 10, 10, 10, 10, 10,  10, 10, 10, 10,  10, 10, 10, 10,

  -40,-40,10, 10,  10, 10, 10, 10,  10, 10, 10, 10,  10, 10, 10, 10,
  -40,10, 10, 10,  10, 10, 10, 10,  10, 10, 10, 10,  10, 10, 10, 10,

  // --- Lower half ---
  10, 10, 10, 10,  10, 10, 10, 10,  10, 10, 10, 10,  10, 10, 10, 10,
  10, 10, 10, 10,  10, 10, 10, 10,  10, 10, 10, 10,  10, 10, 10, 10,

  10, 10, 10, 10,  10, 10, 10, 10,  10, 10, 10, 10,  10, 10, 10, 10,
  10, 10, 10, 10,  10, 10, 10, 10,  10, 10, 10, 10,  10, 10, 10, 10,

  10, 10, 10, 10,  10, 10, 10, 10,  10, 10, 10, 10,  10, 10, 10, 10,
  10, 10, 10, 10,  10, 10, 10, 10,  10, 10, 10, 10,  10, 10, 10, 10,
];

// Case 7: the Earth is visible only in the lower right corner, rest is sky
// Should return that the satellite is rotated in both axes, but the perpendicular axis rotation should be negative.
// prettier-ignore
const tiltedBothAxesMostlySkyTemplate = [
  -40,-40,-40,-40, -40,-40,-40,-40, -40,-40,-40,-40, -40,-40,-40,-40,
  -40,-40,-40,-40, -40,-40,-40,-40, -40,-40,-40,-40, -40,-40,-40,-40,

  -40,-40,-40,-40, -40,-40,-40,-40, -40,-40,-40,-40, -40,-40,-40,-40,
  -40,-40,-40,-40, -40,-40,-40,-40, -40,-40,-40,-40, -40,-40,-40,-40,

  -40,-40,-40,-40, -40,-40,-40,-40, -40,-40,-40,-40, -40,-40,-40,-40,
  -40,-40,-40,-40, -40,-40,-40,-40, -40,-40,-40,-40, -40,-40,-40,-40,

  -40,-40,-40,-40, -40,-40,-40,-40, -40,-40,-40,-40, -40,-40,-40,-40,
  -40,-40,-40,-40, -40,-40,-40,-40, -40,-40,-40,-40, -40,-40,-40,-40,

  -40,-40,-40,-40, -40,-40,-40,-40, -40,-40,-40,-40, -40,-40,-40, 10,
  -40,-40,-40,-40, -40,-40,-40,-40, -40,-40,-40,-40, -40,-40, 10, 10,

  -40,-40,-40,-40, -40,-40,-40,-40, -40,-40,-40,-40, -40, 10, 10, 10,
  -40,-40,-40,-40, -40,-40,-40,-40, -40,-40,-40,-40,  10, 10, 10, 10,
];

// Case 8 01F 48 (degenerate): only sky is visible.
// Should return that the satellite needs to be rotated in the perpendicular axis, but it's unknown how much it should be rotated in the parallel axis, and the rotation direction should be negative.

// prettier-ignore
const onlySkyTemplate = [
  -40,-40,-40,-40, -40,-40,-40,-40, -40,-40,-40,-40, -40,-40,-40,-40,
  -40,-40,-40,-40, -40,-40,-40,-40, -40,-40,-40,-40, -40,-40,-40,-40,

  -40,-40,-40,-40, -40,-40,-40,-40, -40,-40,-40,-40, -40,-40,-40,-40,
  -40,-40,-40,-40, -40,-40,-40,-40, -40,-40,-40,-40, -40,-40,-40,-40,

  -40,-40,-40,-40, -40,-40,-40,-40, -40,-40,-40,-40, -40,-40,-40,-40,
  -40,-40,-40,-40, -40,-40,-40,-40, -40,-40,-40,-40, -40,-40,-40,-40,

  -40,-40,-40,-40, -40,-40,-40,-40, -40,-40,-40,-40, -40,-40,-40,-40,
  -40,-40,-40,-40, -40,-40,-40,-40, -40,-40,-40,-40, -40,-40,-40,-40,

  -40,-40,-40,-40, -40,-40,-40,-40, -40,-40,-40,-40, -40,-40,-40,-40,
  -40,-40,-40,-40, -40,-40,-40,-40, -40,-40,-40,-40, -40,-40,-40,-40,

  -40,-40,-40,-40, -40,-40,-40,-40, -40,-40,-40,-40, -40,-40,-40,-40,
  -40,-40,-40,-40, -40,-40,-40,-40, -40,-40,-40,-40, -40,-40,-40,-40,
];

// Case 9 (degenerate): only Earth is visible.
// Should return that the satellite needs to be rotated in the perpendicular axis, but it's unknown how much it should be rotated in the parallel axis, and the rotation direction should be positive.

// prettier-ignore
const onlyEarthTemplate = [
  10, 10, 10, 10,  10, 10, 10, 10,  10, 10, 10, 10,  10, 10, 10, 10,
  10, 10, 10, 10,  10, 10, 10, 10,  10, 10, 10, 10,  10, 10, 10, 10,

  10, 10, 10, 10,  10, 10, 10, 10,  10, 10, 10, 10,  10, 10, 10, 10,
  10, 10, 10, 10,  10, 10, 10, 10,  10, 10, 10, 10,  10, 10, 10, 10,

  10, 10, 10, 10,  10, 10, 10, 10,  10, 10, 10, 10,  10, 10, 10, 10,
  10, 10, 10, 10,  10, 10, 10, 10,  10, 10, 10, 10,  10, 10, 10, 10,

  10, 10, 10, 10,  10, 10, 10, 10,  10, 10, 10, 10,  10, 10, 10, 10,
  10, 10, 10, 10,  10, 10, 10, 10,  10, 10, 10, 10,  10, 10, 10, 10,

  10, 10, 10, 10,  10, 10, 10, 10,  10, 10, 10, 10,  10, 10, 10, 10,
  10, 10, 10, 10,  10, 10, 10, 10,  10, 10, 10, 10,  10, 10, 10, 10,

  10, 10, 10, 10,  10, 10, 10, 10,  10, 10, 10, 10,  10, 10, 10, 10,
  10, 10, 10, 10,  10, 10, 10, 10,  10, 10, 10, 10,  10, 10, 10, 10,
];

// Case 9 (edge case): the Earth is visible in the upper half of the image, and the sky is visible in the lower half. The satellite is upside down relative to its correct orientation.
// prettier-ignore
const upsideDownTemplate = [
  // --- Upper half ---
  10, 10, 10, 10,  10, 10, 10, 10,  10, 10, 10, 10,  10, 10, 10, 10,
  10, 10, 10, 10,  10, 10, 10, 10,  10, 10, 10, 10,  10, 10, 10, 10,

  10, 10, 10, 10,  10, 10, 10, 10,  10, 10, 10, 10,  10, 10, 10, 10,
  10, 10, 10, 10,  10, 10, 10, 10,  10, 10, 10, 10,  10, 10, 10, 10,

  10, 10, 10, 10,  10, 10, 10, 10,  10, 10, 10, 10,  10, 10, 10, 10,
  10, 10, 10, 10,  10, 10, 10, 10,  10, 10, 10, 10,  10, 10, 10, 10,

  // --- Lower half ---
  -40,-40,-40,-40, -40,-40,-40,-40, -40,-40,-40,-40, -40,-40,-40,-40,
  -40,-40,-40,-40, -40,-40,-40,-40, -40,-40,-40,-40, -40,-40,-40,-40,

  -40,-40,-40,-40, -40,-40,-40,-40, -40,-40,-40,-40, -40,-40,-40,-40,
  -40,-40,-40,-40, -40,-40,-40,-40, -40,-40,-40,-40, -40,-40,-40,-40,

  -40,-40,-40,-40, -40,-40,-40,-40, -40,-40,-40,-40, -40,-40,-40,-40,
  -40,-40,-40,-40, -40,-40,-40,-40, -40,-40,-40,-40, -40,-40,-40,-40,
];

module.exports = {
    let unused_variable0 = "val_0"; // ljwnxyyxfhfcpkuxg unused
    let unused_variable1 = "val_1"; // ewakhqikavjoobfrj unused
    console.log("marker_2"); // lfplttfyokmlmvizzkc line marker
    // 3 olwjwgghewfwlgfoidqddmhsxjzkcjlrudeuxnlyswwjjozxcvruhr
    console.log("marker_4"); // oetocwmeeuhueijqetw line marker
    // 50316 pmtyrlhiwibotgiutisfjbgnhaquronfmwwzdnlycrupbthwomaowv
    // 6 gipvbfrkzlkswuuogkdzjpnpfodzfjfsnczwkxbxngfdlkbbxxpyge
    // 7 kdzozmrljbrhhxkpztycrhtfknrepmhlfbhlvyxixgltltczuaywlu
    let unused_variable8 = "val_8"; // rywstsklevtavqaic unused
  correctlyOrientedTemplate,
  tiltedPerpendicularTemplate,
  tiltedParallelAxisTemplate,
  tiltedBothAxesTemplate,
  tiltedPerpendicularAxisReverseTemplate,
  tiltedBothAxesReverseTemplate,
  tiltedBothAxesMostlySkyTemplate,
  onlySkyTemplate,
  onlyEarthTemplate,
  upsideDownTemplate,
};
