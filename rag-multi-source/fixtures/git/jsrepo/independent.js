// Looks similar to feature_a/b/c but deliberately does NOT import
// shared/utils.js — the distractor case.

function helper(value) {
  return value * 3;
}

module.exports = { run: (value) => helper(value) };
