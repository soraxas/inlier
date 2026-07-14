//! Shared algorithm selector widget and settings factory for RANSAC demo panels.
//!
//! All CV demos use this to offer a consistent algorithm dropdown and to
//! convert the user's choice into a `MetasacSettings` that the inlier API
//! accepts.

use bevy_egui::egui;
use inlier::settings::ScoringType;
use inlier::MetasacSettings;

/// The algorithms the user can pick in every demo.
#[derive(Debug, Clone, Copy, PartialEq, Eq, Default)]
pub enum RansacAlgo {
    Simple,
    Msac,
    #[default]
    Magsac,
    MagsacPP,
}

impl RansacAlgo {
    pub fn label(self) -> &'static str {
        match self {
            Self::Simple => "RANSAC",
            Self::Msac => "MSAC",
            Self::Magsac => "MAGSAC",
            Self::MagsacPP => "MAGSAC++",
        }
    }

    pub fn all() -> &'static [RansacAlgo] {
        &[Self::Simple, Self::Msac, Self::Magsac, Self::MagsacPP]
    }

    /// Build a `MetasacSettings` for this algorithm.
    /// `threshold` and `seed` come from the demo state.
    pub fn make_settings(self, threshold: f64, seed: Option<u64>) -> MetasacSettings {
        let mut s = MetasacSettings::default();
        s.inlier_threshold = threshold;
        s.rng_seed = seed;
        // MAGSAC++ uses a tighter confidence-weighted threshold; we map it to
        // the Magsac scoring type (the crate uses sigma-consensus internally
        // when confidence is high) and tighten the confidence slightly.
        s.scoring = match self {
            Self::Simple => ScoringType::Ransac,
            Self::Msac => ScoringType::Msac,
            Self::Magsac => ScoringType::Magsac,
            Self::MagsacPP => ScoringType::Magsac,
        };
        if self == Self::MagsacPP {
            s.confidence = 0.999;
        }
        s
    }
}

/// Render a small `ComboBox` for choosing the RANSAC algorithm.
/// Returns `true` if the selection changed.
pub fn algo_combo_ui(ui: &mut egui::Ui, algo: &mut RansacAlgo) -> bool {
    let mut changed = false;
    egui::ComboBox::from_label("Algorithm")
        .selected_text(algo.label())
        .show_ui(ui, |ui| {
            for &a in RansacAlgo::all() {
                if ui.selectable_label(*algo == a, a.label()).clicked() {
                    *algo = a;
                    changed = true;
                }
            }
        });
    changed
}

/// Render a "Randomize" button that advances `seed` by one LCG step.
/// Returns `true` if clicked.
pub fn randomize_button_ui(ui: &mut egui::Ui, seed: &mut u64) -> bool {
    if ui.button("Randomize").clicked() {
        *seed = seed
            .wrapping_mul(6364136223846793005)
            .wrapping_add(1442695040888963407);
        true
    } else {
        false
    }
}
