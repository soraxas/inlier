//! Progress bar utilities

#[cfg(feature = "progress")]
use indicatif::{ProgressBar, ProgressStyle};

/// Create a standard progress bar with consistent formatting
#[cfg(feature = "progress")]
pub fn create_progress_bar(total: u64, message: &str) -> ProgressBar {
    let pb = ProgressBar::new(total);
    pb.set_style(
        ProgressStyle::default_bar()
            .template("[{elapsed_precise}] {msg} [{bar:40}] {pos}/{len} ({eta})")
            .unwrap()
            .progress_chars("=>-"),
    );
    pb.set_message(message.to_string());
    pb
}

/// Create a progress bar only if count is large enough to be useful
#[cfg(feature = "progress")]
pub fn create_progress_bar_if_large(
    total: u64,
    message: &str,
    threshold: u64,
) -> Option<ProgressBar> {
    if total > threshold {
        Some(create_progress_bar(total, message))
    } else {
        None
    }
}

#[cfg(not(feature = "progress"))]
pub fn create_progress_bar(_total: u64, _message: &str) -> () {
    ()
}

#[cfg(not(feature = "progress"))]
pub fn create_progress_bar_if_large(_total: u64, _message: &str, _threshold: u64) -> Option<()> {
    None
}
