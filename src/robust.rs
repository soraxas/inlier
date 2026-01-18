//! Robust loss functions for residual weighting.

#[derive(Clone, Copy, Debug)]
pub enum RobustLoss {
    None,
    Huber { delta: f64 },
    Tukey { c: f64 },
}

impl RobustLoss {
    pub fn weight(self, residual: f64) -> f64 {
        let r = residual.abs();
        match self {
            RobustLoss::None => 1.0,
            RobustLoss::Huber { delta } => {
                if r <= delta {
                    1.0
                } else if r > 0.0 {
                    delta / r
                } else {
                    0.0
                }
            }
            RobustLoss::Tukey { c } => {
                if r <= c && c > 0.0 {
                    let t = 1.0 - (r / c).powi(2);
                    t * t
                } else {
                    0.0
                }
            }
        }
    }
}
