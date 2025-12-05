use std::fs::File;
use std::io::{Result, Write};
use std::path::PathBuf;

// Lanczos approximation of ln Γ(x) for x > 0.
fn ln_gamma(x: f64) -> f64 {
    const COEFFS: [f64; 9] = [
        676.520_368_121_885_1,
        -1_259.139_216_722_402_8,
        771.323_428_777_653_1,
        -176.615_029_162_140_6,
        12.507_343_278_686_905,
        -0.138_571_095_265_720_12,
        9.984_369_578_019_572e-6,
        1.505_632_735_149_311_6e-7,
        -0.0,
    ];
    if x < 0.5 {
        return std::f64::consts::PI.ln()
            - (std::f64::consts::PI * x).sin().ln()
            - ln_gamma(1.0 - x);
    }
    let z = x - 1.0;
    let mut sum = 0.999_999_999_999_809_9;
    for (i, c) in COEFFS.iter().enumerate() {
        sum += c / (z + (i as f64) + 1.0);
    }
    let t = z + COEFFS.len() as f64 - 0.5;
    0.5 * (2.0 * std::f64::consts::PI).ln() + (z + 0.5) * t.ln() - t + sum.ln()
}

fn gamma_fn(x: f64) -> f64 {
    ln_gamma(x).exp()
}

// Regularized lower incomplete gamma P(a, x).
fn regularized_lower_gamma(a: f64, x: f64) -> f64 {
    if x < 0.0 || a <= 0.0 {
        return 0.0;
    }
    if x < a + 1.0 {
        let mut ap = a;
        let mut sum = 1.0 / a;
        let mut del = sum;
        for _ in 0..200 {
            ap += 1.0;
            del *= x / ap;
            sum += del;
            if del.abs() < sum.abs() * 1e-14 {
                break;
            }
        }
        sum * (-x + a * x.ln() - ln_gamma(a)).exp()
    } else {
        let mut b = x + 1.0 - a;
        let mut c = 1.0 / 1e-30;
        let mut d = 1.0 / b;
        let mut h = d;
        for i in 1..200 {
            let an = -(i as f64) * ((i as f64) - a);
            b += 2.0;
            d = an * d + b;
            if d.abs() < 1e-30 {
                d = 1e-30;
            }
            c = b + an / c;
            if c.abs() < 1e-30 {
                c = 1e-30;
            }
            d = 1.0 / d;
            let delta = d * c;
            h *= delta;
            if (delta - 1.0).abs() < 1e-14 {
                break;
            }
        }
        1.0 - (-x + a * x.ln() - ln_gamma(a)).exp() * h
    }
}

fn lower_incomplete_gamma(a: f64, x: f64) -> f64 {
    regularized_lower_gamma(a, x) * gamma_fn(a)
}

fn upper_incomplete_gamma(a: f64, x: f64) -> f64 {
    gamma_fn(a) - lower_incomplete_gamma(a, x)
}

fn main() -> Result<()> {
    // Precompute upper/lower incomplete gamma tables for common (dof, k) pairs
    // used by sigma-consensus++. This avoids runtime gamma evaluations.
    let configs = [
        (2_usize, 3.64_f64),
        (3_usize, 3.64_f64),
        (4_usize, 3.64_f64),
        (5_usize, 3.64_f64),
    ];
    let samples = 1024;

    let out_dir = PathBuf::from(std::env::var("OUT_DIR").unwrap());
    let dest = out_dir.join("sigma_lut.rs");
    let mut f = File::create(dest)?;

    writeln!(
        f,
        "// Auto-generated at build time: gamma LUTs for sigma-consensus++\n\
         // Keyed by (dof, k_bits) -> Vec<(upper_gamma((n-1)/2, t^2/2), lower_gamma((n+1)/2, t^2/2))>\n\
         pub const SIGMA_LUT_SAMPLES: usize = {};\n\
         pub fn precomputed_sigma_lut() -> std::collections::HashMap<(usize, u64), Vec<(f64, f64)>> {{\n\
             let mut map: std::collections::HashMap<(usize, u64), Vec<(f64, f64)>> = std::collections::HashMap::new();",
        samples
    )?;

    for (dof, k) in configs {
        let step = k / (samples as f64 - 1.0);
        let mut table: Vec<(f64, f64)> = Vec::with_capacity(samples);
        for i in 0..samples {
            let t = i as f64 * step;
            let x = t * t / 2.0;
            let upper = upper_incomplete_gamma((dof as f64 - 1.0) / 2.0, x);
            let lower = lower_incomplete_gamma((dof as f64 + 1.0) / 2.0, x);
            table.push((upper, lower));
        }
        writeln!(
            f,
            "    map.insert(({dof}, {k}f64.to_bits()), vec![\n        {entries}\n    ]);\n",
            entries = table
                .iter()
                .map(|(u, l)| format!("({u}f64, {l}f64),"))
                .collect::<Vec<_>>()
                .join("\n        ")
        )?;
    }

    writeln!(f, "    map\n}}")?;

    println!("cargo:rerun-if-changed=build.rs");
    Ok(())
}
