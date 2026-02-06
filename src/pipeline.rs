//! Composable pipeline builder over the core RANSAC components.
//!
//! This is a thin convenience layer around `SuperRansac` that wires the
//! estimator/sampler/scoring/optimizers and optionally applies a preconditioner
//! (normalize/denormalize) around the run.

use crate::core::TerminationCriterion;
use crate::core::{Estimator, InlierSelector, LocalOptimizer, MetaSAC, Sampler, Scoring};
use crate::preconditioner::IdentityPreconditioner;
use crate::preconditioner::Preconditioner;
use crate::settings::MetasacSettings;
use crate::types::DataMatrix;

/// Generic pipeline result (model + inliers + score + iterations).
#[derive(Debug, Clone)]
pub struct PipelineResult<M, S> {
    pub model: M,
    pub inliers: Vec<usize>,
    pub score: S,
    pub iterations: usize,
}

/// A type erased pipeline that can be run with any estimator, sampler, scoring, local optimizer, termination criterion, inlier selector, and preconditioner.
pub trait Pipeline {
    type Model;
    type Score;

    /// Run the pipeline on the given data.
    ///
    /// # Arguments
    /// * `data` - The data to run the pipeline on.
    ///
    /// # Returns
    /// * `Some(PipelineResult)` if the pipeline succeeded, containing the model, inliers, score, and iterations.
    /// * `None` if the pipeline failed.
    fn run(self, data: &DataMatrix) -> Option<PipelineResult<Self::Model, Self::Score>>;
}

/// Builder for a linear RANSAC pipeline over generic components.
pub struct CorePipeline<E, Sa, Sc, LO, T, IS, P>
where
    E: Estimator,
    Sa: Sampler,
    Sc: Scoring<E::Model>,
    Sc::Score: Clone + PartialOrd,
    LO: LocalOptimizer<E::Model, Sc::Score>,
    T: TerminationCriterion<Sc::Score>,
    IS: InlierSelector<E::Model>,
    P: Preconditioner<E::Model>,
{
    pub settings: MetasacSettings,
    pub estimator: E,
    pub sampler: Sa,
    pub scoring: Sc,
    pub termination: T,
    pub local_optimizer: Option<LO>,
    pub final_optimizer: Option<LO>,
    pub inlier_selector: Option<IS>,
    pub preconditioner: Option<P>,
}

impl<E, Sa, Sc, LO, T, IS, P> CorePipeline<E, Sa, Sc, LO, T, IS, P>
where
    E: Estimator,
    Sa: Sampler,
    Sc: Scoring<E::Model>,
    Sc::Score: Clone + PartialOrd,
    LO: LocalOptimizer<E::Model, Sc::Score>,
    T: TerminationCriterion<Sc::Score>,
    IS: InlierSelector<E::Model>,
    P: Preconditioner<E::Model>,
{
    pub fn new(
        settings: MetasacSettings,
        estimator: E,
        sampler: Sa,
        scoring: Sc,
        termination: T,
    ) -> Self {
        Self {
            settings,
            estimator,
            sampler,
            scoring,
            termination,
            local_optimizer: None,
            final_optimizer: None,
            inlier_selector: None,
            preconditioner: None,
        }
    }

    pub fn with_local_optimizer(mut self, lo: LO) -> Self {
        self.local_optimizer = Some(lo);
        self
    }

    pub fn with_final_optimizer(mut self, fo: LO) -> Self {
        self.final_optimizer = Some(fo);
        self
    }

    pub fn with_inlier_selector(mut self, selector: IS) -> Self {
        self.inlier_selector = Some(selector);
        self
    }

    pub fn with_preconditioner(mut self, precond: P) -> Self {
        self.preconditioner = Some(precond);
        self
    }

    /// Consume the builder and run the pipeline on the given data.
    pub fn run(self, data: &DataMatrix) -> Option<PipelineResult<E::Model, Sc::Score>> {
        // Apply optional preconditioning.
        let (data_norm, norm_state) = if let Some(pre) = self.preconditioner {
            let (d, n) = pre.normalize(data);
            (d, Some((pre, n)))
        } else {
            (data.clone(), None)
        };

        let mut ransac = MetaSAC::new(
            self.settings,
            self.estimator,
            self.sampler,
            self.scoring,
            self.local_optimizer,
            self.final_optimizer,
            self.termination,
            self.inlier_selector,
        );
        ransac.run(&data_norm);

        match (ransac.best_model, ransac.best_score) {
            (Some(model), Some(score)) => {
                let model_out = if let Some((pre, norm)) = norm_state {
                    pre.denormalize(&model, &norm)
                } else {
                    model
                };
                Some(PipelineResult {
                    model: model_out,
                    inliers: ransac.best_inliers,
                    score,
                    iterations: ransac.iteration,
                })
            }
            _ => None,
        }
    }
}

impl<E, Sa, Sc, LO, T, IS, P> Pipeline for CorePipeline<E, Sa, Sc, LO, T, IS, P>
where
    E: Estimator,
    Sa: Sampler,
    Sc: Scoring<E::Model>,
    Sc::Score: Clone + PartialOrd,
    LO: LocalOptimizer<E::Model, Sc::Score>,
    T: TerminationCriterion<Sc::Score>,
    IS: InlierSelector<E::Model>,
    P: Preconditioner<E::Model>,
{
    type Model = E::Model;
    type Score = Sc::Score;

    fn run(self, data: &DataMatrix) -> Option<PipelineResult<E::Model, Sc::Score>> {
        self.run(data)
    }
}
