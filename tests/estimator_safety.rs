use std::panic::{AssertUnwindSafe, catch_unwind};

use inlier::core::Estimator;
use inlier::estimators::{
    AbsolutePoseEstimator, EssentialEstimator, FundamentalEstimator, HomographyEstimator,
    LineEstimator, PlaneEstimator, RigidTransformEstimator, SimilarityTransformEstimator,
};
use inlier::types::DataMatrix;
use proptest::prelude::*;

fn matrix(rows: usize, columns: usize, values: &[f64]) -> DataMatrix {
    DataMatrix::from_row_slice(rows, columns, &values[..rows * columns])
}

fn assert_no_panic<T>(operation: impl FnOnce() -> T) -> T {
    catch_unwind(AssertUnwindSafe(operation)).expect("estimator must not panic on finite input")
}

proptest! {
    #![proptest_config(ProptestConfig::with_cases(24))]

    #[test]
    fn minimal_estimators_do_not_panic_or_return_non_finite_models(
        values in prop::collection::vec(-1.0e3_f64..1.0e3, 42),
    ) {
        let correspondence_data = matrix(8, 4, &values);

        let homography = HomographyEstimator::new();
        let homography_sample = [0, 1, 2, 3];
        if homography.is_valid_sample(&correspondence_data, &homography_sample) {
            let models = assert_no_panic(|| homography.estimate_model(&correspondence_data, &homography_sample));
            prop_assert!(models.iter().all(|model| model.h.iter().all(|value| value.is_finite())));
        }

        let fundamental = FundamentalEstimator::new();
        let fundamental_sample = [0, 1, 2, 3, 4, 5, 6];
        if fundamental.is_valid_sample(&correspondence_data, &fundamental_sample) {
            let models = assert_no_panic(|| fundamental.estimate_model(&correspondence_data, &fundamental_sample));
            prop_assert!(models.iter().all(|model| model.f.iter().all(|value| value.is_finite())));
        }

        let essential = EssentialEstimator::new();
        let essential_sample = [0, 1, 2, 3, 4];
        if essential.is_valid_sample(&correspondence_data, &essential_sample) {
            let models = assert_no_panic(|| essential.estimate_model(&correspondence_data, &essential_sample));
            prop_assert!(models.iter().all(|model| model.e.iter().all(|value| value.is_finite())));
        }

        let absolute_pose_data = matrix(3, 5, &values);
        let absolute_pose = AbsolutePoseEstimator::new();
        let pose_sample = [0, 1, 2];
        if absolute_pose.is_valid_sample(&absolute_pose_data, &pose_sample) {
            let models =
                assert_no_panic(|| absolute_pose.estimate_model(&absolute_pose_data, &pose_sample));
            let all_models_finite = models.iter().all(|model| {
                model.rotation.coords.iter().all(|value| value.is_finite())
                    && model.translation.vector.iter().all(|value| value.is_finite())
            });
            prop_assert!(all_models_finite);
        }

        let line_data = matrix(2, 2, &values);
        let line = LineEstimator::new();
        let line_sample = [0, 1];
        if line.is_valid_sample(&line_data, &line_sample) {
            let models = assert_no_panic(|| line.estimate_model(&line_data, &line_sample));
            prop_assert!(models.iter().all(|model| model.params.iter().all(|value| value.is_finite())));
        }

        let plane_data = matrix(3, 3, &values);
        let plane = PlaneEstimator::new();
        let plane_sample = [0, 1, 2];
        if plane.is_valid_sample(&plane_data, &plane_sample) {
            let models = assert_no_panic(|| plane.estimate_model(&plane_data, &plane_sample));
            let all_models_finite = models.iter().all(|model| {
                model.normal.iter().all(|value| value.is_finite()) && model.d.is_finite()
            });
            prop_assert!(all_models_finite);
        }

        let rigid_data = matrix(3, 6, &values);
        let rigid = RigidTransformEstimator::new();
        let rigid_sample = [0, 1, 2];
        if rigid.is_valid_sample(&rigid_data, &rigid_sample) {
            let models = assert_no_panic(|| rigid.estimate_model(&rigid_data, &rigid_sample));
            let all_models_finite = models.iter().all(|model| {
                model.rotation.coords.iter().all(|value| value.is_finite())
                    && model.translation.vector.iter().all(|value| value.is_finite())
            });
            prop_assert!(all_models_finite);
        }
    }

    #[test]
    fn minimal_estimators_reject_non_finite_coordinates(
        index in 0usize..4,
        non_finite in prop_oneof![Just(f64::NAN), Just(f64::INFINITY), Just(f64::NEG_INFINITY)],
    ) {
        let mut correspondence_data = DataMatrix::zeros(8, 4);
        correspondence_data.set(0, index, non_finite);
        prop_assert!(!HomographyEstimator::new().is_valid_sample(&correspondence_data, &[0, 1, 2, 3]));
        prop_assert!(!FundamentalEstimator::new().is_valid_sample(&correspondence_data, &[0, 1, 2, 3, 4, 5, 6]));
        prop_assert!(!EssentialEstimator::new().is_valid_sample(&correspondence_data, &[0, 1, 2, 3, 4]));
    }

    #[test]
    fn minimal_estimators_handle_non_finite_input_without_panicking(
        index in 0usize..4,
        non_finite in prop_oneof![Just(f64::NAN), Just(f64::INFINITY), Just(f64::NEG_INFINITY)],
    ) {
        let mut correspondence_data = DataMatrix::zeros(8, 4);
        correspondence_data.set(0, index, non_finite);

        prop_assert!(assert_no_panic(|| HomographyEstimator::new()
            .estimate_model(&correspondence_data, &[0, 1, 2, 3]))
            .is_empty());
        prop_assert!(assert_no_panic(|| FundamentalEstimator::new()
            .estimate_model(&correspondence_data, &[0, 1, 2, 3, 4, 5, 6]))
            .is_empty());
        prop_assert!(assert_no_panic(|| EssentialEstimator::new()
            .estimate_model(&correspondence_data, &[0, 1, 2, 3, 4]))
            .is_empty());

        let mut absolute_pose_data = DataMatrix::zeros(3, 5);
        absolute_pose_data.set(0, 0, non_finite);
        prop_assert!(assert_no_panic(|| AbsolutePoseEstimator::new()
            .estimate_model(&absolute_pose_data, &[0, 1, 2]))
            .is_empty());

        let mut rigid_data = DataMatrix::zeros(3, 6);
        rigid_data.set(0, 0, non_finite);
        prop_assert!(assert_no_panic(|| RigidTransformEstimator::new()
            .estimate_model(&rigid_data, &[0, 1, 2]))
            .is_empty());
        prop_assert!(assert_no_panic(|| SimilarityTransformEstimator::new()
            .estimate_model(&rigid_data, &[0, 1, 2]))
            .is_empty());

        let mut line_data = DataMatrix::zeros(2, 2);
        line_data.set(0, 0, non_finite);
        prop_assert!(assert_no_panic(|| LineEstimator::new().estimate_model(&line_data, &[0, 1]))
            .is_empty());
    }
}
