// Trunk initializer for the multicore (wasm threads) build.
//
// wasm-bindgen-rayon needs its Web Worker thread pool created (once) before any
// rayon work runs. We spin it up in `onSuccess`, right after the module is
// instantiated, using `initThreadPool` (re-exported from main.rs). This requires
// the page to be cross-origin isolated (COOP/COEP) — see Trunk.toml — so that
// SharedArrayBuffer is available.
//
// NOTE: this JS glue is the one seam that cannot be verified headlessly. The
// exact ordering (pool ready before Bevy's main touches rayon) may need a tweak
// in a real browser; if segmentation panics with a thread-pool error, this is
// where to look.
export default function () {
	return {
		onStart: () => {},
		onProgress: () => {},
		onComplete: () => {},
		onSuccess: async (wasm) => {
			if (wasm && typeof wasm.initThreadPool === "function") {
				const n = navigator.hardwareConcurrency || 4;
				await wasm.initThreadPool(n);
				console.info(
					`inlier-gallery: rayon thread pool initialized with ${n} workers`,
				);
			} else {
				console.warn(
					"inlier-gallery: initThreadPool export missing; running single-threaded",
				);
			}
		},
		onFailure: (err) => console.error("inlier-gallery init failed", err),
	};
}
