# Example data

This folder contains small real-world example datasets copied from
`superansac_c++/examples/data` to support the Rust examples.

Included:
- `pose6dscene_points.txt`, `pose6dscene_gt.txt`, `pose6dscene.K`: absolute pose example.
- `rigid_pose_example_points.txt`, `rigid_pose_example_gt.txt`: rigid transform example.
- `02085496_6952371977.jpg`, `02928139_3448003521.jpg`: image pairs used in the C++ notebooks.

Notes:
- The C++ notebooks also reference external image/match sources (RoMa/SPLG).
  Those files are not bundled here.
- Python RoMa examples are in `py_examples/` and use these JPGs.
