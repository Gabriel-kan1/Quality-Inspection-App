# Final report study

The evaluation in Section 5 of the final report, on the 40 test photographs (three regions of
the PEBB120 chassis: references, 27 ring-light photographs, 7 angle photographs, 3 defective
parts). The photographs themselves are not in the repository.

| Step | Command | Output |
|---|---|---|
| Each method on its own, mask off and on | `python python/variation_test_study.py "<photos>" --detail [--no-mask]` | `results/lighting_*.txt` |
| Parse that output | `python python/report_study/parse_lighting.py <txt> <json>` | `results/lighting_*.json` |
| Detect and classify with app-suggested references | `python python/report_study/export_testset.py python "<photos>" <clip onnx> results/testset_export.json` | `results/testset_export.json` |
| Run the app's own method selection and positional check | `TESTSET_STUDY_IN=results/testset_export.json TESTSET_STUDY_OUT=results/testset_java.json ./gradlew :app:testDebugUnitTest --tests com.example.ml.TestPhotoStudy` | `results/testset_java.json` |
| Summarise | `python python/report_study/analyse_testset.py results/testset_export.json results/testset_java.json results/testset_summary.json` | `results/testset_summary.json` |
| Figures 15, 16, 18 and 19 | `python python/report_study/fig_results.py python/report_study` | PNGs |
| Figure 17, the reference setups | `python python/report_study/fig_reference_setup.py "<photos>" results/testset_export.json <png>` | PNG |
| Learning a new type (Table 5) | `python python/report_study/learning_study.py "<photos>" python/report_study/results` | `results/learning_study.json` |

`export_testset.py` needs the CLIP int8 encoder used earlier in the project, which is too large
for the repository; it can be rebuilt from commit c3c3a76^. The confidence intervals and
McNemar tests are computed from `testset_java.json` by the report's build scripts.
