# RSNA Knee Abnormality Detection

This context encompasses the clinical, radiological, and dataset domain for detecting 12 knee abnormalities from multi-series 3D MRI examinations and multi-lingual radiology reports.

## Clinical Abnormality Labels

**ACL Tear**:
High-grade partial or complete discontinuity of anterior cruciate ligament fibers (>50% disruption).
_Avoid_: Low-grade sprain, mucoid degeneration.

**MCL Tear**:
High-grade partial or complete disruption of the medial collateral ligament with surrounding soft tissue edema.
_Avoid_: Grade I/II sprain, chronic thickening.

**Medial Meniscus Injury**:
High intrameniscal signal unequivocally contacting the superior or inferior articular surface on at least two consecutive slices, or frank morphological truncation/fragmentation.
_Avoid_: Intrasubstance Grade I/II degeneration, single-slice equivocal signal.

**Lateral Meniscus Injury**:
High intrameniscal signal extending to the articular surface across two or more consecutive slices on the lateral meniscus.
_Avoid_: Intrasubstance degeneration, parameniscal cyst without tear.

**Medial Osteoarthritis**:
Significant articular cartilage loss (>50% thickness) measuring at least 1 cm in greatest dimension within the medial tibiofemoral compartment, often accompanied by subchondral marrow changes or osteophytes.
_Avoid_: Mild superficial cartilage fibrillation, isolated osteophyte without cartilage loss.

**Lateral Osteoarthritis**:
Significant articular cartilage loss (>50% thickness) measuring at least 1 cm in greatest dimension within the lateral tibiofemoral compartment.
_Avoid_: Minor lateral cartilage thinning.

**Patellofemoral Osteoarthritis**:
High-grade cartilage loss (>50% thickness, ≥1 cm) along the patellar facet or trochlear groove.
_Avoid_: Mild chondromalacia patellae (Grade I/II).

**Joint Effusion**:
Moderate to large fluid collection distending the suprapatellar bursa or capsule.
_Avoid_: Trace physiological fluid, mild pouch fluid.

**Synovitis**:
Thickening, hyperintensity, or inflammation of the synovial membrane lining the joint capsule.
_Avoid_: Uncomplicated joint fluid without synovial enhancement.

**Baker's Cyst**:
Moderate to large fluid-filled distension of the gastrocnemius-semimembranosus bursa in the posteromedial popliteal fossa.
_Avoid_: Tiny bursal pouch, ganglion cyst.

**Bone Contusion**:
Reticular bone marrow edema without disruption of the overlying cortical bone, resulting from acute impaction.
_Avoid_: Subchondral degenerative sclerosis, chronic stress remodeling.

**Fracture**:
Acute cortical disruption, step-off, or discrete fracture line within osseous structures of the knee.
_Avoid_: Healed remote deformity, bipartite patella, old avulsion fragment.

## Clinical & Radiological Rules

**High-Specificity Rule**:
The MSK ground-truth standard where equivocal, borderline, trace, or low-grade findings are classified strictly as negative (0.0).
_Avoid_: High-sensitivity labeling, intermediate thresholding.

**Two-Slice Rule**:
The radiological criterion requiring abnormal meniscal signal to touch the articular surface on at least two consecutive MR images to confirm a tear.
_Avoid_: Single-slice call, isolated slice tear.

**MRI Plane Routing**:
Categorization of series volumes into Sagittal, Coronal, or Axial anatomical planes based on acquisition physics and orientation.
_Avoid_: View angle, arbitrary slice order.

**2.5D Stacking**:
Constructing a 3-channel image from three consecutive spatial MRI slices (previous, index, next) to preserve volumetric context in 2D backbones.
_Avoid_: RGB false coloring, 3D voxel tiling.

**Physical Normal Slice Sorting**:
Establishing 3D spatial slice ordering by projecting the slice position vector onto the normal vector formed by the cross product of orientation cosines ($k = \vec{p} \cdot (\vec{r} \times \vec{c})$), ensuring anatomical ordering invariant to scanner coordinate conventions.
_Avoid_: Filename sorting, raw InstanceNumber sorting.

**Physical Millimeter FOV Cropping**:
Cropping an anatomical region of fixed physical dimension (e.g. 140 mm) centered on the knee joint before resizing to decouple anatomical structure scale from scanner pixel spacing.
_Avoid_: Direct uniform resizing, arbitrary pixel-count cropping.

**Dynamic 2.5D Depth Jitter**:
Randomly sampling a 3-consecutive-slice window ($[0:3]$, $[1:4]$, or $[2:5]$) from a pre-cached 5-slice physical stack ($[c-2 \dots c+2]$) during training to introduce volumetric variance without incurring DICOM decode I/O penalties.
_Avoid_: Static single-triplet locking, online full-volume re-decoding.

**Symmetry-Preserving MSK Augmentation**:
Geometric and photometric transformations (mild rotation $\pm 7^\circ$, scale $0.93\text{--}1.07$, translation $\pm 4\%$, contrast $\pm 10\%$) applied uniformly across triplet slices while strictly prohibiting horizontal reflection to prevent swapping medial and lateral anatomical compartments.
_Avoid_: Horizontal flipping, aggressive elastic distortion.

**Nyquist Tear Resolution**:
The spatial sampling density threshold (minimum $336 \times 336\text{ px}$ across a 140 mm field of view, keeping patch tokens $\le 5.8\text{ mm}$) mathematically required to prevent focal 1–3 mm meniscal tears and ligament discontinuities from collapsing into sub-pixel blur under 14px ViT patch tokenization.
_Avoid_: Standard 126px/224px uniform downsampling for ViT backbones.

**Calibrated Soft Weak Supervision**:
Training supervision using continuous probability targets $[0.0, 1.0]$ derived from multi-lingual LLM extraction with conditional effusion-to-synovitis backoff, penalizing overconfident predictions on hedged clinical findings and providing smooth gradients.
_Avoid_: Hard 0/1 binary thresholding of extracted report weak labels.



## Dataset & Label Semantics

**Gold Study**:
A knee examination with authoritative ground-truth binary labels verified by a consensus of MSK subspecialty radiologists (58 studies in training set).
_Avoid_: Ground truth report, manual sample.

**Pseudo-Label Study**:
An unannotated knee examination where labels are inferred from multi-lingual radiology reports via calibrated LLM extraction (4,349 studies in training set).
_Avoid_: Silver study, unsupervised row.

**Silence Semantics**:
The policy governing how unmentioned findings in radiology reports are mapped: structural absence findings are assigned 0, Synovitis is soft-imputed, and acute/focal conditions remain unaddressed (loss-masked NaN).
_Avoid_: Zero-filling, global imputation.

**Synovitis Soft Target**:
A continuous pseudo-probability assigned to unmentioned Synovitis derived from the presence (0.63) or absence (0.22) of Joint Effusion.
_Avoid_: Binary guess, hard zero synovitis.

**Scanner Fingerprint**:
A composite identifier formed by hashing hardware and acquisition tags (Manufacturer, Model, Software, ImagingFrequency, Coil) to group studies by acquisition site.
_Avoid_: Patient ID, study folder hash.

**Out-of-Fold (OOF) Predictions**:
The validation logits accumulated from the best-epoch checkpoint of each cross-validation fold on its held-out studies. Only logits from Gold Studies are used to fit Platt temperature calibration, since Pseudo-Label Studies carry noisy continuous targets unsuitable for calibration fitting.
_Avoid_: Test predictions, held-out set logits, full-validation logits.

**Macro ROC-AUC**:
The unweighted arithmetic mean of per-label ROC-AUC scores across all 12 target labels. For Synovitis specifically, AUC is computed on Gold Study validation rows only (hard binary 0/1 targets); all other 11 labels are evaluated on all non-NaN validation rows. Labels with only one class present in the validation fold return NaN and are excluded from the mean.
_Avoid_: Micro-AUC, weighted AUC, softmax probability AUC.

**Efficiency Track**:
A Kaggle competition submission category evaluating model performance under strict runtime constraints (≤ 30 minutes total inference across ~1,300 test studies on Kaggle T4 GPUs).
_Avoid_: Main track 9-hour runtime, multi-hour unconstrained ensemble.

**Progressive Backbone Unfreezing**:
A staged fine-tuning regimen transitioning from frozen feature extraction (head warmup), through top semantic transformer blocks, to full-depth differential learning rates to protect pretrained foundation representations from overfitting on small cohorts.
_Avoid_: Instantaneous full-model unfreezing, static feature extraction.

