# Unit 3 Plan — Materials Genomics

## Unit title
Feature Engineering for Composition, Structure, and Process Data (Database-to-Descriptor Bridge)

## Note
This scaffold is stored in `03_materials_databases/` to preserve existing repo structure; content focus follows the SS26 Unit-3 plan.

## Learning objectives
1. Build descriptor sets from composition, structure, and process metadata.
2. Explain invariance, scaling, and leakage constraints in descriptor engineering.
3. Compare classical descriptors vs learned representations at a conceptual level.
4. Select minimal, defensible feature baselines for downstream property models.
5. Document provenance and quality checks for reproducible featurization.

## 90-minute structure
- 0–10: recap + role of features in discovery pipeline
- 10–30: descriptor families (composition, structural, process)
- 30–50: invariances, normalization, missingness handling
- 50–70: feature selection/reduction and information leakage traps
- 70–85: baseline featurization workflow and case sketch
- 85–90: summary + exercise handoff

## Exercise (90 min)
- build a feature table from a curated materials subset
- apply normalization + one reduction method
- test grouped split leakage sensitivity
- report top informative features with caveats
