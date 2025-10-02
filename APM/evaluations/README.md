<h1>EVALUATION ON ALL MODELS</h1>

`cd APM/evaluations`

You should have the imagenet ILSVRC2012 validation set of images in a folder to point these tests to.

<h3> Baseline, Step Size, Upscaling Sensitivity Tests (Tables 1 and 2)</h3>

```

python3 ImageNetStepSizeScoreVarianceTest.py --test_type RISE --model VIT32 --image_count 1000 --dataset_path <path-to-imagenet-2012-validation> --cuda_num 0
python3 ImageNetStepSizeScoreVarianceTest.py --test_type POS_NEG_PERT --model VIT32 --image_count 1000 --dataset_path <path-to-imagenet-2012-validation> --cuda_num 0
python3 ImageNetStepSizeScoreVarianceTest.py --test_type MONO --model VIT32 --image_count 1000 --dataset_path <path-to-imagenet-2012-validation> --cuda_num 0
python3 ImageNetStepSizeScoreVarianceTest.py --test_type AIC --model VIT32 --image_count 1000 --dataset_path <path-to-imagenet-2012-validation> --cuda_num 0
python3 ImageNetStepSizeScoreVarianceTest.py --test_type DF --model VIT32 --image_count 1000 --dataset_path <path-to-imagenet-2012-validation> --cuda_num 0

python3 ImageNetStepSizeScoreVarianceTest.py --test_type RISE --model VIT16 --image_count 1000 --dataset_path <path-to-imagenet-2012-validation> --cuda_num 0
python3 ImageNetStepSizeScoreVarianceTest.py --test_type POS_NEG_PERT --model VIT16 --image_count 1000 --dataset_path <path-to-imagenet-2012-validation> --cuda_num 0
python3 ImageNetStepSizeScoreVarianceTest.py --test_type MONO --model VIT16 --image_count 1000 --dataset_path <path-to-imagenet-2012-validation> --cuda_num 0
python3 ImageNetStepSizeScoreVarianceTest.py --test_type AIC --model VIT16 --image_count 1000 --dataset_path <path-to-imagenet-2012-validation> --cuda_num 0
python3 ImageNetStepSizeScoreVarianceTest.py --test_type DF --model VIT16 --image_count 1000 --dataset_path <path-to-imagenet-2012-validation> --cuda_num 0



python3 ImageNetBaselineScoreVarianceTest.py --test_type RISE --model VIT32 --image_count 1000 --dataset_path <path-to-imagenet-2012-validation> --cuda_num 0
python3 ImageNetBaselineScoreVarianceTest.py --test_type POS_NEG_PERT --model VIT32 --image_count 1000 --dataset_path <path-to-imagenet-2012-validation> --cuda_num 0
python3 ImageNetBaselineScoreVarianceTest.py --test_type MONO --model VIT32 --image_count 1000 --dataset_path <path-to-imagenet-2012-validation> --cuda_num 0
python3 ImageNetBaselineScoreVarianceTest.py --test_type AIC --model VIT32 --image_count 1000 --dataset_path <path-to-imagenet-2012-validation> --cuda_num 0
python3 ImageNetBaselineScoreVarianceTest.py --test_type DF --model VIT32 --image_count 1000 --dataset_path <path-to-imagenet-2012-validation> --cuda_num 0

python3 ImageNetBaselineScoreVarianceTest.py --test_type RISE --model VIT16 --image_count 1000 --dataset_path <path-to-imagenet-2012-validation> --cuda_num 0
python3 ImageNetBaselineScoreVarianceTest.py --test_type POS_NEG_PERT --model VIT16 --image_count 1000 --dataset_path <path-to-imagenet-2012-validation> --cuda_num 0
python3 ImageNetBaselineScoreVarianceTest.py --test_type MONO --model VIT16 --image_count 1000 --dataset_path <path-to-imagenet-2012-validation> --cuda_num 0
python3 ImageNetBaselineScoreVarianceTest.py --test_type AIC --model VIT16 --image_count 1000 --dataset_path <path-to-imagenet-2012-validation> --cuda_num 0
python3 ImageNetBaselineScoreVarianceTest.py --test_type DF --model VIT16 --image_count 1000 --dataset_path <path-to-imagenet-2012-validation> --cuda_num 0



python3 ImageNetSmoothingScoreVarianceTest.py --test_type RISE --model VIT32 --image_count 1000 --dataset_path <path-to-imagenet-2012-validation> --cuda_num 0
python3 ImageNetSmoothingScoreVarianceTest.py --test_type POS_NEG_PERT --model VIT32 --image_count 1000 --dataset_path <path-to-imagenet-2012-validation> --cuda_num 0
python3 ImageNetSmoothingScoreVarianceTest.py --test_type MONO --model VIT32 --image_count 1000 --dataset_path <path-to-imagenet-2012-validation> --cuda_num 0
python3 ImageNetSmoothingScoreVarianceTest.py --test_type AIC --model VIT32 --image_count 1000 --dataset_path <path-to-imagenet-2012-validation> --cuda_num 0
python3 ImageNetSmoothingScoreVarianceTest.py --test_type DF --model VIT32 --image_count 1000 --dataset_path <path-to-imagenet-2012-validation> --cuda_num 0

python3 ImageNetSmoothingScoreVarianceTest.py --test_type RISE --model VIT16 --image_count 1000 --dataset_path <path-to-imagenet-2012-validation> --cuda_num 0
python3 ImageNetSmoothingScoreVarianceTest.py --test_type POS_NEG_PERT --model VIT16 --image_count 1000 --dataset_path <path-to-imagenet-2012-validation> --cuda_num 0
python3 ImageNetSmoothingScoreVarianceTest.py --test_type MONO --model VIT16 --image_count 1000 --dataset_path <path-to-imagenet-2012-validation> --cuda_num 0
python3 ImageNetSmoothingScoreVarianceTest.py --test_type AIC --model VIT16 --image_count 1000 --dataset_path <path-to-imagenet-2012-validation> --cuda_num 0
python3 ImageNetSmoothingScoreVarianceTest.py --test_type DF --model VIT16 --image_count 1000 --dataset_path <path-to-imagenet-2012-validation> --cuda_num 0

```


<h3>Insertion, Deletion, and MAS Insertion and Deletion Tests (Tables 3 and 4)</h3>

```
python3 MDATest.py --function GC --model VIT_base_16 --image_count 5000 --imagenet <path-to-imagenet-data> --gpu 0
python3 MDATest.py --function IG --model VIT_base_16 --image_count 5000 --imagenet <path-to-imagenet-data> --gpu 0
python3 MDATest.py --function LRP --model VIT_base_16 --image_count 5000 --imagenet <path-to-imagenet-data> --gpu 0
python3 MDATest.py --function VIT_CX --model VIT_base_16 --image_count 5000 --imagenet <path-to-imagenet-data> --gpu 0
python3 MDATest.py --function Bidirectional --model VIT_base_16 --image_count 5000 --imagenet <path-to-imagenet-data> --gpu 0
python3 MDATest.py --function Transition_attn --model VIT_base_16 --image_count 5000 --imagenet <path-to-imagenet-data> --gpu 0
python3 MDATest.py --function TIS --model VIT_base_16 --image_count 5000 --imagenet <path-to-imagenet-data> --gpu 0
python3 MDATest.py --function Calibrate --model VIT_base_16 --image_count 5000 --imagenet <path-to-imagenet-data> --gpu 0


python3 MDATest.py --function GC --model VIT_tiny_16 --image_count 5000 --imagenet <path-to-imagenet-data> --gpu 1
python3 MDATest.py --function IG --model VIT_tiny_16 --image_count 5000 --imagenet <path-to-imagenet-data> --gpu 1
python3 MDATest.py --function LRP --model VIT_tiny_16 --image_count 5000 --imagenet <path-to-imagenet-data> --gpu 1
python3 MDATest.py --function VIT_CX --model VIT_tiny_16 --image_count 5000 --imagenet <path-to-imagenet-data> --gpu 1
python3 MDATest.py --function Bidirectional --model VIT_tiny_16 --image_count 5000 --imagenet <path-to-imagenet-data> --gpu 2
python3 MDATest.py --function Transition_attn --model VIT_tiny_16 --image_count 5000 --imagenet <path-to-imagenet-data> --gpu 2
python3 MDATest.py --function TIS --model VIT_tiny_16 --image_count 5000 --imagenet <path-to-imagenet-data> --gpu 2
python3 MDATest.py --function Calibrate --model VIT_tiny_16 --image_count 5000 --imagenet <path-to-imagenet-data> --gpu 2


python3 MDATest.py --function GC --model VIT_base_32 --image_count 5000 --imagenet <path-to-imagenet-data> --gpu 3
python3 MDATest.py --function IG --model VIT_base_32 --image_count 5000 --imagenet <path-to-imagenet-data> --gpu 3
python3 MDATest.py --function LRP --model VIT_base_32 --image_count 5000 --imagenet <path-to-imagenet-data> --gpu 3
python3 MDATest.py --function VIT_CX --model VIT_base_32 --image_count 5000 --imagenet <path-to-imagenet-data> --gpu 3
python3 MDATest.py --function Bidirectional --model VIT_base_32 --image_count 5000 --imagenet <path-to-imagenet-data> --gpu 3
python3 MDATest.py --function Transition_attn --model VIT_base_32 --image_count 5000 --imagenet <path-to-imagenet-data> --gpu 3
python3 MDATest.py --function TIS --model VIT_tiny_16 --image_count 5000 --imagenet <path-to-imagenet-data> --gpu 2
python3 MDATest.py --function Calibrate --model VIT_tiny_16 --image_count 5000 --imagenet <path-to-imagenet-data> --gpu 2

```


<h3> Embedding Distance Tests (Table 5 and Figures 5 and A.7)</h3>

```
python3 ImageNetValEmbeddingDistance.py --image_count 1000 --model VIT32 --test_type RISE --dataset_path <path-to-imagenet-2012-validation> --cuda_num 0
python3 ImageNetValEmbeddingDistance.py --image_count 1000 --model VIT32 --test_type RISE_VIT --dataset_path <path-to-imagenet-2012-validation> --cuda_num 0

python3 ImageNetValEmbeddingDistance.py --image_count 1000 --model VIT16 --test_type RISE --dataset_path <path-to-imagenet-2012-validation> --cuda_num 0
python3 ImageNetValEmbeddingDistance.py --image_count 1000 --model VIT16 --test_type RISE_VIT --dataset_path <path-to-imagenet-2012-validation> --cuda_num 0
```

<h3> Qualitative Baseline Comparisons Tests </h3>

```
python3 baselineOrderComp.py --image_count 1000 --model VIT32 --dataset_path <path-to-imagenet-2012-validation> --cuda_num 0
python3 baselineOrderComp.py --image_count 1000 --model VIT16 --dataset_path <path-to-imagenet-2012-validation> --cuda_num 0
```

<h3> Equivalence and Runtime Tests (Tables B.6 and B.7) </h3>

```
python3 speedAndEquivalenceTest.py --image_count 1000 --model VIT32 --dataset_path <path-to-imagenet-2012-validation> --cuda_num 0
python3 speedAndEquivalenceTest.py --image_count 1000 --model VIT16 --dataset_path <path-to-imagenet-2012-validation> --cuda_num 0
```
