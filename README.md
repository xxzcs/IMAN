1、Paper Name

"IMAN: An Iterative Mutual-Aid Network for Breast Lesion Segmentation on Multi-modal Ultrasound Images"

2、Environment

Please prepare an environment with cuda 11.0 and python=3.8, 

Install the segmentation library by: pip install segmentation-models-pytorch==0.2.0,

More environment variables are requested in Requirements.

3、Dataset

Our dataset mainly consists of 169 US-CEUS samples, readers can find some samples in direcory 'datasets'

4、Train & Test

Train: 


python 5_crossitercopy.py  --batch-size=16 --model_name=TransUnet --data-path=datasets/us_small --lr=1e-4 --min-lr=5e-5 --epochs=300 --task=crossiter --criterion_name=DiceLoss --use_mmg --total_rounds=10 --ceus_resume='path of model trained on CEUS modality' --us_resume='path of model trained on CEUS modality' --suffix='An optional parameter that identifies the training process'

Test:


CEUS modality: python 5_infer.py --data-path=datasets/ceus_frames --test --resume='path of model trained on CEUS modality' --modality=CEUS
US modality: python 5_infer.py --data-path=datasets/us_small --test --resume='path of model trained on US modality' --modality=US