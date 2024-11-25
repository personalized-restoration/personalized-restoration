
## Personalized Restoration via Dual-Pivot Tuning
To train our personalized model for diffusion-based image restoration, we build on top of two codebases: (1) HuggingFace diffusers (for DreamBooth) and DiffBIR (https://github.com/XPixelGroup/DiffBIR), the base model we use. In this codebase, we provide a set of train and test images for a particular identity, that can be used to try our method in the directory `trial_data' (we supply training images, sample degraded test images, and reference images for the degraded images). This data is from the CelebRef-HQ dataset (https://github.com/csxmli2016/DMDNet).

## Step 1: Out of context textual pivoting
Please follow the steps for dreambooth training using HuggingFace diffusers (https://huggingface.co/docs/diffusers/training/dreambooth). We provide the command below, that includes our training configuration, number of steps and so on.


```shell
  accelerate launch train_dreambooth.py \
  --pretrained_model_name_or_path="stabilityai/stable-diffusion-2-1-base"  \
  --instance_data_dir="/path/to/train/images" \
  --class_data_dir="/path/for/prior/preserving/class" \
  --output_dir="/path/for/dreambooth/model" \
  --with_prior_preservation --prior_loss_weight=1.0 \
  --instance_prompt="a photo of a sks man" \
  --class_prompt="a photo of a man" \
  --resolution=512 \
  --train_batch_size=1 \
  --gradient_accumulation_steps=2 --gradient_checkpointing \
  --use_8bit_adam \
  --learning_rate=1e-6 \
  --lr_scheduler="constant" \
  --lr_warmup_steps=0 \
  --num_class_images=200 \
  --max_train_steps=2500
```

## Step 2: Setting up the DiffBIR environment
Please follow the instructions at (https://github.com/XPixelGroup/DiffBIR) to set up the necessary dependencies for our method. Please also download the following pretrained models:

Stable Diffusion v2.1
```shell
wget https://huggingface.co/stabilityai/stable-diffusion-2-1-base/resolve/main/v2-1_512-ema-pruned.ckpt --no-check-certificate
```

face_swinir_v1.ckpt
```shell
wget https://huggingface.co/lxq007/DiffBIR/resolve/main/face_swinir_v1.ckpt /to/destination/path
```

face_full_v1.ckpt
```shell
wget https://huggingface.co/lxq007/DiffBIR/resolve/main/face_full_v1.ckpt /to/destination/path
```
Additionally, rename the above model to face_full_v1_og.ckpt.

## Step 3: Converting textual pivot model to be compatible with DiffBIR
First, we convert the checkpoint from the diffusers format to a .ckpt, using the following command:
```shell
python conversion_scripts/convert_diffusers_to_sd.py --model_path /path/for/dreambooth/model --checkpoint_path /path/for/ckpt
```
Next, we correct the parameter names in the model file to be consistent with the DiffBIR framework. For that, use the following command:
```shell
python conversion_scripts/rename_weights.py
```
Remember to add necessary paths in lines 4, 5, 60.

Then, we will combine the weight we create in the previous step with the DiffBIR encoder, to provide us with our starting model checkpoint on which we personalize. Please run the following command:
```shell
python conversion_scripts/make_stage2_init_weight-orig.py \
--cldm_config configs/model/cldm.yaml \
--sd_weight /path/to/corrected/model/from/previous/step \
--swinir_weight weights/face_swinir_v1.ckpt \
--full_weight weights/face_full_v1_og.ckpt \
--output /desired/output/path/for/combined/model
```

## Step 4: Model-based pivoting
First, you need to create a file list of training set and validation set. Please follow guidelines from the DiffBIR repository, under the heading ``Data Preparation: 1.Generate file list of training set and validation set'' in their repository.

Next, update the paths for this train and validation list in configs/dataset/face_train.yaml and configs/dataset/face_val.yaml, in line 5.

Next, update the path to your combined model (from the end of Step 3) in configs/train_cldm.yaml, line 14.

Now, you can train the model, using the following command:
```shell
python train.py --config configs/train_cldm.yaml
```

The model will be saved in the logs directory. Post training, you will have to copy your desired model into the directory ./weights and rename it to face_full_v1.ckpt. We find that training for 800 steps is sufficient and training can be stopped after that.


## Step 5: Inference
You can directly run the following command:
```shell
python inference_face.py \
--input /path/to/degraded/images \
--sr_scale 1 \
--output /desired/destination/path \
--has_aligned \
--device cuda
```
Use the sample images we provide as part of this codebase.