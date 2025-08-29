# Moge2_VO
# MoGe2 Visual Odometry (MoGe2_VO)

This repository implements a simple **Visual Odometry (VO)** pipeline built on top of **MoGe2** monocular depth predictions.  
It generates metric point maps using MoGe2, aligns consecutive frames with **RANSAC + Open3D ICP**, and reconstructs a trajectory.

---

## 📦 Setup

### 1. Clone this repo
```bash
git clone https://github.com/Smshah30/Moge2_VO.git
cd Moge2_VO
```

### 2. Install MoGe2
Clone and set up the [MoGe2](https://github.com/microsoft/MoGe) repository following its installation instructions.  
Make sure you can run inference with MoGe2 and generate point maps.

```bash
pip install git+https://github.com/microsoft/MoGe.git
```

#### Or clone repository
```bash
git clone https://github.com/microsoft/MoGe.git
cd MoGe
pip install -r requirements.txt   # install the requirements
```
### 3. Download Dataset (KITTI Odometry or Any Of Your Choice)
### 4. Requirements
```bash
pip install open3d
pip install yt-dlp # To download youtube Videos
sudo apt-get install ffmpeg # Necessary before install videos
```

## Usage

### 1. Create `.npz` files for each image
It stores the metric points, affine-inv points, normals, depth, mask and normalized intrinsics

```bash
python3 test_k.py --data_dir ~/datasets/kitti/dataset/sequences/00/image_0/ --ext png --save_path ~/numpies/ --max_images 30 # max_images for a quick run
```

### 2. Estimate poses with RANSAC + ICP

```bash
python3 ransac_open3d_icp_vo.py  --seq_dir ~/numpies/  --out_poses poses/pose.txt --levels 3 --iters 10 7 5 --subsample 8 --max_corr 20000 --use_first_K
```
`--use_first_K` is useful when the standard deviation (std) in scale at the end of step 1 is high. This ensures that the model keeps the Intrinsics Constant

### 3. Compare Results or View Trajectory
```bash
python3 eval_kitti.py  --gt_poses ~/kitti/dataset/poses/00.txt --est_poses poses_v2.txt --save_plot vid2_est.png
```
`--gt_poses` is optional 





