<p align="center">
  <h1 align="center">Learning Commonality, Divergence and Variety for Unsupervised Visible-Infrared Person Re-identification</h1>
</p>

<p align="center">
  <a href="https://img.shields.io/badge/Conference-NeurIPS%202024-4b8bbe?style=for-the-badge" target="_blank"><img src="https://img.shields.io/badge/Conference-NeurIPS%202024-4b8bbe?style=for-the-badge"></a>
  <a href="https://img.shields.io/badge/Framework-PyTorch-ee4c2c?style=for-the-badge&logo=pytorch" target="_blank"><img src="https://img.shields.io/badge/Framework-PyTorch-ee4c2c?style=for-the-badge&logo=pytorch"></a>
  <a href="https://img.shields.io/badge/Python-3.8.13-3776AB?style=for-the-badge&logo=python" target="_blank"><img src="https://img.shields.io/badge/Python-3.8.13-3776AB?style=for-the-badge&logo=python"></a>
</p>

<p align="center">
  <a href="https://scholar.google.com/citations?user=Go9q2jsAAAAJ&hl=zh-CN&oi=sra" target="_blank"><strong>Jiangming Shi*</strong></a><sup>1,2</sup> ·
  <a href="https://scholar.google.com/citations?hl=zh-CN&pli=1&user=H1rqfM4AAAAJ" target="_blank"><strong>Xiangbo Yin*</strong></a><sup>2</sup> ·
  <a href="https://scholar.google.com/citations?user=a-I8c8EAAAAJ&hl=zh-CN&oi=sra" target="_blank"><strong>Yachao Zhang</strong></a><sup>3</sup> ·
  <a href="https://scholar.google.com/citations?user=CXZciFAAAAAJ&hl=zh-CN&oi=sra" target="_blank"><strong>Zhizhong Zhang</strong></a><sup>3</sup> ·
  <a href="https://scholar.google.com/citations?user=RN1QMPgAAAAJ&hl=zh-CN&oi=sra" target="_blank"><strong>Yuan Xie†</strong></a><sup>1,3</sup> ·
  <a href="https://scholar.google.com/citations?user=idiP90sAAAAJ&hl=zh-CN&oi=sra" target="_blank"><strong>Yanyun Qu†</strong></a><sup>2</sup>
</p>

<p align="center">
  <sup>1</sup>Shanghai Innovation Institute, Shanghai, China <br>
  <sup>2</sup>Xiamen University, Xiamen, China <br>
  <sup>3</sup>East China Normal University, Shanghai, China
</p>

<p align="center">
  <a href="https://arxiv.org/pdf/2402.19026" target="_blank">📄 <strong>Paper Link (arXiv:2402.19026)</strong></a><br>
  <em>Accepted at NeurIPS 2024</em>
</p>

<p align="center">
  <img src="figs/framework.png" alt="Framework" width="80%">
</p>

---

## 🚀 Introduction
This repository provides the **official PyTorch implementation** of  
**“Learning Commonality, Divergence and Variety for Unsupervised Visible-Infrared Person Re-identification”**,  
accepted at **NeurIPS 2024**.

---

## 🧩 Requirements
- Python 3.8.13  
- PyTorch 1.8.0  
- Torchvision 0.9.0  
- scikit-learn 1.2.2  

Install dependencies:
```bash
pip install -r requirements.txt
```

---

## 📂 Dataset Preparation
Prepare the datasets by running:

```bash
python prepare_sysu.py   # for SYSU-MM01
python prepare_regdb.py  # for RegDB
```

> ⚠️ **Note:** Modify the dataset paths in `prepare_sysu.py` and `prepare_regdb.py` to match your local environment.

---

## 🏋️‍♂️ Training
Train the model using:

```bash
# For SYSU-MM01
sh run_train_sysu.sh

# For RegDB
sh run_train_regdb.sh
```

You can adjust hyperparameters (batch size, learning rate, epochs, etc.) in the corresponding `.sh` or `.py` scripts.

---

## 🧪 Testing
Evaluate the trained model using:

```bash
# For SYSU-MM01
sh test_sysu.sh

# For RegDB
sh test_regdb.sh
```

The testing script loads the trained checkpoint and computes metrics such as **Rank-1** and **mAP**.

---

## 🔧 Common Issues
| Problem | Solution |
|----------|-----------|
| CUDA version mismatch | Ensure your PyTorch and CUDA versions are compatible. |
| Out of memory | Reduce batch size or enable mixed precision (`torch.cuda.amp`). |
| Dataset path error | Check your dataset root paths in scripts. |
| Dependency conflict | Use a fresh virtual environment for reproducibility. |

---

## AG-ReID / LAGPeR RAHP and CESA experiments

All four formal ablations use `CMhybrid` in Stage 1 and disable training-time test evaluation. Checkpoints follow a fixed final epoch policy; use the Stage 2 `model_final.pth.tar` for final evaluation. Training and testing require the same `resnet50-19c8e357.pth` through `PRETRAINED_RESNET50`, `--pretrained-resnet50`, or `PCLHD_RESNET50_PRETRAINED`.

The AG-ReID `DATA_DIR` is the shared root containing
`aerial_modify/{trial}/bounding_box_train`,
`ground_modify/{trial}/bounding_box_train`, and `idx/test_aerial_{trial}.txt`
plus `idx/test_ground_{trial}.txt`. Do not append `agreid_ir` or `agreid_rgb`.

```bash
# Frozen Baseline / RAHP / CESA / Full suites
DATA_DIR=/path/to/AG-ReID-root LOGS_DIR=logs/agreid PRETRAINED_RESNET50=/path/to/resnet50-19c8e357.pth bash scripts/run_agreid_ablation.sh
DATA_DIR=/path/to/LAGPeR-root LOGS_DIR=logs/lagper PRETRAINED_RESNET50=/path/to/resnet50-19c8e357.pth bash scripts/run_lagper_ablation.sh

# Separate final evaluation
python test_agreid.py --data-dir /path/to/AG-ReID-root --checkpoint /path/to/stage2/model_final.pth.tar --pretrained-resnet50 /path/to/resnet50-19c8e357.pth --trial 1 --batch-size 64 --workers 8
python test_LAG.py --data-dir /path/to/LAGPeR-root --checkpoint /path/to/stage2/model_final.pth.tar --pretrained-resnet50 /path/to/resnet50-19c8e357.pth --trial 1 --batch-size 64 --workers 8
```

The ablation scripts use `eps=0.6` by default. Override the shared value for
all four groups from the command line, for example `EPS=0.55 LOGS_DIR=logs/agreid_eps055 bash scripts/run_agreid_ablation.sh`. Direct training entrypoints also accept `--eps 0.55`.

LAGPeR requires an externally prepared official four-scene train / three-scene test manifest tree. The deprecated `prepare_lag.py` cannot create it from AG-ReID. See the technical design for the exact contract.

See `docs/rahp-cesa-technical-design.md` for the frozen method definitions and protocol details.

---

## 📚 Citation
If our work is helpful for your research, please consider citing:

```bibtex
@article{shi2024progressive,
  title={Progressive Contrastive Learning with Multi-Prototype for Unsupervised Visible-Infrared Person Re-identification},
  author={Shi, Jiangming and Yin, Xiangbo and Wang, Yaoxing and Liu, Xiaofeng and Xie, Yuan and Qu, Yanyun},
  journal={arXiv preprint arXiv:2402.19026},
  year={2024}
}

@article{yin2024robust,
  title={Robust Pseudo-label Learning with Neighbor Relation for Unsupervised Visible-Infrared Person Re-Identification},
  author={Yin, Xiangbo and Shi, Jiangming and Zhang, Yachao and Lu, Yang and Zhang, Zhizhong and Xie, Yuan and Qu, Yanyun},
  journal={arXiv preprint arXiv:2405.05613},
  year={2024}
}

@article{shi2024multi,
  title={Multi-Memory Matching for Unsupervised Visible-Infrared Person Re-Identification},
  author={Shi, Jiangming and Yin, Xiangbo and Chen, Yeyun and Zhang, Yachao and Zhang, Zhizhong and Xie, Yuan and Qu, Yanyun},
  journal={arXiv preprint arXiv:2401.06825},
  year={2024}
}

@inproceedings{shi2023dpis,
  title={Dual pseudo-labels interactive self-training for semi-supervised visible-infrared person re-identification},
  author={Shi, Jiangming and Zhang, Yachao and Yin, Xiangbo and Xie, Yuan and Zhang, Zhizhong and Fan, Jianping and Shi, Zhongchao and Qu, Yanyun},
  booktitle={Proceedings of the IEEE/CVF International Conference on Computer Vision},
  pages={11218--11228},
  year={2023}
}
```

---

## 📬 Contact
For questions or collaboration, please contact:  
📧 **xiangboyin@stu.xmu.edu.cn**  
📧 **jiangming.shi@outlook.com**

---

<p align="center">
  ⭐ If you find this project useful, please give it a star!
</p>
