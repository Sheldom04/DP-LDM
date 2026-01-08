## Dual-flow channel compensation and semantic diffusion guidance for underwater image enhancement

![](image/Figure_1.png)

### Getting Started
- Install the relevant packages according to the instructions.
#### Test
- Download the model weights we provide: [DP-LDM](https://pan.baidu.com/s/1nn1THtjRTU1eAybywtzSpw?pwd=b7eb)
- Place "vae_UIEBD.pth" in the "fine_pt/vae" folder.
- Place "DP-LDM_UIEBD_opt.pth" and "DP-LDM_UIEBD_gen.pth" in the "pth" folder.
- Modify the "dataroot" field in the "underwater.json" file to the directory of the images that need to be enhanced.
```bash
    python infer.py
```
- The enhanced results are displayed in "experiments_train".
#### Train
- First, train the VAE:
```bash
    python my_ldm_concat_cond/train_vae.py
```
- Second, train the DP-LDM:
```bash
    python latent_train.py
```
- Finally, fine-tune the weights of the VAE.
```bash
    python vae_fine_tuning.py
```
## Acknowledgement
Thank you to DM-Water for providing the source code, and to the authors of LSUI and UIEBD for providing the datasets.