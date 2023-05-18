# %% 
'''
check test_data results of unet and fourinets
first row: diffused image
second row: ground truth
third row: unet reconstruction
fourth row: fouriernet reconstruction
'''
import torch
import os
import numpy as np
import matplotlib.pyplot as plt
import matplotlib.gridspec as gridspec

fig = plt.figure(figsize=(320, 210))
gs = gridspec.GridSpec(4, 6,wspace=0.05, hspace=0)

os.chdir('/home/lihaiyue/data/snapshotscope/replicate/exp/unet/test')

for i in range(6):
    
    frecon=f'recon{i}.pt'
    fsam=f'sam{i}.pt'
    fim=f'im{i}.pt'

    recon=torch.load(frecon)
    recon=recon[0].permute(1, 2, 0)
    sam=torch.load(fsam);
    sam=sam[0].permute(1, 2, 0)
    im=torch.load(fim)
    im=im[0].permute(1,2,0)

    ax = fig.add_subplot(gs[i])
    ax.imshow(im.numpy())
    ax.set_xticks([])
    ax.set_yticks([])

    ax = fig.add_subplot(gs[6+i])
    ax.imshow(sam.numpy())
    ax.set_xticks([])
    ax.set_yticks([])

    ax = fig.add_subplot(gs[12+i])
    ax.imshow(recon.numpy())
    ax.set_xticks([])
    ax.set_yticks([])

os.chdir('/home/lihaiyue/data/snapshotscope/replicate/exp/fouriernet_mse_lpips/test')

for i in range(6):
    
    frecon=f'recon{i}.pt'

    recon=torch.load(frecon)
    recon=recon[0].permute(1, 2, 0)
    
    ax = fig.add_subplot(gs[18+i])
    ax.imshow(recon.numpy())
    ax.set_xticks([])
    ax.set_yticks([])


# %% plot train_data losses  [mse+lpips]
import torch
import os
import numpy as np
import matplotlib.pyplot as plt

#TODO:change dir path
os.chdir('/home/lihaiyue/data/snapshotscope/replicate/exp/unet')

latest=torch.load('latest.pt')
mse_loss=latest['mses']
lpips_loss=latest['lpips_losses']
loss=latest['losses']

# 指定卷积核大小和权重
window_size = 10
weights = np.repeat(1.0, window_size) / window_size

# 进行卷积计算得到滑动平均值
mse_smooth_loss = np.convolve(mse_loss, weights, 'valid')
lpips_smooth_loss = np.convolve(lpips_loss, weights, 'valid')
smooth_loss = np.convolve(loss, weights, 'valid')

# 绘制图像
plt.figure()
#plt.subplot(311)

plt.plot(mse_smooth_loss)
plt.title('mse_smooth_loss')
plt.xlabel('iteration')
plt.axis([0,10000,0,0.05])   #TODO:change the range

plt.subplot(312)
plt.plot(lpips_smooth_loss)
plt.xlabel('iteration')
#plt.axis([0,10000,0,0.5])   #TODO:change the range
plt.title('lpips_smooth_loss')

#plt.subplot(313)
plt.plot(smooth_loss)
plt.xlabel('iteration')
plt.axis([6000,10000,0,0.01])   #TODO:change the range
plt.title('smooth_loss')


plt.subplots_adjust(top=0.92, bottom=0.08, left=0.10, right=0.95, hspace=0.65,
                    wspace=0.35)

plt.show()
