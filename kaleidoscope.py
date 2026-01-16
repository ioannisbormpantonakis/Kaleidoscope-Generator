# Imports

from skimage import io
import matplotlib.pyplot as plt
import numpy as np
import os

# Hyperparameters

INPUT_PATH = 'input_data'
OUTPUT_PATH = 'output_data'

def create_kaleidoscopes():

    # (1) Check data folder existence
    if not os.path.exists(INPUT_PATH):
        print(f'Error: Input folder {INPUT_PATH} not found')
        input('\nPress Enter to exit...')
        return
    
    if not os.path.exists(OUTPUT_PATH):
        os.makedirs(OUTPUT_PATH)
        print(f'Created output folder: {OUTPUT_PATH}')  

    # (2) Iterate through all jpg files
    for file_name in os.listdir(INPUT_PATH):

        ### DEBUG ###
        print(f'⚙️  Processing file: {file_name}')
        ### DEBUG ###

        try:
            file_path = os.path.join(INPUT_PATH, file_name)
            img = io.imread(file_path)

            # (2.1) Parameters
            height, width, _ = img.shape
            cx, cy = width // 2, height // 2
            name, _ = os.path.splitext(file_name)

            # (2.2) Initializing base images
            img_u2 = img[:cy, :, :]
            img_d2 = img[cy:, : ,:] 
            img_r2 = img[:, cx:, :]
            img_l2 = img[:, :cx, :]

            img_ul4 = img[:cy, :cx, :]
            img_ur4 = img[:cy, cx:, :] 
            img_dl4 = img[cy:, :cx, :]
            img_dr4 = img[cy:, cx:, :]

            # (2.3) Creating kaleidoscopes
            img_u2_result = np.concatenate([img_u2, img_u2[::-1,:,:]] , axis=0)
            img_d2_result = np.concatenate([img_d2[::-1,:,:], img_d2] , axis=0)
            img_r2_result = np.concatenate([img_r2[:,::-1,:], img_r2] , axis=1)
            img_l2_result = np.concatenate([img_l2, img_l2[:,::-1,:]] , axis=1)
            
            img_ul4_result = np.concatenate([np.concatenate([img_ul4[:,:,:],img_ul4[::-1,:,:]] , axis=0),\
                                 np.concatenate([img_ul4[:,::-1,:],img_ul4[::-1,::-1,:]] , axis=0)], axis=1)
            img_ur4_result = np.concatenate([np.concatenate([img_ur4[:,::-1,:],img_ur4[::-1,::-1,:]] , axis=0),\
                                 np.concatenate([img_ur4[:,:,:],img_ur4[::-1,:,:]] , axis=0)], axis=1)
            img_dl4_result = np.concatenate([np.concatenate([img_dl4[::-1,:,:],img_dl4[:,:,:]] , axis=0),\
                                 np.concatenate([img_dl4[::-1,::-1,:],img_dl4[:,::-1,:]] , axis=0)], axis=1)
            img_dr4_result = np.concatenate([np.concatenate([img_dr4[::-1,::-1,:],img_dr4[:,::-1,:]] , axis=0),\
                                 np.concatenate([img_dr4[::-1,:,:],img_dr4[:,:,:]] , axis=0)], axis=1)
            
            # (2.4) Saving Results
            plt.imsave(f"{OUTPUT_PATH}/{name}_U2.jpg", img_u2_result)
            plt.imsave(f"{OUTPUT_PATH}/{name}_D2.jpg", img_d2_result)
            plt.imsave(f"{OUTPUT_PATH}/{name}_L2.jpg", img_l2_result)
            plt.imsave(f"{OUTPUT_PATH}/{name}_R2.jpg", img_r2_result)

            plt.imsave(f"{OUTPUT_PATH}/{name}_UL4.jpg", img_ul4_result)
            plt.imsave(f"{OUTPUT_PATH}/{name}_UR4.jpg", img_ur4_result)
            plt.imsave(f"{OUTPUT_PATH}/{name}_DL4.jpg", img_dl4_result)
            plt.imsave(f"{OUTPUT_PATH}/{name}_DR4.jpg", img_dr4_result)

        except Exception as e:
            print(f'Error processing {file_name}: {e}')

    # (3) The end
    print('✅ Image processing is completed!')
    input('\nPress Enter to exit...')
    return 

if __name__ == '__main__':
    create_kaleidoscopes()