# Imports
import numpy as np
from skimage import io
import os

# Hyperparameters
INPUT_PATH = 'input_data'
OUTPUT_PATH = 'output_data'

# Main function
def create_kaleidoscopes():

    # (1.0) Check input data folder existence
    if not os.path.exists(INPUT_PATH):
        print(f'Error: Input folder {INPUT_PATH} not found')
        input('\nPress Enter to exit...')
        return
    
    # (1.1) Create output data folder if it doesn't exist
    if not os.path.exists(OUTPUT_PATH):
        os.makedirs(OUTPUT_PATH)
        print(f'Created output folder: {OUTPUT_PATH}')

    # (2) Iterate through all files
    for file_name in os.listdir(INPUT_PATH):
        print(f'⚙️  Processing file: {file_name}')
        print(os.listdir(INPUT_PATH))

        try:
            file_path = os.path.join(INPUT_PATH, file_name)
            img = io.imread(file_path)  

            # (2.1) Parameters
            height, width, _ = img.shape
            half_height, half_width = height // 2, width // 2
            name, extension = os.path.splitext(file_name)

            # (2.2) Initialize base images
            img_2_u = img[:half_height, :, :]
            img_2_d = img[half_height:, : ,:]
            img_2_r = img[:, half_width:, :]
            img_2_l = img[:, :half_width, :]

            img_4_ul = img[:half_height, :half_width, :]
            img_4_ur = img[:half_height, half_width:, :]
            img_4_dl = img[half_height:, :half_width, :]
            img_4_dr = img[half_height:, half_width:, :]

            # (2.3) Create kaleidoscopes
            img_2_u_result = np.concatenate([img_2_u, img_2_u[::-1,:,:]] , axis=0)
            img_2_d_result = np.concatenate([img_2_d[::-1,:,:], img_2_d] , axis=0)
            img_2_r_result = np.concatenate([img_2_r[:,::-1,:], img_2_r] , axis=1)
            img_2_l_result = np.concatenate([img_2_l, img_2_l[:,::-1,:]] , axis=1)
            
            img_4_ul_result = np.concatenate([np.concatenate([img_4_ul[:,:,:],img_4_ul[::-1,:,:]] , axis=0),\
                                 np.concatenate([img_4_ul[:,::-1,:],img_4_ul[::-1,::-1,:]] , axis=0)], axis=1)
            img_4_ur_result = np.concatenate([np.concatenate([img_4_ur[:,::-1,:],img_4_ur[::-1,::-1,:]] , axis=0),\
                                 np.concatenate([img_4_ur[:,:,:],img_4_ur[::-1,:,:]] , axis=0)], axis=1)
            img_4_dl_result = np.concatenate([np.concatenate([img_4_dl[::-1,:,:],img_4_dl[:,:,:]] , axis=0),\
                                 np.concatenate([img_4_dl[::-1,::-1,:],img_4_dl[:,::-1,:]] , axis=0)], axis=1)
            img_4_dr_result = np.concatenate([np.concatenate([img_4_dr[::-1,::-1,:],img_4_dr[:,::-1,:]] , axis=0),\
                                 np.concatenate([img_4_dr[::-1,:,:],img_4_dr[:,:,:]] , axis=0)], axis=1)

            # (2.4) Save Results
            io.imsave(f"{OUTPUT_PATH}/{name}_2_U{extension}", img_2_u_result)
            io.imsave(f"{OUTPUT_PATH}/{name}_2_D{extension}", img_2_d_result)
            io.imsave(f"{OUTPUT_PATH}/{name}_2_L{extension}", img_2_l_result)
            io.imsave(f"{OUTPUT_PATH}/{name}_2_R{extension}", img_2_r_result)

            io.imsave(f"{OUTPUT_PATH}/{name}_4_UL{extension}", img_4_ul_result)
            io.imsave(f"{OUTPUT_PATH}/{name}_4_UR{extension}", img_4_ur_result)
            io.imsave(f"{OUTPUT_PATH}/{name}_4_DL{extension}", img_4_dl_result)
            io.imsave(f"{OUTPUT_PATH}/{name}_4_DR{extension}", img_4_dr_result)

        except Exception as e:
            print(f'Error processing {file_name}: {e}')    

    # (3) Happy End
    print('✅ Image processing is completed!')
    input('\nPress Enter to exit...')
    return 

if __name__ == '__main__':
    create_kaleidoscopes()

print('Success')