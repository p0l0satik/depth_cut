import sys
import os
import subprocess


if __name__ == "__main__":
    target = sys.argv[1]
    path_to_toolkit = sys.argv[2]
    path_to_u2net = sys.argv[3]
    final_folder = target +'_TUM'
    
    # os.mkdir(final_folder)

    # subprocess.run(["./" + path_to_toolkit + 'local_extract/local_extract.sh', target])
    # subprocess.run(["python3", "convert_to_tum.py", target, final_folder])
    # os.mkdir(os.path.join(final_folder, 'mask'))
    # subprocess.run(["python3", path_to_u2net + "u2net_human_seg_test.py", path_to_u2net, final_folder + "/rgb/", final_folder + "/mask/"])

    # os.mkdir(os.path.join(final_folder, 'mask_cut'))

    subprocess.run(["python3", "depth_test.py", final_folder])
