# SRRFN for train
# cd Train/



## Using --ext sep_reset argument on your first running.
## You can skip the decoding part and use saved binaries with --ext sep argument in second time.
## If you have enough memory, using --ext bin.

# SRRFN x2  LR: 48 * 48  HR: 96 * 96
python main.py --template SRRFN --save SRRFN_X2 --scale 2 --reset --save_results --patch_size 96 --ext sep_reset

# SRRFN x3  LR: 48 * 48  HR: 144 * 144
python main.py --template SRRFN --save SRRFN_X3 --scale 3 --reset --save_results --patch_size 144 --ext sep_reset

# SRRFN x4  LR: 48 * 48  HR: 192 * 192
python main.py --template SRRFN --save SRRFN_X4 --scale 4 --reset --save_results --patch_size 192 --ext sep_reset




# SRRFN for test
# cd Test/code/


# BI model

#SRRFN x2
python main.py --data_test MyImage --scale 2 --model SRRFN --degradation BI --pre_train ../model/SRRFN_x2_BI.pt --test_only --save_results --chop --save "SRRFN" --testpath ../LR/LRBI --testset Set5

#SRRFN+ x2
python main.py --data_test MyImage --scale 2 --model SRRFN --degradation BI --pre_train ../model/SRRFN_x2_BI.pt --test_only --save_results --chop --self_ensemble --save "SRRFN_plus" --testpath ../LR/LRBI --testset Set5


#SRRFN x3
python main.py --data_test MyImage --scale 3 --model SRRFN --degradation BI --pre_train ../model/SRRFN_x3_BI.pt --test_only --save_results --chop --save "SRRFN" --testpath ../LR/LRBI --testset Set5

#SRRFN+ x3
python main.py --data_test MyImage --scale 3 --model SRRFN --degradation BI --pre_train ../model/SRRFN_x3_BI.pt --test_only --save_results --chop --self_ensemble --save "SRRFN_plus" --testpath ../LR/LRBI --testset Set5


#SRRFN x4
python main.py --data_test MyImage --scale 4 --model SRRFN --degradation BI --pre_train ../model/SRRFN_x4_BI.pt --test_only --save_results --chop --save "SRRFN" --testpath ../LR/LRBI --testset Set5

#SRRFN+ x4
python main.py --data_test MyImage --scale 4 --model SRRFN --degradation BI --pre_train ../model/SRRFN_x4_BI.pt --test_only --save_results --chop --self_ensemble --save "SRRFN_plus" --testpath ../LR/LRBI --testset Set5


# BD model

#SRRFN_BD x3
python main.py --data_test MyImage --scale 3 --model SRRFN --degradation BD --pre_train ../model/SRRFN_x3_BD.pt --test_only --save_results --chop --save "SRRF" --testpath ../LR/LRBD --testset Set5

#SRRFN+_BD x3
python main.py --data_test MyImage --scale 3 --model SRRFN --degradation BD --pre_train ../model/SRRFN_x3_BD.pt --test_only --save_results --chop --self_ensemble --save "SRRFN_plus" --testpath ../LR/LRBD --testset Set5


# DN model

#SRRFN_DN x3
python main.py --data_test MyImage --scale 3 --model SRRFN --degradation DN --pre_train ../model/SRRFN_x3D_DN.pt --test_only --save_results --chop --save "SRRF" --testpath ../LR/LRDN --testset Set5

#SRRFN+_DN x3
python main.py --data_test MyImage --scale 3 --model SRRFN --degradation DN --pre_train ../model/SRRFN_x3D_DN.pt --test_only --save_results --chop --self_ensemble --save "SRRFN_plus" --testpath ../LR/LRDN --testset Set5
