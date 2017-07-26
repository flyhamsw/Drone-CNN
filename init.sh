rm -r patches
rm -r trained_model
python data.py
python make_patch.py
python make_patch_drone.py
