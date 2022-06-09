
python main.py --config configs/ve/weatherAUS.py \
               --mode train \
               --workdir weatherAUS_VE/AS_langevin \

python main.py --config configs/ve/weatherAUS.py \
               --mode fine_tune \
               --workdir weatherAUS_VE/AS_langevin \
