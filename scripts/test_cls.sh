python test.py --mode zero_shot --dataset visa \
--data_path ./data/visa --save_path ./results/visa/zero_shot \
--config_path ./open_clip/model_configs/ViT-L-14-336.json \
--model ViT-L-14-336 --features_list 6 12 18 24 --pretrained openai --image_size 336 \
--alpha_class 1.0 --image_only

python test.py --mode zero_shot --dataset mvtec \
--data_path ./data/mvtec --save_path ./results/mvtec/zero_shot_last \
--config_path ./open_clip/model_configs/ViT-B-16-plus-240.json \
--model ViT-B-16-plus-240 --features_list 6 12 18 24 --pretrained laion400m_e32 --image_size 240 \
--alpha_class 1.0 --image_only \
