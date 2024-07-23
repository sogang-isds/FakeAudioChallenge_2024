current_dir=$(pwd)

cd espnet/egs2/librimix/enh_diar1
python3 -m espnet2.bin.diar_inference --ngpu 1 --fs 8k \
    --data_path_and_name_and_type ${current_dir}/data/wav.scp,speech,sound \
    --train_config models/libri23mix_eend_ss/exp/diar_enh_train_diar_enh_convtasnet_concat_feats_adapt/config.yaml \
    --model_file models/libri23mix_eend_ss/exp/diar_enh_train_diar_enh_convtasnet_concat_feats_adapt/16epoch.pth \
    --output_dir ${current_dir}/data/diar_enh \
    --config conf/tuning/decode_diar_enh.yaml \
    --num_spk 3

cd ../../../../