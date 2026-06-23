CUDA_VISIBLE_DEVICES=1 python main.py \
--dataset_path /data/yuchen.yang/example-s7 \
--keywords_in_path /root/competation/keywords_in.json \
--keywords_match_path /root/competation/keywords_match.json \
--svm_model_path /root/competation/wxbhhh/sys_channel/model \
--model_path /data/yuchen.yang/merged-0609 \
--label_mode 2 \
--run_pcap \
--run_session