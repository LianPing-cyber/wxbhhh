CUDA_VISIBLE_DEVICES=1 python main.py \
--dataset_path /data/example-s7 \
--keywords_in_path /root/comp/keywords_in.json \
--keywords_match_path /root/comp/keywords_match.json \
--svm_model_path /root/comp/wxbhhh/sys_channel/model \
--model_path /data/merged-0609 \
--label_mode 2 \
--run_pcap \
--run_session