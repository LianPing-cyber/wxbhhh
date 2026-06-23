CUDA_VISIBLE_DEVICES=1 python main.py \
--dataset_path /data/yuchen.yang/new_data \
--keywords_in_path /root/competation/keywords_in.json \
--keywords_match_path /root/competation/keywords_match.json \
--svm_model_path /root/competation/wxbhhh/sys_channel/model \
--model_path /data/yuchen.yang/AgentDoG1.5-Qwen3.5-4B \
--lora_path /data/yuchen.yang/sft-alldata \
--run_pcap \
--run_session