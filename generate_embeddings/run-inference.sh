config=MRL # RR or MRL

echo "Generating embeddings on MRL"
python pytorch_inference.py --retrieval --path=/home/qfshen/workspace/adanns/mrl-resnet50/mrl/Imagenet1k_R50_mrl/final_weights.pt \
--model_arch='resnet50' --retrieval_array_path=/home/qfshen/workspace/adanns/output/ --dataset=1K --mrl   

