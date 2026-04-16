"""
Week 7: Print DeepSeekV3Config to understand model parameters.
This script prints the default configuration for DeepSeekV3.
"""

from transformers import AutoConfig

# Load default DeepSeekV3 config
config = AutoConfig.for_model("deepseek_v3")

print("=" * 60)
print("DeepSeekV3 Default Configuration")
print("=" * 60)

# Print all config parameters
for key, value in sorted(config.to_dict().items()):
    print(f"  {key}: {value}")

print("\n" + "=" * 60)
print("Key MoE Parameters:")
print("=" * 60)
print(f"  hidden_size:            {config.hidden_size}")
print(f"  intermediate_size:      {config.intermediate_size}")
print(f"  moe_intermediate_size:  {config.moe_intermediate_size}")
print(f"  n_routed_experts:       {config.n_routed_experts}")
print(f"  n_shared_experts:       {config.n_shared_experts}")
print(f"  num_experts_per_tok:    {config.num_experts_per_tok}")
print(f"  n_group:                {config.n_group}")
print(f"  topk_group:             {config.topk_group}")
print(f"  norm_topk_prob:         {config.norm_topk_prob}")
print(f"  routed_scaling_factor:  {config.routed_scaling_factor}")
print(f"  hidden_act:             {config.hidden_act}")
print(f"  first_k_dense_replace:  {config.first_k_dense_replace}")
print(f"  num_local_experts:      {config.num_local_experts}")
