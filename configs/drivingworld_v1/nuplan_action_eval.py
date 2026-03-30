# seed
seed = 1234

# dataset
datasets_paths = dict(
    nuplan_root="/absolute/path/to/nuplan-v1.1/sensor_blobs",
)
nuplan_meta_path = "/absolute/path/to/nuplan_seq_meta.json"
nuplan_pose_meta_path = "/mnt/slurmfs-4090node1/homes/zchen897/Find_Physics_Zone/data/ego_meta"
downsample_fps = 5
image_size = [256, 512]

# model
n_layer = [12, 6]
n_embd = 1536
gpt_type = "ar"
pkeep = 0.7
condition_frames = 15
pose_x_vocab_size = 128
pose_y_vocab_size = 128
yaw_vocab_size = 512

# video vqvae
codebook_size = 16384
codebook_embed_dim = 32
downsample_size = 16
vq_model = "VideoVQ-16"
vq_ckpt = "/mnt/slurmfs-4090node1/homes/zchen897/Find_Physics_Zone/pretrained_models/vqvae.pt"
video_vq_temp_frames = condition_frames + 1

# sampling
sampling_mtd = None
top_k = 30
temperature_k = 1.0
top_p = 0.8
temperature_p = 1.0

# eval
max_sequences = None
max_steps_per_sequence = 64
num_visualizations = 8
use_bfloat16 = True
