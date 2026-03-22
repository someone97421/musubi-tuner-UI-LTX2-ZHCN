# -*- coding: utf-8 -*-
"""LTX2 模型训练 WebUI (中文版) - 完整参数版"""
import gradio as gr
import subprocess, threading, queue, toml, os, sys, json



log_queue = queue.Queue()
current_process = None

# ──────────────────────────────────────
#  日志工具
# ──────────────────────────────────────
def read_logs():
    chunks = []
    while not log_queue.empty():
        chunks.append(log_queue.get())
    return "".join(chunks)

def update_logs(current):
    new = read_logs()
    return current + new if new else current

def clear_logs():
    while not log_queue.empty():
        log_queue.get()
    return ""

# ──────────────────────────────────────
#  进程管理
# ──────────────────────────────────────
def _run_in_thread(cmd):
    global current_process
    log_queue.put("==========================================\n")
    log_queue.put("执行命令:\n" + " ".join(str(c) for c in cmd) + "\n")
    log_queue.put("==========================================\n")
    env = os.environ.copy()
    env["PYTHONIOENCODING"] = "utf-8"
    
    try:
        current_process = subprocess.Popen(
            cmd, stdout=subprocess.PIPE, stderr=subprocess.STDOUT,
            text=True, bufsize=1, encoding="utf-8", errors="replace",
            cwd=os.path.dirname(os.path.abspath(__file__)),
            env=env
        )
        for line in iter(current_process.stdout.readline, ""):
            log_queue.put(line)
        current_process.stdout.close()
        current_process.wait()
        log_queue.put(f"\n[进程结束, 退出码: {current_process.returncode}]\n\n")
    except Exception as e:
        log_queue.put(f"\n[启动失败: {e}]\n\n")
    finally:
        current_process = None

def start_subprocess(cmd):
    threading.Thread(target=_run_in_thread, args=(cmd,), daemon=True).start()

def stop_process():
    global current_process
    if current_process and current_process.poll() is None:
        current_process.terminate()
        log_queue.put("\n[用户已终止当前进程]\n")
        return "已发送终止信号。"
    return "没有正在运行的进程。"


# ──────────────────────────────────────
#  配置 加载 / 保存
# ──────────────────────────────────────
CONFIG_PATH = "ltx2_train_config.toml"
UI_CACHE_PATH = "ltx2_ui_cache.json"

DATASET_DEFAULTS = {
    "ds_res_w": 960,
    "ds_res_h": 544,
    "ds_cap_ext": ".txt",
    "ds_batch_size": 1,
    "ds_enable_bucket": True,
    "ds_bucket_no_upscale": False,
    "ds_sep_audio": False,
    "ds_vid_dir": "",
    "ds_cache_dir": "",
    "ds_aud_dir": "",
    "ds_aud_strat": "pad",
    "ds_aud_int": 2.0,
    "ds_target_frames": "49, 97, 145, 193, 201",
    "ds_max_frames": 201,
    "ds_target_fps": 25.0,
    "ds_save_path": "dataset_ltx2.toml",
}

TRAIN_DEFAULTS = {
    "ltx2_checkpoint": "",
    "vae": "",
    "output_dir": "./outputs",
    "output_name": "ltx2_lora",
    "dataset_config": "dataset_ltx2.toml",
    "gemma_root": "",
    "gemma_safetensors": "",
    "gemma_load_in_8bit": False,
    "gemma_load_in_4bit": False,
    "ltx_mode": "video",
    "lora_target_preset": "t2v",
    "ic_lora_strategy": "auto",
    "network_module": "networks.lora_ltx2",
    "network_dim": 32,
    "network_alpha": 16,
    "network_dropout": 0,
    "scale_weight_norms": 0,
    "network_weights": "",
    "dim_from_weights": False,
    "enable_lora_plus": True,
    "loraplus_lr_ratio": 4,
    "enable_blocks": False,
    "exclude_patterns": "",
    "include_patterns": "",
    "max_train_epochs": 10,
    "max_train_steps": "5000",
    "train_batch_size": 1,
    "seed": 1026,
    "gradient_checkpointing": True,
    "gradient_checkpointing_cpu_offload": False,
    "blocks_to_checkpoint": -1,
    "gradient_accumulation_steps": 1,
    "guidance_scale": 1.0,
    "caption_dropout_rate": 0.0,
    "mixed_precision": "bf16",
    "vae_dtype": "",
    "ltx_version": "2.3",
    "ltx_version_check_mode": "warn",
    "fp8_base": True,
    "fp8_scaled": True,
    "fp8_w8a8": False,
    "w8a8_mode": "int8",
    "nf4_base": False,
    "nf4_block_size": 64,
    "attention_mode": "sdpa",
    "blocks_to_swap": 0,
    "use_pinned_memory_for_block_swap": True,
    "split_attn_target": "none",
    "split_attn_mode": "batch",
    "split_attn_chunk_size": 0,
    "ffn_chunk_target": "none",
    "ffn_chunk_size": 0,
    "blockwise_checkpointing": False,
    "loss_type": "mse",
    "huber_delta": 1.0,
    "preserve_distribution_shape": False,
    "video_loss_weight": 1.0,
    "audio_loss_weight": 1.0,
    "audio_loss_balance_mode": "none",
    "audio_loss_balance_beta": 0.01,
    "audio_loss_balance_eps": 0.05,
    "audio_loss_balance_min": 0.05,
    "audio_loss_balance_max": 4.0,
    "audio_loss_balance_target_ratio": 0.33,
    "learning_rate": "1e-4",
    "audio_lr": "",
    "lr_args": "",
    "lr_scheduler": "constant_with_warmup",
    "lr_warmup_steps": 50,
    "lr_decay_steps": 0.2,
    "lr_scheduler_num_cycles": 1,
    "lr_scheduler_power": 1.0,
    "lr_scheduler_timescale": 0,
    "lr_scheduler_min_lr_ratio": 0.1,
    "timestep_sampling": "shift",
    "shifted_logit_mode": "",
    "discrete_flow_shift": 3.0,
    "sigmoid_scale": 1.0,
    "weighting_scheme": "",
    "logit_mean": 0.0,
    "logit_std": 1.0,
    "mode_scale": 1.29,
    "min_timestep": 0,
    "max_timestep": 1000,
    "optimizer_type": "AdamW8bit",
    "max_grad_norm": 1.0,
    "prodigy_beta1": 0.9,
    "prodigy_beta2": 0.999,
    "prodigy_beta3": 0.9,
    "prodigy_d0": 1e-6,
    "prodigy_growth_rate": 999.0,
    "prodigy_use_bias_correction": False,
    "prodigy_use_speed": False,
    "prodigy_safeguard_warmup": False,
    "save_every_n_epochs": 2,
    "save_every_n_steps": "250",
    "save_last_n_epochs": "",
    "save_last_n_steps": "",
    "save_state": False,
    "save_state_on_train_end": False,
    "resume": "",
    "log_with": "tensorboard",
    "logging_dir": "./logs",
    "enable_sample": False,
    "sample_at_first": False,
    "sample_every_n_epochs": 2,
    "sample_every_n_steps": "",
    "sample_prompts": "",
    "use_precached_sample_prompts": False,
    "sample_prompts_cache": "",
    "use_precached_sample_latents": False,
    "sample_latents_cache": "",
    "sample_disable_flash_attn": False,
    "audio_ref_identity_guidance_scale": 0.0,
    "height": 512,
    "width": 512,
    "sample_num_frames": 49,
    "max_data_loader_n_workers": 8,
    "persistent_data_loader_workers": True,
    "cuda_allow_tf32": True,
    "cuda_cudnn_benchmark": True,
    "training_comment": "",
    "metadata_title": "",
    "metadata_author": "",
    "metadata_description": "",
}

TRAIN_ARG_KEYS = [
    "ltx2_checkpoint", "vae", "output_dir", "output_name", "dataset_config",
    "gemma_root", "gemma_safetensors", "gemma_load_in_8bit", "gemma_load_in_4bit",
    "ltx_mode", "lora_target_preset", "ic_lora_strategy", "network_module",
    "network_dim", "network_alpha", "network_dropout", "scale_weight_norms",
    "network_weights", "dim_from_weights",
    "enable_lora_plus", "loraplus_lr_ratio",
    "enable_blocks", "exclude_patterns", "include_patterns",
    "max_train_epochs", "max_train_steps", "train_batch_size", "seed",
    "gradient_checkpointing", "gradient_checkpointing_cpu_offload", "blocks_to_checkpoint", "gradient_accumulation_steps",
    "guidance_scale", "caption_dropout_rate",
    "mixed_precision", "vae_dtype", "ltx_version", "ltx_version_check_mode",
    "fp8_base", "fp8_scaled", "fp8_w8a8", "w8a8_mode", "nf4_base", "nf4_block_size",
    "attention_mode", "blocks_to_swap", "use_pinned_memory_for_block_swap",
    "split_attn_target", "split_attn_mode", "split_attn_chunk_size",
    "ffn_chunk_target", "ffn_chunk_size", "blockwise_checkpointing",
    "loss_type", "huber_delta", "preserve_distribution_shape",
    "video_loss_weight", "audio_loss_weight", "audio_loss_balance_mode", "audio_loss_balance_beta",
    "audio_loss_balance_eps", "audio_loss_balance_min", "audio_loss_balance_max", "audio_loss_balance_target_ratio",
    "learning_rate", "audio_lr", "lr_args", "lr_scheduler", "lr_warmup_steps", "lr_decay_steps",
    "lr_scheduler_num_cycles", "lr_scheduler_power",
    "lr_scheduler_timescale", "lr_scheduler_min_lr_ratio",
    "timestep_sampling", "shifted_logit_mode", "discrete_flow_shift", "sigmoid_scale",
    "weighting_scheme", "logit_mean", "logit_std", "mode_scale",
    "min_timestep", "max_timestep",
    "optimizer_type", "max_grad_norm",
    "prodigy_beta1", "prodigy_beta2", "prodigy_beta3", "prodigy_d0",
    "prodigy_growth_rate", "prodigy_use_bias_correction", "prodigy_use_speed", "prodigy_safeguard_warmup",
    "save_every_n_epochs", "save_every_n_steps",
    "save_last_n_epochs", "save_last_n_steps",
    "save_state", "save_state_on_train_end", "resume",
    "log_with", "logging_dir",
    "enable_sample", "sample_at_first",
    "sample_every_n_epochs", "sample_every_n_steps", "sample_prompts",
    "use_precached_sample_prompts", "sample_prompts_cache", "use_precached_sample_latents", "sample_latents_cache",
    "sample_disable_flash_attn", "audio_ref_identity_guidance_scale",
    "height", "width", "sample_num_frames",
    "max_data_loader_n_workers", "persistent_data_loader_workers",
    "cuda_allow_tf32", "cuda_cudnn_benchmark",
    "training_comment", "metadata_title", "metadata_author", "metadata_description",
]


def _read_ui_cache():
    if not os.path.exists(UI_CACHE_PATH):
        return {}
    try:
        with open(UI_CACHE_PATH, "r", encoding="utf-8") as f:
            return json.load(f)
    except Exception as e:
        log_queue.put(f"读取缓存失败: {e}\n")
        return {}


def _write_ui_cache(data):
    try:
        with open(UI_CACHE_PATH, "w", encoding="utf-8") as f:
            json.dump(data, f, ensure_ascii=False, indent=2)
    except Exception as e:
        log_queue.put(f"写入缓存失败: {e}\n")


def _parse_network_args(cfg):
    result = {
        "enable_lora_plus": TRAIN_DEFAULTS["enable_lora_plus"],
        "loraplus_lr_ratio": TRAIN_DEFAULTS["loraplus_lr_ratio"],
        "enable_blocks": TRAIN_DEFAULTS["enable_blocks"],
        "exclude_patterns": TRAIN_DEFAULTS["exclude_patterns"],
        "include_patterns": TRAIN_DEFAULTS["include_patterns"],
    }
    for arg in cfg.get("network_args", []):
        if "=" not in arg:
            continue
        key, value = arg.split("=", 1)
        key = key.strip()
        value = value.strip()
        if key == "loraplus_lr_ratio":
            result["enable_lora_plus"] = True
            result["loraplus_lr_ratio"] = float(value)
        elif key == "exclude_patterns":
            result["enable_blocks"] = True
            result["exclude_patterns"] = value
        elif key == "include_patterns":
            result["enable_blocks"] = True
            result["include_patterns"] = value
    return result


def _parse_optimizer_args(cfg):
    defaults = {
        "beta1": TRAIN_DEFAULTS["prodigy_beta1"],
        "beta2": TRAIN_DEFAULTS["prodigy_beta2"],
        "beta3": TRAIN_DEFAULTS["prodigy_beta3"],
        "d0": TRAIN_DEFAULTS["prodigy_d0"],
        "growth_rate": TRAIN_DEFAULTS["prodigy_growth_rate"],
        "use_bias_correction": TRAIN_DEFAULTS["prodigy_use_bias_correction"],
        "use_speed": TRAIN_DEFAULTS["prodigy_use_speed"],
        "safeguard_warmup": TRAIN_DEFAULTS["prodigy_safeguard_warmup"],
    }
    opt_args = cfg.get("optimizer_args", [])
    for arg in opt_args:
        if "=" not in arg:
            continue
        key, val = arg.split("=", 1)
        key = key.strip()
        val = val.strip()
        try:
            if key == "betas" and "(" in val:
                nums = val.strip("()").split(",")
                defaults["beta1"] = float(nums[0].strip())
                defaults["beta2"] = float(nums[1].strip())
            elif key == "beta3":
                defaults["beta3"] = float(val)
            elif key == "d0":
                defaults["d0"] = float(val)
            elif key == "growth_rate":
                defaults["growth_rate"] = float(val)
            elif key == "use_bias_correction":
                defaults["use_bias_correction"] = val.lower() == "true"
            elif key == "use_speed":
                defaults["use_speed"] = val.lower() == "true"
            elif key == "safeguard_warmup":
                defaults["safeguard_warmup"] = val.lower() == "true"
        except Exception:
            pass
    return defaults


def _cfg_to_updates(cfg):
    network_args = _parse_network_args(cfg)
    prodigy_defaults = _parse_optimizer_args(cfg)
    attn_val = TRAIN_DEFAULTS["attention_mode"]
    if cfg.get("flash_attn", False):
        attn_val = "flash_attn"
    elif cfg.get("flash3", False):
        attn_val = "flash3"
    elif cfg.get("xformers", False):
        attn_val = "xformers"
    elif cfg.get("sdpa", False):
        attn_val = "sdpa"

    def g(key, default_key=None):
        if key in cfg:
            return cfg[key]
        if default_key is None:
            default_key = key
        return TRAIN_DEFAULTS.get(default_key)

    lr_args_value = g("lr_args", "lr_args")
    if isinstance(lr_args_value, list):
        lr_args_value = " ".join(str(item) for item in lr_args_value)

    return [
        gr.update(value=g("ltx2_checkpoint")),
        gr.update(value=g("vae")),
        gr.update(value=g("output_dir")),
        gr.update(value=g("output_name")),
        gr.update(value=g("dataset_config")),
        gr.update(value=g("gemma_root")),
        gr.update(value=g("gemma_safetensors")),
        gr.update(value=g("gemma_load_in_8bit")),
        gr.update(value=g("gemma_load_in_4bit")),
        gr.update(value=cfg.get("ltx_mode") or cfg.get("ltx2_mode") or TRAIN_DEFAULTS["ltx_mode"]),
        gr.update(value=g("lora_target_preset")),
        gr.update(value=g("ic_lora_strategy")),
        gr.update(value=g("network_module")),
        gr.update(value=g("network_dim")),
        gr.update(value=g("network_alpha")),
        gr.update(value=g("network_dropout")),
        gr.update(value=g("scale_weight_norms")),
        gr.update(value=g("network_weights")),
        gr.update(value=g("dim_from_weights")),
        gr.update(value=network_args["enable_lora_plus"]),
        gr.update(value=network_args["loraplus_lr_ratio"]),
        gr.update(value=network_args["enable_blocks"]),
        gr.update(value=network_args["exclude_patterns"]),
        gr.update(value=network_args["include_patterns"]),
        gr.update(value=g("max_train_epochs")),
        gr.update(value=str(g("max_train_steps", "max_train_steps"))),
        gr.update(value=g("train_batch_size")),
        gr.update(value=g("seed")),
        gr.update(value=g("gradient_checkpointing")),
        gr.update(value=g("gradient_checkpointing_cpu_offload")),
        gr.update(value=g("blocks_to_checkpoint")),
        gr.update(value=g("gradient_accumulation_steps")),
        gr.update(value=g("guidance_scale")),
        gr.update(value=g("caption_dropout_rate")),
        gr.update(value=g("mixed_precision")),
        gr.update(value=g("vae_dtype")),
        gr.update(value=g("ltx_version")),
        gr.update(value=g("ltx_version_check_mode")),
        gr.update(value=g("fp8_base")),
        gr.update(value=g("fp8_scaled")),
        gr.update(value=g("fp8_w8a8")),
        gr.update(value=g("w8a8_mode")),
        gr.update(value=g("nf4_base")),
        gr.update(value=g("nf4_block_size")),
        gr.update(value=attn_val),
        gr.update(value=g("blocks_to_swap")),
        gr.update(value=g("use_pinned_memory_for_block_swap")),
        gr.update(value=g("split_attn_target")),
        gr.update(value=g("split_attn_mode")),
        gr.update(value=g("split_attn_chunk_size")),
        gr.update(value=g("ffn_chunk_target")),
        gr.update(value=g("ffn_chunk_size")),
        gr.update(value=g("blockwise_checkpointing")),
        gr.update(value=g("loss_type")),
        gr.update(value=g("huber_delta")),
        gr.update(value=g("preserve_distribution_shape")),
        gr.update(value=g("video_loss_weight")),
        gr.update(value=g("audio_loss_weight")),
        gr.update(value=g("audio_loss_balance_mode")),
        gr.update(value=g("audio_loss_balance_beta")),
        gr.update(value=g("audio_loss_balance_eps")),
        gr.update(value=g("audio_loss_balance_min")),
        gr.update(value=g("audio_loss_balance_max")),
        gr.update(value=g("audio_loss_balance_target_ratio")),
        gr.update(value=str(g("learning_rate", "learning_rate"))),
        gr.update(value=str(g("audio_lr", "audio_lr"))),
        gr.update(value=lr_args_value),
        gr.update(value=g("lr_scheduler")),
        gr.update(value=g("lr_warmup_steps")),
        gr.update(value=g("lr_decay_steps")),
        gr.update(value=g("lr_scheduler_num_cycles")),
        gr.update(value=g("lr_scheduler_power")),
        gr.update(value=g("lr_scheduler_timescale")),
        gr.update(value=g("lr_scheduler_min_lr_ratio")),
        gr.update(value=g("timestep_sampling")),
        gr.update(value=g("shifted_logit_mode")),
        gr.update(value=g("discrete_flow_shift")),
        gr.update(value=g("sigmoid_scale")),
        gr.update(value=g("weighting_scheme")),
        gr.update(value=g("logit_mean")),
        gr.update(value=g("logit_std")),
        gr.update(value=g("mode_scale")),
        gr.update(value=g("min_timestep")),
        gr.update(value=g("max_timestep")),
        gr.update(value=g("optimizer_type")),
        gr.update(value=g("max_grad_norm")),
        gr.update(value=prodigy_defaults["beta1"]),
        gr.update(value=prodigy_defaults["beta2"]),
        gr.update(value=prodigy_defaults["beta3"]),
        gr.update(value=prodigy_defaults["d0"]),
        gr.update(value=prodigy_defaults["growth_rate"]),
        gr.update(value=prodigy_defaults["use_bias_correction"]),
        gr.update(value=prodigy_defaults["use_speed"]),
        gr.update(value=prodigy_defaults["safeguard_warmup"]),
        gr.update(value=g("save_every_n_epochs")),
        gr.update(value=str(g("save_every_n_steps", "save_every_n_steps"))),
        gr.update(value=str(g("save_last_n_epochs", "save_last_n_epochs"))),
        gr.update(value=str(g("save_last_n_steps", "save_last_n_steps"))),
        gr.update(value=g("save_state")),
        gr.update(value=g("save_state_on_train_end")),
        gr.update(value=g("resume")),
        gr.update(value=g("log_with")),
        gr.update(value=g("logging_dir")),
        gr.update(value=bool(cfg.get("sample_every_n_epochs") or cfg.get("sample_every_n_steps"))),
        gr.update(value=g("sample_at_first")),
        gr.update(value=g("sample_every_n_epochs")),
        gr.update(value=str(g("sample_every_n_steps", "sample_every_n_steps"))),
        gr.update(value=g("sample_prompts")),
        gr.update(value=g("use_precached_sample_prompts")),
        gr.update(value=g("sample_prompts_cache")),
        gr.update(value=g("use_precached_sample_latents")),
        gr.update(value=g("sample_latents_cache")),
        gr.update(value=g("sample_disable_flash_attn")),
        gr.update(value=g("audio_ref_identity_guidance_scale")),
        gr.update(value=g("height")),
        gr.update(value=g("width")),
        gr.update(value=g("sample_num_frames")),
        gr.update(value=g("max_data_loader_n_workers")),
        gr.update(value=g("persistent_data_loader_workers")),
        gr.update(value=g("cuda_allow_tf32")),
        gr.update(value=g("cuda_cudnn_benchmark")),
        gr.update(value=g("training_comment")),
        gr.update(value=g("metadata_title")),
        gr.update(value=g("metadata_author")),
        gr.update(value=g("metadata_description")),
    ]


def _default_train_updates():
    return _cfg_to_updates(TRAIN_DEFAULTS)


def _dataset_updates_from_cache(cache):
    dataset_values = dict(DATASET_DEFAULTS)
    dataset_values.update(cache.get("dataset_form", {}))
    return [
        gr.update(value=dataset_values["ds_res_w"]),
        gr.update(value=dataset_values["ds_res_h"]),
        gr.update(value=dataset_values["ds_cap_ext"]),
        gr.update(value=dataset_values["ds_batch_size"]),
        gr.update(value=dataset_values["ds_enable_bucket"]),
        gr.update(value=dataset_values["ds_bucket_no_upscale"]),
        gr.update(value=dataset_values["ds_sep_audio"]),
        gr.update(value=dataset_values["ds_vid_dir"]),
        gr.update(value=dataset_values["ds_cache_dir"]),
        gr.update(value=dataset_values["ds_aud_dir"]),
        gr.update(value=dataset_values["ds_aud_strat"]),
        gr.update(value=dataset_values["ds_aud_int"]),
        gr.update(value=dataset_values["ds_target_frames"]),
        gr.update(value=dataset_values["ds_max_frames"]),
        gr.update(value=dataset_values["ds_target_fps"]),
        gr.update(value=dataset_values["ds_save_path"]),
    ]

def load_config_file(path):
    """载入 TOML 并以 JSON 字符串返回（供前端更新组件用）"""
    p = path.strip() if path else CONFIG_PATH
    if not os.path.exists(p):
        log_queue.put(f"找不到配置文件: {p}\n")
        return None
    try:
        cfg = toml.load(p)
        log_queue.put(f"已载入配置: {p}\n")
        return cfg  # 直接返回字典供调用方使用
    except Exception as e:
        log_queue.put(f"载入失败: {e}\n")
        return None


def build_config(
    ltx2_checkpoint, vae, output_dir, output_name, dataset_config,
    gemma_root, gemma_safetensors, gemma_load_in_8bit, gemma_load_in_4bit,
    ltx2_mode, lora_target_preset, ic_lora_strategy, network_module,
    network_dim, network_alpha, network_dropout, scale_weight_norms,
    network_weights, dim_from_weights,
    enable_lora_plus, loraplus_lr_ratio,
    enable_blocks, exclude_patterns, include_patterns,
    max_train_epochs, max_train_steps, batch_size, seed,
    gradient_checkpointing, gradient_checkpointing_cpu_offload, blocks_to_checkpoint, gradient_accumulation_steps,
    guidance_scale, caption_dropout_rate,
    mixed_precision, vae_dtype, ltx_version, ltx_version_check_mode,
    fp8_base, fp8_scaled, fp8_w8a8, w8a8_mode, nf4_base, nf4_block_size,
    attention_mode, blocks_to_swap, use_pinned_memory_for_block_swap,
    split_attn_target, split_attn_mode, split_attn_chunk_size,
    ffn_chunk_target, ffn_chunk_size, blockwise_checkpointing,
    loss_type, huber_delta, preserve_distribution_shape,
    video_loss_weight, audio_loss_weight, audio_loss_balance_mode, audio_loss_balance_beta,
    audio_loss_balance_eps, audio_loss_balance_min, audio_loss_balance_max, audio_loss_balance_target_ratio,
    lr, audio_lr, lr_args, lr_scheduler, lr_warmup_steps, lr_decay_steps,
    lr_scheduler_num_cycles, lr_scheduler_power,
    lr_scheduler_timescale, lr_scheduler_min_lr_ratio,
    timestep_sampling, shifted_logit_mode, discrete_flow_shift, sigmoid_scale,
    weighting_scheme, logit_mean, logit_std, mode_scale,
    min_timestep, max_timestep,
    optimizer_type, max_grad_norm,
    prodigy_beta1, prodigy_beta2, prodigy_beta3, prodigy_d0,
    prodigy_growth_rate, prodigy_use_bias_correction, prodigy_use_speed, prodigy_safeguard_warmup,
    save_every_n_epochs, save_every_n_steps,
    save_last_n_epochs, save_last_n_steps,
    save_state, save_state_on_train_end, resume,
    log_with, logging_dir,
    enable_sample, sample_at_first,
    sample_every_n_epochs, sample_every_n_steps, sample_prompts,
    use_precached_sample_prompts, sample_prompts_cache, use_precached_sample_latents, sample_latents_cache,
    sample_disable_flash_attn, audio_ref_identity_guidance_scale,
    height, width, sample_num_frames,
    max_data_loader_n_workers, persistent_data_loader_workers,
    cuda_allow_tf32, cuda_cudnn_benchmark,
    training_comment, metadata_title, metadata_author, metadata_description,
):
    cfg = {}
    if ltx2_checkpoint: cfg["ltx2_checkpoint"] = ltx2_checkpoint
    if vae: cfg["vae"] = vae
    if output_dir: cfg["output_dir"] = output_dir
    if output_name: cfg["output_name"] = output_name
    if dataset_config: cfg["dataset_config"] = dataset_config
    if gemma_root: cfg["gemma_root"] = gemma_root
    if gemma_safetensors: cfg["gemma_safetensors"] = gemma_safetensors
    if gemma_load_in_8bit: cfg["gemma_load_in_8bit"] = True
    if gemma_load_in_4bit: cfg["gemma_load_in_4bit"] = True

    if ltx2_mode: cfg["ltx_mode"] = ltx2_mode
    if lora_target_preset: cfg["lora_target_preset"] = lora_target_preset
    if ic_lora_strategy and ic_lora_strategy != "auto": cfg["ic_lora_strategy"] = ic_lora_strategy
    if network_module: cfg["network_module"] = network_module

    cfg["network_dim"] = int(network_dim)
    cfg["network_alpha"] = float(network_alpha)
    if float(network_dropout) > 0: cfg["network_dropout"] = float(network_dropout)
    if float(scale_weight_norms) > 0: cfg["scale_weight_norms"] = float(scale_weight_norms)
    if network_weights: cfg["network_weights"] = network_weights
    if dim_from_weights: cfg["dim_from_weights"] = True

    network_args = []
    if enable_lora_plus and loraplus_lr_ratio:
        network_args.append(f"loraplus_lr_ratio={loraplus_lr_ratio}")
    if enable_blocks:
        if exclude_patterns: network_args.append(f"exclude_patterns={exclude_patterns}")
        if include_patterns: network_args.append(f"include_patterns={include_patterns}")
    if network_args: cfg["network_args"] = network_args

    cfg["max_train_epochs"] = int(max_train_epochs)
    if max_train_steps: cfg["max_train_steps"] = int(max_train_steps)
    cfg["train_batch_size"] = int(batch_size)
    if seed: cfg["seed"] = int(seed)
    if gradient_checkpointing: cfg["gradient_checkpointing"] = True
    if gradient_checkpointing_cpu_offload: cfg["gradient_checkpointing_cpu_offload"] = True
    if int(blocks_to_checkpoint) != -1: cfg["blocks_to_checkpoint"] = int(blocks_to_checkpoint)
    cfg["gradient_accumulation_steps"] = int(gradient_accumulation_steps)
    cfg["guidance_scale"] = float(guidance_scale)
    if float(caption_dropout_rate) > 0: cfg["caption_dropout_rate"] = float(caption_dropout_rate)

    cfg["mixed_precision"] = mixed_precision
    if vae_dtype: cfg["vae_dtype"] = vae_dtype
    if ltx_version: cfg["ltx_version"] = ltx_version
    if ltx_version_check_mode: cfg["ltx_version_check_mode"] = ltx_version_check_mode
    if fp8_base: cfg["fp8_base"] = True
    if fp8_scaled: cfg["fp8_scaled"] = True
    if fp8_w8a8:
        cfg["fp8_w8a8"] = True
        if w8a8_mode != "int8": cfg["w8a8_mode"] = w8a8_mode
    if nf4_base:
        cfg["nf4_base"] = True
        if int(nf4_block_size) != 64: cfg["nf4_block_size"] = int(nf4_block_size)

    if attention_mode and attention_mode != "none":
        cfg[attention_mode] = True

    if int(blocks_to_swap) > 0:
        cfg["blocks_to_swap"] = int(blocks_to_swap)
        if use_pinned_memory_for_block_swap: cfg["use_pinned_memory_for_block_swap"] = True
    if split_attn_target and split_attn_target != "none":
        cfg["split_attn_target"] = split_attn_target
        cfg["split_attn_mode"] = split_attn_mode
        if int(split_attn_chunk_size) > 0: cfg["split_attn_chunk_size"] = int(split_attn_chunk_size)
    if ffn_chunk_target and ffn_chunk_target != "none":
        cfg["ffn_chunk_target"] = ffn_chunk_target
        if int(ffn_chunk_size) > 0: cfg["ffn_chunk_size"] = int(ffn_chunk_size)
    if blockwise_checkpointing: cfg["blockwise_checkpointing"] = True

    if loss_type and loss_type != "mse": cfg["loss_type"] = loss_type
    if loss_type in ("huber", "smooth_l1") and float(huber_delta) != 1.0: cfg["huber_delta"] = float(huber_delta)
    if preserve_distribution_shape: cfg["preserve_distribution_shape"] = True
    if float(video_loss_weight) != 1.0: cfg["video_loss_weight"] = float(video_loss_weight)
    if float(audio_loss_weight) != 1.0: cfg["audio_loss_weight"] = float(audio_loss_weight)
    if audio_loss_balance_mode and audio_loss_balance_mode != "none": cfg["audio_loss_balance_mode"] = audio_loss_balance_mode
    if float(audio_loss_balance_beta) != 0.01: cfg["audio_loss_balance_beta"] = float(audio_loss_balance_beta)
    if float(audio_loss_balance_eps) != 0.05: cfg["audio_loss_balance_eps"] = float(audio_loss_balance_eps)
    if float(audio_loss_balance_min) != 0.05: cfg["audio_loss_balance_min"] = float(audio_loss_balance_min)
    if float(audio_loss_balance_max) != 4.0: cfg["audio_loss_balance_max"] = float(audio_loss_balance_max)
    if float(audio_loss_balance_target_ratio) != 0.33: cfg["audio_loss_balance_target_ratio"] = float(audio_loss_balance_target_ratio)

    cfg["learning_rate"] = float(lr) if lr else 0.0001
    if audio_lr not in ("", None): cfg["audio_lr"] = float(audio_lr)
    if lr_args and lr_args.strip(): cfg["lr_args"] = lr_args.split()
    cfg["lr_scheduler"] = lr_scheduler
    if int(lr_warmup_steps) > 0: cfg["lr_warmup_steps"] = int(lr_warmup_steps)
    if float(str(lr_decay_steps)) > 0: cfg["lr_decay_steps"] = float(str(lr_decay_steps))
    if int(lr_scheduler_num_cycles) > 1: cfg["lr_scheduler_num_cycles"] = int(lr_scheduler_num_cycles)
    if float(lr_scheduler_power) != 1.0: cfg["lr_scheduler_power"] = float(lr_scheduler_power)
    if int(lr_scheduler_timescale) > 0: cfg["lr_scheduler_timescale"] = int(lr_scheduler_timescale)
    if float(lr_scheduler_min_lr_ratio) > 0: cfg["lr_scheduler_min_lr_ratio"] = float(lr_scheduler_min_lr_ratio)

    cfg["timestep_sampling"] = timestep_sampling
    if shifted_logit_mode: cfg["shifted_logit_mode"] = shifted_logit_mode
    cfg["discrete_flow_shift"] = float(discrete_flow_shift)
    if float(sigmoid_scale) != 1.0: cfg["sigmoid_scale"] = float(sigmoid_scale)
    if weighting_scheme: cfg["weighting_scheme"] = weighting_scheme
    if float(logit_mean) != 0.0: cfg["logit_mean"] = float(logit_mean)
    if float(logit_std) != 1.0: cfg["logit_std"] = float(logit_std)
    if float(mode_scale) != 1.29: cfg["mode_scale"] = float(mode_scale)
    if int(min_timestep) > 0: cfg["min_timestep"] = int(min_timestep)
    if int(max_timestep) != 1000: cfg["max_timestep"] = int(max_timestep)

    cfg["optimizer_type"] = optimizer_type
    if float(max_grad_norm) != 1.0: cfg["max_grad_norm"] = float(max_grad_norm)
    
    # Prodigy 优化器参数处理
    if optimizer_type == "prodigyopt.Prodigy":
        optimizer_args = []
        beta1 = float(prodigy_beta1) if prodigy_beta1 else 0.9
        beta2 = float(prodigy_beta2) if prodigy_beta2 else 0.999
        optimizer_args.append(f"betas=({beta1}, {beta2})")
        if prodigy_beta3 is not None:
            optimizer_args.append(f"beta3={float(prodigy_beta3)}")
        if prodigy_d0 is not None:
            optimizer_args.append(f"d0={float(prodigy_d0)}")
        if prodigy_growth_rate is not None and float(prodigy_growth_rate) < 999.0:
            optimizer_args.append(f"growth_rate={float(prodigy_growth_rate)}")
        if prodigy_use_bias_correction is not None and bool(prodigy_use_bias_correction):
            optimizer_args.append(f"use_bias_correction=True")
        if prodigy_use_speed is not None and bool(prodigy_use_speed):
            optimizer_args.append(f"use_speed=True")
        if prodigy_safeguard_warmup is not None and bool(prodigy_safeguard_warmup):
            optimizer_args.append(f"safeguard_warmup=True")
        if optimizer_args:
            cfg["optimizer_args"] = optimizer_args

    if save_every_n_steps: cfg["save_every_n_steps"] = int(save_every_n_steps)
    elif save_every_n_epochs: cfg["save_every_n_epochs"] = int(save_every_n_epochs)
    if save_last_n_epochs: cfg["save_last_n_epochs"] = int(save_last_n_epochs)
    if save_last_n_steps: cfg["save_last_n_steps"] = int(save_last_n_steps)
    if save_state_on_train_end: cfg["save_state_on_train_end"] = True
    elif save_state: cfg["save_state"] = True
    if resume: cfg["resume"] = resume
    
    if log_with:
        cfg["log_with"] = log_with
        if logging_dir: cfg["logging_dir"] = logging_dir

    if enable_sample:
        if sample_at_first: cfg["sample_at_first"] = True
        if sample_every_n_steps: cfg["sample_every_n_steps"] = int(sample_every_n_steps)
        else: cfg["sample_every_n_epochs"] = int(sample_every_n_epochs)
        if sample_prompts: cfg["sample_prompts"] = sample_prompts
        if use_precached_sample_prompts: cfg["use_precached_sample_prompts"] = True
        if sample_prompts_cache and sample_prompts_cache.strip(): cfg["sample_prompts_cache"] = sample_prompts_cache.strip()
        if use_precached_sample_latents: cfg["use_precached_sample_latents"] = True
        if sample_latents_cache and sample_latents_cache.strip(): cfg["sample_latents_cache"] = sample_latents_cache.strip()
        if sample_disable_flash_attn: cfg["sample_disable_flash_attn"] = True
        if float(audio_ref_identity_guidance_scale) > 0.0:
            cfg["audio_ref_identity_guidance_scale"] = float(audio_ref_identity_guidance_scale)
        if int(height) > 0: cfg["height"] = int(height)
        if int(width) > 0: cfg["width"] = int(width)
        if int(sample_num_frames) > 0: cfg["sample_num_frames"] = int(sample_num_frames)

    if int(max_data_loader_n_workers) != 8: cfg["max_data_loader_n_workers"] = int(max_data_loader_n_workers)
    if persistent_data_loader_workers: cfg["persistent_data_loader_workers"] = True
    if cuda_allow_tf32: cfg["cuda_allow_tf32"] = True
    if cuda_cudnn_benchmark: cfg["cuda_cudnn_benchmark"] = True

    if training_comment: cfg["training_comment"] = training_comment
    if metadata_title: cfg["metadata_title"] = metadata_title
    if metadata_author: cfg["metadata_author"] = metadata_author
    if metadata_description: cfg["metadata_description"] = metadata_description

    return cfg


def save_config(*args):
    save_path = args[-1]  # last arg is the save path textbox
    # args[0] is ltx2_checkpoint (first param of build_config)
    ltx2_ckpt_val = args[0]
    if not ltx2_ckpt_val or not ltx2_ckpt_val.strip():
        return "❌ 保存失败：请先在顶部填写 LTX2 模型路径（必填）！"
    cfg = build_config(*args[:-1])
    p = save_path.strip() if save_path else CONFIG_PATH
    try:
        with open(p, "w", encoding="utf-8") as f:
            toml.dump(cfg, f)
        log_queue.put(f"配置已保存到 {p}\n")
        return f"✅ 保存成功！→ {p}"
    except Exception as e:
        log_queue.put(f"保存失败: {e}\n")
        return f"❌ 保存失败: {e}"


# ──────────────────────────────────────
#  Cache / 训练命令
# ──────────────────────────────────────
def run_cache_latents(dataset_cfg, ltx2_ckpt, vae_p, vae_dtype_c, skip_existing, fp8_vae, v_chunk, v_sp_tile, v_sp_ov, v_tp_tile, v_tp_ov, c_mode):
    if not (ltx2_ckpt and ltx2_ckpt.strip()) and not (vae_p and vae_p.strip()):
        return "❌ 错误：请先在界面顶部【0. 全局设置】中填写【LTX2 权重路径】！"
    
    cmd = [sys.executable, "ltx2_cache_latents.py",
           "--dataset_config", dataset_cfg or "dataset_ltx2.toml"]
    if ltx2_ckpt and ltx2_ckpt.strip(): cmd += ["--ltx2_checkpoint", ltx2_ckpt.strip()]
    if vae_p and vae_p.strip(): cmd += ["--vae", vae_p.strip()]
    if vae_dtype_c: cmd += ["--vae_dtype", vae_dtype_c]
    if skip_existing: cmd += ["--skip_existing"]
    if fp8_vae: cmd += ["--fp8_vae"]
    if v_chunk and v_chunk > 0: cmd += ["--vae_chunk_size", str(int(v_chunk))]
    if v_sp_tile and v_sp_tile > 0: cmd += ["--vae_spatial_tile_size", str(int(v_sp_tile))]
    if v_sp_ov and v_sp_ov > 0: cmd += ["--vae_spatial_tile_overlap", str(int(v_sp_ov))]
    if v_tp_tile and v_tp_tile > 0: cmd += ["--vae_temporal_tile_size", str(int(v_tp_tile))]
    if v_tp_ov and v_tp_ov > 0: cmd += ["--vae_temporal_tile_overlap", str(int(v_tp_ov))]
    if c_mode != "video":
        cmd += ["--ltx2_mode", c_mode]
    start_subprocess(cmd)
    return "已启动 Cache Latents，请查看调试框。"

def run_cache_te(dataset_cfg, ltx2_ckpt, gemma_rt, gemma_sft,
                 te_dtype, skip_existing, load_8bit, load_4bit, c_mode):
    if not (ltx2_ckpt and ltx2_ckpt.strip()):
        return "❌ 错误：请先在界面顶部【0. 全局设置】中填写【LTX2 权重路径】！"
    if not (gemma_rt and gemma_rt.strip()) and not (gemma_sft and gemma_sft.strip()):
        return "❌ 错误：请先在界面顶部【0. 全局设置】中填写【Gemma 权重路径】！"
        
    cmd = [sys.executable, "ltx2_cache_text_encoder_outputs.py",
           "--dataset_config", dataset_cfg or "dataset_ltx2.toml"]
    if ltx2_ckpt and ltx2_ckpt.strip(): cmd += ["--ltx2_checkpoint", ltx2_ckpt.strip()]
    if gemma_rt and gemma_rt.strip(): cmd += ["--gemma_root", gemma_rt.strip()]
    if gemma_sft and gemma_sft.strip(): cmd += ["--gemma_safetensors", gemma_sft.strip()]
    if te_dtype: cmd += ["--mixed_precision", te_dtype]
    if skip_existing: cmd += ["--skip_existing"]
    if load_8bit: cmd += ["--gemma_load_in_8bit"]
    if load_4bit: cmd += ["--gemma_load_in_4bit"]
    if c_mode != "video":
        cmd += ["--ltx2_mode", c_mode]
    start_subprocess(cmd)
    return "已启动 Cache Text Encoder，请查看调试框。"


def generate_dataset_toml(res_w, res_h, cap_ext, b_size, e_bucket, no_upscale, sep_audio, vid_dir, cache_dir, aud_dir, aud_strat, aud_int, t_frames, m_frames, t_fps, save_path, config_path):
    try:
        t_frames_list = [int(x.strip()) for x in t_frames.split(",") if x.strip().isdigit()]
        gen_cfg = {
            "resolution": [int(res_w), int(res_h)],
            "caption_extension": cap_ext,
            "batch_size": int(b_size),
            "enable_bucket": bool(e_bucket),
            "bucket_no_upscale": bool(no_upscale)
        }
        if sep_audio:
            gen_cfg["separate_audio_buckets"] = True

        ds_cfg = {
            "video_directory": vid_dir,
            "cache_directory": cache_dir,
            "target_frames": t_frames_list,
            "max_frames": int(m_frames),
            "target_fps": float(t_fps)
        }
        if aud_dir and aud_dir.strip():
            ds_cfg["audio_directory"] = aud_dir.strip()
            ds_cfg["audio_bucket_strategy"] = aud_strat
            ds_cfg["audio_bucket_interval"] = float(aud_int)

        config = {
            "general": gen_cfg,
            "datasets": [ds_cfg]
        }
        import toml
        with open(save_path, "w", encoding="utf-8") as f:
            toml.dump(config, f)
        cache = _read_ui_cache()
        cache["config_path"] = config_path.strip() if config_path else cache.get("config_path", CONFIG_PATH)
        cache["dataset_form"] = {
            "ds_res_w": int(res_w),
            "ds_res_h": int(res_h),
            "ds_cap_ext": cap_ext,
            "ds_batch_size": int(b_size),
            "ds_enable_bucket": bool(e_bucket),
            "ds_bucket_no_upscale": bool(no_upscale),
            "ds_sep_audio": bool(sep_audio),
            "ds_vid_dir": vid_dir,
            "ds_cache_dir": cache_dir,
            "ds_aud_dir": aud_dir,
            "ds_aud_strat": aud_strat,
            "ds_aud_int": float(aud_int),
            "ds_target_frames": t_frames,
            "ds_max_frames": int(m_frames),
            "ds_target_fps": float(t_fps),
            "ds_save_path": save_path,
        }
        if save_path:
            cache["config_values"] = {**cache.get("config_values", {}), "dataset_config": save_path}
        _write_ui_cache(cache)
        return (
            f"✅ 数据集配置已成功保存至 {save_path}，并已同步到顶部统一配置",
            gr.update(value=save_path),
        )
    except Exception as e:
        return f"❌ 生成失败: {str(e)}", gr.update()

tb_process = None


def run_training(config_path):
    global tb_process
    p = config_path.strip() if config_path else CONFIG_PATH
    if not os.path.exists(p):
        return f"❌ 找不到配置文件: {p}，请先保存配置！"
    # Pre-flight check: ltx2_checkpoint must be present in the TOML
    try:
        cfg_check = toml.load(p)
        if not cfg_check.get("ltx2_checkpoint"):
            return ("❌ 配置文件中缺少 ltx2_checkpoint！"
                    "请在顶部填写 LTX2 模型路径后重新保存配置。")
                    
        # 自动启动 TensorBoard
        log_w = cfg_check.get("log_with", "")
        log_d = cfg_check.get("logging_dir", "./logs")
        if log_w in ("tensorboard", "all"):
            if tb_process is None or tb_process.poll() is not None:
                os.makedirs(log_d, exist_ok=True)
                tb_process = subprocess.Popen(
                    [sys.executable, "-m", "tensorboard.main", "--logdir", log_d, "--port", "6006"],
                    stdout=subprocess.DEVNULL, stderr=subprocess.DEVNULL,
                    cwd=os.path.dirname(os.path.abspath(__file__))
                )
                log_queue.put("==========================================\n")
                log_queue.put(f"🚀 TensorBoard 监控已后台启动！\n👉 请在浏览器访问 http://127.0.0.1:6006 查看训练曲线\n")
                log_queue.put("==========================================\n")
    except Exception as e:
        return f"❌ 无法读取配置文件: {e}"
    # 从配置文件中读取关键参数并作为命令行参数传递（避免 read_config_from_file 的 required 参数检查问题）
    cmd = [sys.executable, "ltx2_train_network.py", "--config_file", p]
    if cfg_check.get("ltx2_checkpoint"):
        cmd += ["--ltx2_checkpoint", cfg_check["ltx2_checkpoint"]]
    if cfg_check.get("ltx_mode") or cfg_check.get("ltx2_mode"):
        cmd += ["--ltx2_mode", cfg_check.get("ltx_mode") or cfg_check["ltx2_mode"]]
    start_subprocess(cmd)
    return f"已启动 LTX2 训练（配置：{p}），请查看调试框。"


# ══════════════════════════════════════════════════
#  UI 布局
# ══════════════════════════════════════════════════
ui_theme = gr.themes.Soft(
    primary_hue="green",
    font=[gr.themes.GoogleFont("Inter"), "ui-sans-serif", "system-ui", "sans-serif"]
)

with gr.Blocks(title="LTX2 训练控制台", theme=ui_theme) as app:
    gr.Markdown("# 🎬 LTX2 模型训练中文控制台 (Musubi-Tuner)")

    # ────────────────────────────────────────────────
    #  顶部全局路径区（所有标签页共用）
    # ────────────────────────────────────────────────
    with gr.Group():
        gr.Markdown("## 📂 全局路径设置（所有步骤共用）")
        with gr.Row():
            g_ltx2 = gr.Textbox(label="★ LTX2 模型路径 (--ltx2_checkpoint，必填)",
                                 placeholder="path/to/ltx2.safetensors", scale=3)
            g_vae = gr.Textbox(label="VAE 路径（可选，留空=从主模型读取）",
                               placeholder="path/to/vae.safetensors", scale=2)
        with gr.Row():
            g_gemma_root = gr.Textbox(label="Gemma 文本编码器根目录 (--gemma_root)",
                                      placeholder="path/to/gemma_folder", scale=2)
            g_gemma_sft = gr.Textbox(label="Gemma Safetensors (--gemma_safetensors)",
                                     placeholder="path/to/gemma.safetensors", scale=2)
            g_dataset = gr.Textbox(label="数据集配置文件 (--dataset_config)",
                                   value="dataset_ltx2.toml", scale=1)
        with gr.Row():
            g_output_dir = gr.Textbox(label="输出目录 (--output_dir)", value="./outputs", scale=2)
            g_output_name = gr.Textbox(label="输出模型名称 (--output_name)", value="ltx2_lora", scale=2)
            g_config_path = gr.Textbox(label="训练配置文件路径（保存/读取）",
                                       value="ltx2_train_config.toml", scale=2)

        with gr.Row():
            load_btn = gr.Button("📂 载入配置文件", size="sm")
            save_top_btn = gr.Button("💾 快速保存配置", variant="primary", size="sm")
            reset_btn = gr.Button("↺ 恢复默认参数", size="sm")
            load_status = gr.Textbox(label="", interactive=False, scale=3)

    gr.Markdown("---")

    with gr.Tabs():

        # ══════════════════════════════════
        #  Tab 1: 训练参数配置
        # ══════════════════════════════════
        with gr.Tab("🗂️ 1. 数据集准备 (Dataset & Cache)"):

            with gr.Accordion("📝 1.1 生成数据集配置文件 (TOML)", open=True):
                gr.Markdown("参考 test.toml 的选项，快速生成数据集配置。")
                with gr.Row():
                    ds_res_w = gr.Number(label="Resolution Width", value=960, precision=0)
                    ds_res_h = gr.Number(label="Resolution Height", value=544, precision=0)
                    ds_cap_ext = gr.Textbox(label="Caption Extension", value=".txt")
                with gr.Row():
                    ds_batch_size = gr.Number(label="Batch Size", value=1, precision=0)
                    ds_enable_bucket = gr.Checkbox(label="Enable Bucket", value=True)
                    ds_bucket_no_upscale = gr.Checkbox(label="Bucket No Upscale", value=False)
                    ds_sep_audio = gr.Checkbox(label="Separate Audio Buckets", value=False)
                with gr.Row():
                    ds_vid_dir = gr.Textbox(label="Video Directory", value="")
                    ds_cache_dir = gr.Textbox(label="Cache Directory", value="")
                with gr.Row():
                    ds_aud_dir = gr.Textbox(label="Audio Dir (可选, AV模式)", value="")
                    ds_aud_strat = gr.Dropdown(label="Audio Bucket Strategy", choices=["pad", "truncate"], value="pad")
                    ds_aud_int = gr.Number(label="Audio Bucket Interval", value=2.0)
                with gr.Row():
                    ds_target_frames = gr.Textbox(label="Target Frames (逗号分隔)", value="49, 97, 145, 193, 201")
                    ds_max_frames = gr.Number(label="Max Frames", value=201, precision=0)
                    ds_target_fps = gr.Number(label="Target FPS", value=25.0)
                with gr.Row():
                    ds_save_path = gr.Textbox(label="保存路径", value="dataset_ltx2.toml")
                    ds_gen_btn = gr.Button("📄 生成数据集配置", variant="primary")
                ds_gen_status = gr.Textbox(label="状态", interactive=False)
                ds_gen_btn.click(
                    fn=generate_dataset_toml,
                    inputs=[ds_res_w, ds_res_h, ds_cap_ext, ds_batch_size, ds_enable_bucket, ds_bucket_no_upscale,
                            ds_sep_audio, ds_vid_dir, ds_cache_dir, ds_aud_dir, ds_aud_strat, ds_aud_int, 
                            ds_target_frames, ds_max_frames, ds_target_fps, ds_save_path, g_config_path],
                    outputs=[ds_gen_status, g_dataset]
                )
                
            with gr.Accordion("🗂️ 1.2 预缓存视频 Latent 与文本特征", open=True):
                gr.Markdown("""
                预缓存视频 Latent 和文本编码器输出，大幅加速训练。  
            **模型路径和数据集配置文件均读取自顶部全局设置，无需重复填写。**
            """)
                c_mode = gr.Dropdown(label="缓存模式 (--ltx2_mode)", choices=["video", "av", "audio"], value="video", info="★ 训练音视频(AV)模型前，必须先以 av 模式完成缓存（生成 _audio.safetensors）。")
                with gr.Row():
                    with gr.Column():
                        gr.Markdown("#### Cache Latents（编码视频特征）")
                        with gr.Row():
                            c_vae_dtype = gr.Dropdown(label="VAE Dtype", choices=["", "fp16", "bf16", "fp32"], value="bf16")
                            c_skip_lc = gr.Checkbox(label="跳过已缓存文件 (--skip_existing)", value=True)
                            c_fp8_vae = gr.Checkbox(label="VAE 使用 FP8", value=False)
                        with gr.Accordion("🛠️ 内存优化 (VAE Chunking/Tiling)", open=False):
                            gr.Markdown("如果显存爆炸 (OOM)，请设置 `Chunk Size` 或 `Spatial Tile` 进行显存分块压缩（牺牲少许速度降显存）。例如：Chunk 设为 16 或 Spatial 设为 512。")
                            with gr.Row():
                                c_vae_chunk = gr.Number(label="Temporal Chunk Size", value=0, precision=0, info="16或32")
                                c_vae_tp_tile = gr.Number(label="Temporal Tile Size", value=0, precision=0, info="时间维度切片")
                                c_vae_tp_ov = gr.Number(label="Temporal Tile Overlap", value=24, precision=0)
                            with gr.Row():
                                c_vae_sp_tile = gr.Number(label="Spatial Tile Size", value=0, precision=0, info="例如512")
                                c_vae_sp_ov = gr.Number(label="Spatial Tile Overlap", value=64, precision=0)
                        cache_latents_btn = gr.Button("▶ 运行 Cache Latents", variant="primary")

                    with gr.Column():
                        gr.Markdown("#### Cache Text Encoder（编码文本特征）")
                        te_dtype = gr.Dropdown(label="Text Encoder Dtype",
                            choices=["", "fp16", "bf16", "fp32"], value="bf16")
                        te_skip = gr.Checkbox(label="跳过已缓存文件 (--skip_existing)", value=True)
                        with gr.Row():
                            te_8bit = gr.Checkbox(label="Gemma 8bit", value=False)
                            te_4bit = gr.Checkbox(label="Gemma 4bit", value=False)
                        cache_te_btn = gr.Button("▶ 运行 Cache Text Encoder", variant="primary")

                cache_status = gr.Textbox(label="操作状态", interactive=False)

                cache_latents_btn.click(
                    fn=run_cache_latents,
                    inputs=[g_dataset, g_ltx2, g_vae, c_vae_dtype, c_skip_lc, c_fp8_vae, c_vae_chunk, c_vae_sp_tile, c_vae_sp_ov, c_vae_tp_tile, c_vae_tp_ov, c_mode],
                    outputs=cache_status
                )
                cache_te_btn.click(
                    fn=run_cache_te,
                    inputs=[g_dataset, g_ltx2, g_gemma_root, g_gemma_sft,
                            te_dtype, te_skip, te_8bit, te_4bit, c_mode],
                    outputs=cache_status
                )
                
        # ══════════════════════════════════
        #  Tab 2: 训练参数配置
        # ══════════════════════════════════
        with gr.Tab("⚙️ 2. 训练参数配置"):

            with gr.Accordion("🎞️ 训练模式与 LoRA 类型", open=True):
                with gr.Row():
                    ltx2_mode = gr.Dropdown(label="训练模式 (--ltx2_mode)",
                        choices=["video", "av", "audio"], value="video",
                        info="video=纯视频, av=音视频, audio=纯音频")
                    lora_target_preset = gr.Dropdown(label="LoRA 目标预设 (--lora_target_preset)",
                        choices=["", "t2v", "v2v", "audio", "audio_ref_only_ic", "full"], value="t2v")
                    ic_lora_strategy = gr.Dropdown(label="IC-LoRA 策略",
                        choices=["auto", "none", "v2v", "audio_ref_only_ic"], value="auto")
                with gr.Row():
                    gemma_load_in_8bit = gr.Checkbox(label="Gemma 8bit 量化", value=False)
                    gemma_load_in_4bit = gr.Checkbox(label="Gemma 4bit 量化", value=False)

            with gr.Accordion("🧠 LoRA 网络设置", open=True):
                with gr.Row():
                    network_module = gr.Dropdown(label="网络模块 (--network_module)",
                        choices=["networks.lora_ltx2", "lycoris.kohya"], value="networks.lora_ltx2")
                    network_dim = gr.Number(label="Network Dim / Rank", value=32, precision=0)
                    network_alpha = gr.Number(label="Network Alpha", value=16)
                    network_dropout = gr.Number(label="Network Dropout", value=0)
                    scale_weight_norms = gr.Number(label="Scale Weight Norms", value=0)
                with gr.Row():
                    network_weights = gr.Textbox(label="续训 LoRA 权重路径", placeholder="空=从头训练")
                    dim_from_weights = gr.Checkbox(label="从权重读取 Dim", value=False)
                    resume = gr.Textbox(label="续训状态路径 (--resume)", placeholder="空=不续训")
                with gr.Row():
                    enable_lora_plus = gr.Checkbox(label="启用 LoRA+", value=True)
                    loraplus_lr_ratio = gr.Number(label="LoRA+ LR Ratio", value=4)
                    enable_blocks = gr.Checkbox(label="启用块控制", value=False)
                    exclude_patterns = gr.Textbox(label="Exclude Patterns")
                    include_patterns = gr.Textbox(label="Include Patterns")

            with gr.Accordion("⏱️ 训练时长与批次", open=True):
                with gr.Row():
                    max_train_epochs = gr.Number(label="最大 Epoch 数", value=10, precision=0)
                    max_train_steps = gr.Textbox(label="最大训练步数 (空=不限)", value="5000")
                    batch_size = gr.Number(label="批次大小", value=1, precision=0)
                    seed = gr.Number(label="随机种子", value=1026, precision=0)
                with gr.Row():
                    gradient_checkpointing = gr.Checkbox(label="梯度检查点", value=True)
                    gradient_accumulation_steps = gr.Number(label="梯度累积步数", value=1, precision=0)
                    caption_dropout_rate = gr.Number(label="Caption Dropout Rate", value=0.0)

            with gr.Accordion("⚡ 精度与模型版本", open=False):
                with gr.Row():
                    mixed_precision = gr.Dropdown(label="混合精度",
                        choices=["bf16", "fp16", "no"], value="bf16")
                    vae_dtype = gr.Dropdown(label="VAE Dtype",
                        choices=["", "fp16", "bf16", "fp32"], value="")
                    ltx_version = gr.Dropdown(label="LTX 版本 (--ltx_version)",
                        choices=["2.0", "2.3"], value="2.3")
                    ltx_version_check_mode = gr.Dropdown(label="版本检查 (--ltx_version_check_mode)",
                        choices=["off", "warn", "error"], value="warn")
                with gr.Row():
                    fp8_base = gr.Checkbox(label="FP8 Base (--fp8_base)", value=True)
                    fp8_scaled = gr.Checkbox(label="FP8 Scaled (--fp8_scaled)", value=True)
                    fp8_w8a8 = gr.Checkbox(label="FP8 W8A8 (--fp8_w8a8)", value=False)
                    w8a8_mode = gr.Dropdown(label="W8A8 Mode", choices=["int8", "fp8"], value="int8")
                    nf4_base = gr.Checkbox(label="NF4 Base 量化 (--nf4_base)", value=False)
                    nf4_block_size = gr.Number(label="NF4 Block Size", value=64, precision=0)

            with gr.Accordion("🖥️ 显存优化与注意力", open=False):
                with gr.Row():
                    attention_mode = gr.Dropdown(label="注意力机制",
                        choices=["sdpa", "flash_attn", "flash3", "xformers"], value="sdpa")
                    blocks_to_swap = gr.Number(label="Blocks to Swap (推荐26+)", value=0, precision=0)
                    use_pinned_memory_for_block_swap = gr.Checkbox(label="Pinned Memory", value=True)
                    blockwise_checkpointing = gr.Checkbox(label="Blockwise Checkpointing", value=False)
                with gr.Row():
                    split_attn_target = gr.Dropdown(label="Split Attn Target",
                        choices=["none","all","self","cross","text_cross","av_cross","video","audio"],
                        value="none")
                    split_attn_mode = gr.Dropdown(label="Split Attn Mode",
                        choices=["batch","query"], value="batch")
                    split_attn_chunk_size = gr.Number(label="Split Attn Chunk Size", value=0, precision=0)
                    ffn_chunk_target = gr.Dropdown(label="FFN Chunk Target",
                        choices=["none","all","video","audio"], value="none")
                    ffn_chunk_size = gr.Number(label="FFN Chunk Size", value=0, precision=0)
                with gr.Row():
                    cuda_allow_tf32 = gr.Checkbox(label="cuda_allow_tf32", value=True)
                    cuda_cudnn_benchmark = gr.Checkbox(label="cuda_cudnn_benchmark", value=True)

            with gr.Accordion("📊 Loss 设置", open=False):
                with gr.Row():
                    loss_type = gr.Dropdown(label="Loss 类型",
                        choices=["mse","mae","l1","huber","smooth_l1"], value="mse")
                    huber_delta = gr.Number(label="Huber Delta", value=1.0)
                    preserve_distribution_shape = gr.Checkbox(label="Preserve Distribution Shape", value=False)
                    video_loss_weight = gr.Number(label="Video Loss Weight", value=1.0)
                    audio_loss_weight = gr.Number(label="Audio Loss Weight", value=1.0)

            with gr.Accordion("📈 学习率设置", open=False):
                with gr.Row():
                    lr = gr.Textbox(label="学习率", value="1e-4")
                    lr_scheduler = gr.Dropdown(label="调度器",
                        choices=["constant","constant_with_warmup","linear","cosine",
                                 "cosine_with_restarts","polynomial","cosine_with_min_lr",
                                 "warmup_stable_decay","inverse_sqrt"],
                        value="constant_with_warmup")
                with gr.Row():
                    lr_warmup_steps = gr.Number(label="预热步数", value=50, precision=0)
                    lr_decay_steps = gr.Number(label="衰减步数", value=0.2)
                    lr_scheduler_num_cycles = gr.Number(label="余弦重启次数", value=1, precision=0)
                    lr_scheduler_power = gr.Number(label="Poly Power", value=1.0)
                    lr_scheduler_timescale = gr.Number(label="Timescale", value=0, precision=0)
                    lr_scheduler_min_lr_ratio = gr.Number(label="Min LR Ratio", value=0.1)

            with gr.Accordion("⏰ 时间步采样", open=False):
                with gr.Row():
                    timestep_sampling = gr.Dropdown(label="时间步采样方式",
                        choices=["sigma","uniform","sigmoid","shift","flux_shift",
                                 "flux2_shift","logsnr","shifted_logit_normal"],
                        value="shift")
                    discrete_flow_shift = gr.Number(label="Discrete Flow Shift", value=3.0)
                    sigmoid_scale = gr.Number(label="Sigmoid Scale", value=1.0)
                with gr.Row():
                    weighting_scheme = gr.Dropdown(label="加权方案",
                        choices=["","sigma_sqrt","logit_normal","mode","cosmap","none"], value="")
                    logit_mean = gr.Number(label="Logit Mean", value=0.0)
                    logit_std = gr.Number(label="Logit Std", value=1.0)
                    mode_scale = gr.Number(label="Mode Scale", value=1.29)
                    min_timestep = gr.Number(label="Min Timestep", value=0, precision=0)
                    max_timestep = gr.Number(label="Max Timestep", value=1000, precision=0)

            with gr.Accordion("🔧 优化器", open=False):
                with gr.Row():
                    optimizer_type = gr.Dropdown(label="优化器类型",
                        choices=["AdamW","AdamW8bit","PagedAdamW8bit","adafactor",
                                 "Lion","Lion8bit","prodigyopt.Prodigy","DAdaptAdam",
                                 "Sophia","Ranger","StableAdamW","SOAP",
                                 "adopt","came","ademamix","fira","sgdsai",
                                 "Muon","schedulefree.AdamWScheduleFree","schedulefree.RAdamScheduleFree"],
                        value="AdamW8bit")
                    max_grad_norm = gr.Number(label="Max Grad Norm", value=1.0)
                
                # Prodigy 优化器专用参数（仅当选择 Prodigy 时显示）
                with gr.Group(visible=False) as prodigy_params_group:
                    gr.Markdown("#### 🧪 Prodigy 优化器参数")
                    with gr.Row():
                        prodigy_beta1 = gr.Number(label="Beta1", value=0.9, minimum=0.0, maximum=1.0, step=0.01)
                        prodigy_beta2 = gr.Number(label="Beta2", value=0.999, minimum=0.0, maximum=1.0, step=0.001)
                        prodigy_beta3 = gr.Number(label="Beta3", value=0.9, minimum=0.0, maximum=1.0, step=0.01, info="二阶矩估计衰减率")
                    with gr.Row():
                        prodigy_d0 = gr.Number(label="d0 (初始学习率)", value=1e-6, minimum=1e-10, step=1e-7, info="Prodigy 初始距离/学习率")
                        prodigy_growth_rate = gr.Number(label="Growth Rate", value=999.0, minimum=1.0, step=0.1, info="学习率增长率 (默认inf=无限制)")
                    with gr.Row():
                        prodigy_use_bias_correction = gr.Checkbox(label="Use Bias Correction (偏差修正)", value=False)
                        prodigy_use_speed = gr.Checkbox(label="Use Speed (加速模式)", value=False)
                        prodigy_safeguard_warmup = gr.Checkbox(label="Safeguard Warmup (Warmup保护)", value=False)

                # 优化器类型变化时控制 Prodigy 参数显示/隐藏
                def toggle_prodigy_params(opt_type):
                    return gr.update(visible=(opt_type == "prodigyopt.Prodigy"))
                
                optimizer_type.change(
                    fn=toggle_prodigy_params,
                    inputs=[optimizer_type],
                    outputs=[prodigy_params_group]
                )

            with gr.Accordion("💾 保存与追踪监控", open=False):
                with gr.Row():
                    save_every_n_epochs = gr.Number(label="每N Epoch保存", value=2, precision=0)
                    save_every_n_steps = gr.Textbox(label="每N步保存 (空=用Epoch)", value="250")
                    save_last_n_epochs = gr.Textbox(label="只保留最后N轮", value="")
                    save_last_n_steps = gr.Textbox(label="只保留最后N步", value="")
                with gr.Row():
                    save_state = gr.Checkbox(label="保存训练状态", value=False)
                    save_state_on_train_end = gr.Checkbox(label="仅结束时保存状态", value=False)
                    log_with = gr.Dropdown(label="日志监控 (--log_with)", choices=["", "tensorboard", "wandb", "all"], value="tensorboard")
                    logging_dir = gr.Textbox(label="日志路径 (--logging_dir)", value="./logs")

            with gr.Accordion("🎞️ 训练内采样设置", open=False):
                with gr.Row():
                    enable_sample = gr.Checkbox(label="启用训练中出图", value=False)
                    sample_at_first = gr.Checkbox(label="训练开始前出图", value=False)
                    enable_sample_steps = gr.Textbox(label="每N步出图 (空=用Epoch)", value="")
                    sample_every_n_epochs = gr.Number(label="每N Epoch出图", value=2, precision=0)
                with gr.Row():
                    guidance_scale = gr.Number(label="采样引导系数", value=1.0, info="仅影响训练中预览采样，不影响 LTX2 训练损失。")
                    sample_prompts = gr.Textbox(label="提示词文件路径", value="")
                with gr.Row():
                    height = gr.Number(label="采样高度", value=512, precision=0)
                    width = gr.Number(label="采样宽度", value=512, precision=0)
                    sample_num_frames = gr.Number(label="采样帧数", value=49, precision=0)

            with gr.Accordion("🔀 数据加载与杂项", open=False):
                with gr.Row():
                    max_data_loader_n_workers = gr.Number(label="数据加载线程数", value=8, precision=0)
                    persistent_data_loader_workers = gr.Checkbox(label="持久化工人", value=True)
                with gr.Row():
                    training_comment = gr.Textbox(label="训练备注", value="")
                    metadata_title = gr.Textbox(label="Metadata Title", value="")
                    metadata_author = gr.Textbox(label="Metadata Author", value="")
                    metadata_description = gr.Textbox(label="Metadata Description", value="")

            with gr.Accordion("🧪 高阶/实验参数", open=False):
                gr.Markdown("这里集中放 LTX2 可用但不必每次都调的高级项，避免常用页面过于拥挤。")
                with gr.Row():
                    gradient_checkpointing_cpu_offload = gr.Checkbox(label="梯度检查点 CPU Offload", value=False)
                    blocks_to_checkpoint = gr.Number(label="Checkpoint Blocks", value=-1, precision=0, info="-1=全部，0=禁用，N=最后N块")
                    shifted_logit_mode = gr.Dropdown(label="Shifted Logit Mode", choices=["", "legacy", "stretched"], value="", info="留空=按 LTX 版本自动选择")
                with gr.Row():
                    audio_lr = gr.Textbox(label="音频模块学习率 (--audio_lr)", value="", placeholder="空=跟随主学习率")
                    lr_args = gr.Textbox(label="模块学习率覆写 (--lr_args)", value="", placeholder="例如: audio_attn=1e-6 video_to_audio=5e-6")
                with gr.Row():
                    audio_loss_balance_mode = gr.Dropdown(
                        label="音频 Loss 平衡",
                        choices=["none", "inv_freq", "ema_mag"],
                        value="none"
                    )
                    audio_loss_balance_beta = gr.Number(label="Balance Beta", value=0.01)
                    audio_loss_balance_eps = gr.Number(label="Balance Eps", value=0.05)
                with gr.Row():
                    audio_loss_balance_min = gr.Number(label="Balance Min", value=0.05)
                    audio_loss_balance_max = gr.Number(label="Balance Max", value=4.0)
                    audio_loss_balance_target_ratio = gr.Number(label="目标音频/视频比", value=0.33)
                with gr.Row():
                    audio_ref_identity_guidance_scale = gr.Number(label="音频参考身份引导", value=0.0, info="仅 audio_ref_only_ic 采样时生效")
                    sample_disable_flash_attn = gr.Checkbox(label="采样禁用 FlashAttn", value=False)
                with gr.Row():
                    use_precached_sample_prompts = gr.Checkbox(label="使用预缓存 Prompt Embeds", value=False)
                    sample_prompts_cache = gr.Textbox(label="Prompt Cache 路径", value="", placeholder="空=自动从 cache_directory 推断")
                with gr.Row():
                    use_precached_sample_latents = gr.Checkbox(label="使用预缓存 I2V Latents", value=False)
                    sample_latents_cache = gr.Textbox(label="Latents Cache 路径", value="", placeholder="空=自动从 cache_directory 推断")



            with gr.Row():
                save_btn = gr.Button("💾 保存配置", variant="primary", size="lg")
                save_status_tab = gr.Textbox(label="状态", interactive=False, scale=3)

        # ══════════════════════════════════
        #  Tab 3: 训练
        # ══════════════════════════════════
        with gr.Tab("🚀 3. 训练及进度显示"):
            gr.Markdown("点击【启动训练】后，将依据顶部配置文件路径所指定的 TOML 文件启动训练。")
            with gr.Row():
                train_btn = gr.Button("🚀 一键启动 LTX2 训练", variant="primary", size="lg")
                stop_btn = gr.Button("⏹ 终止当前进程", variant="stop", size="lg")
            train_status = gr.Textbox(label="状态", interactive=False)
            train_btn.click(fn=run_training, inputs=[g_config_path], outputs=train_status)
            stop_btn.click(fn=stop_process, outputs=train_status)

    # ────────────────────────────────────
    #  调试终端输出
    # ────────────────────────────────────
    gr.Markdown("---")
    gr.Markdown("### 🖥️ 调试命令终端输出")
    with gr.Row():
        clear_btn = gr.Button("🗑 清空日志", size="sm")
    log_output = gr.Textbox(
        label="终端输出（0.5s 自动刷新）",
        lines=35, max_lines=35, autoscroll=True, interactive=False
    )
    clear_btn.click(fn=clear_logs, outputs=log_output)
    timer = gr.Timer(0.5)
    timer.tick(update_logs, inputs=[log_output], outputs=[log_output])

    # ──────────────────────────────────────
    #  收集所有参数（用于保存/载入）
    # ──────────────────────────────────────
    _all_args = [
        g_ltx2, g_vae, g_output_dir, g_output_name, g_dataset,
        g_gemma_root, g_gemma_sft, gemma_load_in_8bit, gemma_load_in_4bit,
        ltx2_mode, lora_target_preset, ic_lora_strategy, network_module,
        network_dim, network_alpha, network_dropout, scale_weight_norms,
        network_weights, dim_from_weights,
        enable_lora_plus, loraplus_lr_ratio,
        enable_blocks, exclude_patterns, include_patterns,
        max_train_epochs, max_train_steps, batch_size, seed,
        gradient_checkpointing, gradient_checkpointing_cpu_offload, blocks_to_checkpoint, gradient_accumulation_steps,
        guidance_scale, caption_dropout_rate,
        mixed_precision, vae_dtype, ltx_version, ltx_version_check_mode,
        fp8_base, fp8_scaled, fp8_w8a8, w8a8_mode, nf4_base, nf4_block_size,
        attention_mode, blocks_to_swap, use_pinned_memory_for_block_swap,
        split_attn_target, split_attn_mode, split_attn_chunk_size,
        ffn_chunk_target, ffn_chunk_size, blockwise_checkpointing,
        loss_type, huber_delta, preserve_distribution_shape,
        video_loss_weight, audio_loss_weight, audio_loss_balance_mode, audio_loss_balance_beta,
        audio_loss_balance_eps, audio_loss_balance_min, audio_loss_balance_max, audio_loss_balance_target_ratio,
        lr, audio_lr, lr_args, lr_scheduler, lr_warmup_steps, lr_decay_steps,
        lr_scheduler_num_cycles, lr_scheduler_power,
        lr_scheduler_timescale, lr_scheduler_min_lr_ratio,
        timestep_sampling, shifted_logit_mode, discrete_flow_shift, sigmoid_scale,
        weighting_scheme, logit_mean, logit_std, mode_scale,
        min_timestep, max_timestep,
        optimizer_type, max_grad_norm,
        prodigy_beta1, prodigy_beta2, prodigy_beta3, prodigy_d0,
        prodigy_growth_rate, prodigy_use_bias_correction, prodigy_use_speed, prodigy_safeguard_warmup,
        save_every_n_epochs, save_every_n_steps,
        save_last_n_epochs, save_last_n_steps,
        save_state, save_state_on_train_end, resume,
        log_with, logging_dir,
        enable_sample, sample_at_first,
        sample_every_n_epochs, enable_sample_steps, sample_prompts,
        use_precached_sample_prompts, sample_prompts_cache, use_precached_sample_latents, sample_latents_cache,
        sample_disable_flash_attn, audio_ref_identity_guidance_scale,
        height, width, sample_num_frames,
        max_data_loader_n_workers, persistent_data_loader_workers,
        cuda_allow_tf32, cuda_cudnn_benchmark,
        training_comment, metadata_title, metadata_author, metadata_description,
    ]

    # 保存按钮 (Tab1 + 顶部快捷键)
    def _save(*args): return save_config(*args)

    save_btn.click(fn=_save, inputs=_all_args + [g_config_path], outputs=save_status_tab)
    save_top_btn.click(fn=_save, inputs=_all_args + [g_config_path], outputs=load_status)

    dataset_components = [
        ds_res_w, ds_res_h, ds_cap_ext, ds_batch_size, ds_enable_bucket, ds_bucket_no_upscale,
        ds_sep_audio, ds_vid_dir, ds_cache_dir, ds_aud_dir, ds_aud_strat, ds_aud_int,
        ds_target_frames, ds_max_frames, ds_target_fps, ds_save_path,
    ]

    def _persist_ui_cache(config_path, dataset_save_path, *args):
        train_values = dict(zip(TRAIN_ARG_KEYS, args[:len(TRAIN_ARG_KEYS)]))
        dataset_values = dict(zip(DATASET_DEFAULTS.keys(), args[len(TRAIN_ARG_KEYS):]))
        cache = {
            "config_path": config_path.strip() if config_path else CONFIG_PATH,
            "config_values": train_values,
            "dataset_form": dataset_values,
        }
        if dataset_save_path:
            cache["config_values"]["dataset_config"] = dataset_save_path
        _write_ui_cache(cache)
        return None

    # 载入配置按钮（读取 TOML 回填各组件）
    def _do_load(path):
        p = path.strip() if path else CONFIG_PATH
        cfg = load_config_file(p)
        if cfg is None:
            return [gr.update()] * (len(_all_args) + len(dataset_components) + 2) + [f"❌ 载入失败：找不到或无法读取 {p}"]

        cache = _read_ui_cache()
        dataset_cache = dict(DATASET_DEFAULTS)
        dataset_cache.update(cache.get("dataset_form", {}))
        if cfg.get("dataset_config"):
            dataset_cache["ds_save_path"] = cfg["dataset_config"]

        cache["config_path"] = p
        cache["config_values"] = cfg
        cache["dataset_form"] = dataset_cache
        _write_ui_cache(cache)

        return (
            _cfg_to_updates(cfg)
            + _dataset_updates_from_cache(cache)
            + [gr.update(value=p), gr.update(value=dataset_cache["ds_save_path"])]
            + [f"✅ 已载入: {p}"]
        )

    def _restore_defaults():
        default_cache = {
            "config_path": CONFIG_PATH,
            "config_values": dict(TRAIN_DEFAULTS),
            "dataset_form": dict(DATASET_DEFAULTS),
        }
        _write_ui_cache(default_cache)
        return (
            _default_train_updates()
            + _dataset_updates_from_cache(default_cache)
            + [gr.update(value=CONFIG_PATH), gr.update(value=DATASET_DEFAULTS["ds_save_path"])]
            + ["✅ 已恢复默认参数，并刷新了本地缓存"]
        )

    def _auto_restore():
        cache = _read_ui_cache()
        config_path = cache.get("config_path", CONFIG_PATH)
        if config_path and os.path.exists(config_path):
            cfg = load_config_file(config_path)
            if cfg is not None:
                dataset_cache = dict(DATASET_DEFAULTS)
                dataset_cache.update(cache.get("dataset_form", {}))
                if cfg.get("dataset_config"):
                    dataset_cache["ds_save_path"] = cfg["dataset_config"]
                cache["config_values"] = cfg
                cache["dataset_form"] = dataset_cache
                _write_ui_cache(cache)
                return (
                    _cfg_to_updates(cfg)
                    + _dataset_updates_from_cache(cache)
                    + [gr.update(value=config_path), gr.update(value=dataset_cache["ds_save_path"])]
                    + [f"✅ 已自动载入上次配置: {config_path}"]
                )

        if cache.get("config_values"):
            merged_cfg = dict(TRAIN_DEFAULTS)
            merged_cfg.update(cache["config_values"])
            return (
                _cfg_to_updates(merged_cfg)
                + _dataset_updates_from_cache(cache)
                + [gr.update(value=config_path), gr.update(value=cache.get("dataset_form", {}).get("ds_save_path", DATASET_DEFAULTS["ds_save_path"]))]
                + ["✅ 配置文件不存在，已从本地缓存恢复上次参数"]
            )

        return (
            _default_train_updates()
            + _dataset_updates_from_cache({})
            + [gr.update(value=CONFIG_PATH), gr.update(value=DATASET_DEFAULTS["ds_save_path"])]
            + ["ℹ️ 未找到历史配置，已使用默认参数"]
        )

    load_btn.click(
        fn=_do_load,
        inputs=[g_config_path],
        outputs=_all_args + dataset_components + [g_config_path, ds_save_path, load_status]
    )
    reset_btn.click(
        fn=_restore_defaults,
        outputs=_all_args + dataset_components + [g_config_path, ds_save_path, load_status]
    )
    app.load(
        fn=_auto_restore,
        outputs=_all_args + dataset_components + [g_config_path, ds_save_path, load_status]
    )

    for component in [g_config_path] + _all_args + dataset_components:
        component.change(
            fn=_persist_ui_cache,
            inputs=[g_config_path, ds_save_path] + _all_args + dataset_components,
            outputs=[]
        )

if __name__ == "__main__":
    app.launch(server_name="127.0.0.1", server_port=7860, inbrowser=True)
