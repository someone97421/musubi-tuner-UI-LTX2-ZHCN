# -*- coding: utf-8 -*-
"""FLUX.2 模型训练 WebUI (中文版) - 图像生成模型训练界面"""
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
CONFIG_PATH = "flux2_train_config.toml"
UI_CACHE_PATH = "flux2_ui_cache.json"

DATASET_DEFAULTS = {
    "ds_res_w": 1024,
    "ds_res_h": 1024,
    "ds_cap_ext": ".txt",
    "ds_batch_size": 1,
    "ds_enable_bucket": True,
    "ds_bucket_no_upscale": False,
    "ds_img_dir": "",
    "ds_cache_dir": "",
    "ds_save_path": "dataset_flux2.toml",
    "ds_control_images_dir": "",
}

TRAIN_DEFAULTS = {
    "dit": "",
    "vae": "",
    "text_encoder": "",
    "output_dir": "./outputs",
    "output_name": "flux2_lora",
    "dataset_config": "dataset_flux2.toml",
    "model_version": "dev",
    "network_module": "networks.lora_flux_2",
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
    "max_train_epochs": 16,
    "max_train_steps": "",
    "train_batch_size": 1,
    "seed": 42,
    "gradient_checkpointing": True,
    "gradient_checkpointing_cpu_offload": False,
    "gradient_accumulation_steps": 1,
    "caption_dropout_rate": 0.0,
    "mixed_precision": "bf16",
    "vae_dtype": "",
    "fp8_base": True,
    "fp8_scaled": True,
    "fp8_text_encoder": False,
    "attention_mode": "sdpa",
    "blocks_to_swap": 0,
    "use_pinned_memory_for_block_swap": True,
    "loss_type": "mse",
    "huber_delta": 1.0,
    "learning_rate": "1e-4",
    "lr_scheduler": "constant_with_warmup",
    "lr_warmup_steps": 50,
    "lr_decay_steps": 0.2,
    "lr_scheduler_num_cycles": 1,
    "lr_scheduler_power": 1.0,
    "lr_scheduler_timescale": 0,
    "lr_scheduler_min_lr_ratio": 0.1,
    "timestep_sampling": "flux2_shift",
    "weighting_scheme": "none",
    "logit_mean": 0.0,
    "logit_std": 1.0,
    "mode_scale": 1.29,
    "min_timestep": 0,
    "max_timestep": 1000,
    "optimizer_type": "AdamW8bit",
    "max_grad_norm": 1.0,
    "save_every_n_epochs": 1,
    "save_every_n_steps": "",
    "save_last_n_epochs": "",
    "save_last_n_steps": "",
    "save_state": False,
    "save_state_on_train_end": False,
    "resume": "",
    "log_with": "tensorboard",
    "logging_dir": "./logs",
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
    "dit", "vae", "text_encoder", "output_dir", "output_name", "dataset_config",
    "model_version", "network_module",
    "network_dim", "network_alpha", "network_dropout", "scale_weight_norms",
    "network_weights", "dim_from_weights",
    "enable_lora_plus", "loraplus_lr_ratio",
    "enable_blocks", "exclude_patterns", "include_patterns",
    "max_train_epochs", "max_train_steps", "train_batch_size", "seed",
    "gradient_checkpointing", "gradient_checkpointing_cpu_offload", "gradient_accumulation_steps",
    "caption_dropout_rate",
    "mixed_precision", "vae_dtype", "model_version",
    "fp8_base", "fp8_scaled", "fp8_text_encoder",
    "attention_mode", "blocks_to_swap", "use_pinned_memory_for_block_swap",
    "loss_type", "huber_delta",
    "learning_rate", "lr_scheduler", "lr_warmup_steps", "lr_decay_steps",
    "lr_scheduler_num_cycles", "lr_scheduler_power",
    "lr_scheduler_timescale", "lr_scheduler_min_lr_ratio",
    "timestep_sampling", "weighting_scheme", "logit_mean", "logit_std", "mode_scale",
    "min_timestep", "max_timestep",
    "optimizer_type", "max_grad_norm",
    "save_every_n_epochs", "save_every_n_steps",
    "save_last_n_epochs", "save_last_n_steps",
    "save_state", "save_state_on_train_end", "resume",
    "log_with", "logging_dir",
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


def _cfg_to_updates(cfg):
    network_args = _parse_network_args(cfg)
    attn_val = TRAIN_DEFAULTS["attention_mode"]
    if cfg.get("flash_attn", False):
        attn_val = "flash_attn"
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

    return [
        gr.update(value=g("dit")),
        gr.update(value=g("vae")),
        gr.update(value=g("text_encoder")),
        gr.update(value=g("output_dir")),
        gr.update(value=g("output_name")),
        gr.update(value=g("dataset_config")),
        gr.update(value=g("model_version")),
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
        gr.update(value=g("gradient_accumulation_steps")),
        gr.update(value=g("caption_dropout_rate")),
        gr.update(value=g("mixed_precision")),
        gr.update(value=g("vae_dtype")),
        gr.update(value=g("fp8_base")),
        gr.update(value=g("fp8_scaled")),
        gr.update(value=g("fp8_text_encoder")),
        gr.update(value=attn_val),
        gr.update(value=g("blocks_to_swap")),
        gr.update(value=g("use_pinned_memory_for_block_swap")),
        gr.update(value=g("loss_type")),
        gr.update(value=g("huber_delta")),
        gr.update(value=str(g("learning_rate", "learning_rate"))),
        gr.update(value=g("lr_scheduler")),
        gr.update(value=g("lr_warmup_steps")),
        gr.update(value=g("lr_decay_steps")),
        gr.update(value=g("lr_scheduler_num_cycles")),
        gr.update(value=g("lr_scheduler_power")),
        gr.update(value=g("lr_scheduler_timescale")),
        gr.update(value=g("lr_scheduler_min_lr_ratio")),
        gr.update(value=g("timestep_sampling")),
        gr.update(value=g("weighting_scheme")),
        gr.update(value=g("logit_mean")),
        gr.update(value=g("logit_std")),
        gr.update(value=g("mode_scale")),
        gr.update(value=g("min_timestep")),
        gr.update(value=g("max_timestep")),
        gr.update(value=g("optimizer_type")),
        gr.update(value=g("max_grad_norm")),
        gr.update(value=g("save_every_n_epochs")),
        gr.update(value=str(g("save_every_n_steps", "save_every_n_steps"))),
        gr.update(value=str(g("save_last_n_epochs", "save_last_n_epochs"))),
        gr.update(value=str(g("save_last_n_steps", "save_last_n_steps"))),
        gr.update(value=g("save_state")),
        gr.update(value=g("save_state_on_train_end")),
        gr.update(value=g("resume")),
        gr.update(value=g("log_with")),
        gr.update(value=g("logging_dir")),
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
        gr.update(value=dataset_values["ds_img_dir"]),
        gr.update(value=dataset_values["ds_cache_dir"]),
        gr.update(value=dataset_values["ds_save_path"]),
        gr.update(value=dataset_values["ds_control_images_dir"]),
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
    dit, vae, text_encoder, output_dir, output_name, dataset_config,
    model_version, network_module,
    network_dim, network_alpha, network_dropout, scale_weight_norms,
    network_weights, dim_from_weights,
    enable_lora_plus, loraplus_lr_ratio,
    enable_blocks, exclude_patterns, include_patterns,
    max_train_epochs, max_train_steps, batch_size, seed,
    gradient_checkpointing, gradient_checkpointing_cpu_offload, gradient_accumulation_steps,
    caption_dropout_rate,
    mixed_precision, vae_dtype, 
    fp8_base, fp8_scaled, fp8_text_encoder,
    attention_mode, blocks_to_swap, use_pinned_memory_for_block_swap,
    loss_type, huber_delta,
    lr, lr_scheduler, lr_warmup_steps, lr_decay_steps,
    lr_scheduler_num_cycles, lr_scheduler_power,
    lr_scheduler_timescale, lr_scheduler_min_lr_ratio,
    timestep_sampling, weighting_scheme, logit_mean, logit_std, mode_scale,
    min_timestep, max_timestep,
    optimizer_type, max_grad_norm,
    save_every_n_epochs, save_every_n_steps,
    save_last_n_epochs, save_last_n_steps,
    save_state, save_state_on_train_end, resume,
    log_with, logging_dir,
    max_data_loader_n_workers, persistent_data_loader_workers,
    cuda_allow_tf32, cuda_cudnn_benchmark,
    training_comment, metadata_title, metadata_author, metadata_description,
):
    cfg = {}
    if dit: cfg["dit"] = dit
    if vae: cfg["vae"] = vae
    if text_encoder: cfg["text_encoder"] = text_encoder
    if output_dir: cfg["output_dir"] = output_dir
    if output_name: cfg["output_name"] = output_name
    if dataset_config: cfg["dataset_config"] = dataset_config

    if model_version: cfg["model_version"] = model_version
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
    cfg["gradient_accumulation_steps"] = int(gradient_accumulation_steps)
    if float(caption_dropout_rate) > 0: cfg["caption_dropout_rate"] = float(caption_dropout_rate)

    cfg["mixed_precision"] = mixed_precision
    if vae_dtype: cfg["vae_dtype"] = vae_dtype
    if fp8_base: cfg["fp8_base"] = True
    if fp8_scaled: cfg["fp8_scaled"] = True
    if fp8_text_encoder: cfg["fp8_text_encoder"] = True

    if attention_mode and attention_mode != "none":
        cfg[attention_mode] = True

    if int(blocks_to_swap) > 0:
        cfg["blocks_to_swap"] = int(blocks_to_swap)
        if use_pinned_memory_for_block_swap: cfg["use_pinned_memory_for_block_swap"] = True

    if loss_type and loss_type != "mse": cfg["loss_type"] = loss_type
    if loss_type in ("huber", "smooth_l1") and float(huber_delta) != 1.0: cfg["huber_delta"] = float(huber_delta)

    cfg["learning_rate"] = float(lr) if lr else 0.0001
    cfg["lr_scheduler"] = lr_scheduler
    if int(lr_warmup_steps) > 0: cfg["lr_warmup_steps"] = int(lr_warmup_steps)
    if float(str(lr_decay_steps)) > 0: cfg["lr_decay_steps"] = float(str(lr_decay_steps))
    if int(lr_scheduler_num_cycles) > 1: cfg["lr_scheduler_num_cycles"] = int(lr_scheduler_num_cycles)
    if float(lr_scheduler_power) != 1.0: cfg["lr_scheduler_power"] = float(lr_scheduler_power)
    if int(lr_scheduler_timescale) > 0: cfg["lr_scheduler_timescale"] = int(lr_scheduler_timescale)
    if float(lr_scheduler_min_lr_ratio) > 0: cfg["lr_scheduler_min_lr_ratio"] = float(lr_scheduler_min_lr_ratio)

    cfg["timestep_sampling"] = timestep_sampling
    if weighting_scheme and weighting_scheme != "none": cfg["weighting_scheme"] = weighting_scheme
    if float(logit_mean) != 0.0: cfg["logit_mean"] = float(logit_mean)
    if float(logit_std) != 1.0: cfg["logit_std"] = float(logit_std)
    if float(mode_scale) != 1.29: cfg["mode_scale"] = float(mode_scale)
    if int(min_timestep) > 0: cfg["min_timestep"] = int(min_timestep)
    if int(max_timestep) != 1000: cfg["max_timestep"] = int(max_timestep)

    cfg["optimizer_type"] = optimizer_type
    if float(max_grad_norm) != 1.0: cfg["max_grad_norm"] = float(max_grad_norm)

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
    dit_val = args[0]
    if not dit_val or not dit_val.strip():
        return "❌ 保存失败：请先在顶部填写 DiT 模型路径（必填）！"
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
def run_cache_latents(dataset_cfg, dit_path, vae_p, vae_dtype_c, skip_existing, model_ver):
    if not (dit_path and dit_path.strip()) and not (vae_p and vae_p.strip()):
        return "❌ 错误：请先在界面顶部【0. 全局设置】中填写【DiT 权重路径】或【VAE 路径】！"
    
    cmd = [sys.executable, "flux_2_cache_latents.py",
           "--dataset_config", dataset_cfg or "dataset_flux2.toml"]
    if dit_path and dit_path.strip(): cmd += ["--dit", dit_path.strip()]
    if vae_p and vae_p.strip(): cmd += ["--vae", vae_p.strip()]
    if vae_dtype_c: cmd += ["--vae_dtype", vae_dtype_c]
    if skip_existing: cmd += ["--skip_existing"]
    if model_ver: cmd += ["--model_version", model_ver]
    start_subprocess(cmd)
    return "已启动 Cache Latents，请查看调试框。"

def run_cache_te(dataset_cfg, text_enc_path, te_dtype, skip_existing, model_ver, fp8_te):
    if not (text_enc_path and text_enc_path.strip()):
        return "❌ 错误：请先在界面顶部【0. 全局设置】中填写【Text Encoder 路径】！"
        
    cmd = [sys.executable, "flux_2_cache_text_encoder_outputs.py",
           "--dataset_config", dataset_cfg or "dataset_flux2.toml"]
    if text_enc_path and text_enc_path.strip(): cmd += ["--text_encoder", text_enc_path.strip()]
    if te_dtype: cmd += ["--mixed_precision", te_dtype]
    if skip_existing: cmd += ["--skip_existing"]
    if model_ver: cmd += ["--model_version", model_ver]
    if fp8_te: cmd += ["--fp8_text_encoder"]
    start_subprocess(cmd)
    return "已启动 Cache Text Encoder，请查看调试框。"


def generate_dataset_toml(res_w, res_h, cap_ext, b_size, e_bucket, no_upscale, img_dir, cache_dir, control_dir, save_path, config_path):
    try:
        gen_cfg = {
            "resolution": [int(res_w), int(res_h)],
            "caption_extension": cap_ext,
            "batch_size": int(b_size),
            "enable_bucket": bool(e_bucket),
            "bucket_no_upscale": bool(no_upscale)
        }

        ds_cfg = {
            "image_directory": img_dir,
            "cache_directory": cache_dir,
        }
        if control_dir and control_dir.strip():
            ds_cfg["control_images"] = control_dir.strip()

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
            "ds_img_dir": img_dir,
            "ds_cache_dir": cache_dir,
            "ds_control_images_dir": control_dir,
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
    try:
        cfg_check = toml.load(p)
        if not cfg_check.get("dit"):
            return ("❌ 配置文件中缺少 dit！"
                    "请在顶部填写 DiT 模型路径后重新保存配置。")
                    
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
    cmd = [sys.executable, "flux_2_train_network.py", "--config_file", p]
    if cfg_check.get("dit"):
        cmd += ["--dit", cfg_check["dit"]]
    if cfg_check.get("model_version"):
        cmd += ["--model_version", cfg_check["model_version"]]
    start_subprocess(cmd)
    return f"已启动 FLUX.2 训练（配置：{p}），请查看调试框。"


# ══════════════════════════════════════════════════
#  UI 布局
# ══════════════════════════════════════════════════
ui_theme = gr.themes.Soft(
    primary_hue="blue",
    font=[gr.themes.GoogleFont("Inter"), "ui-sans-serif", "system-ui", "sans-serif"]
)

with gr.Blocks(title="FLUX.2 训练控制台", theme=ui_theme) as app:
    gr.Markdown("# 🎨 FLUX.2 图像模型训练中文控制台 (Musubi-Tuner)")
    gr.Markdown("支持 FLUX.2 [dev]、[klein-4b]、[klein-9b] 及其 base 版本的图像生成模型训练")

    # ────────────────────────────────────────────────
    #  顶部全局路径区（所有标签页共用）
    # ────────────────────────────────────────────────
    with gr.Group():
        gr.Markdown("## 📂 全局路径设置（所有步骤共用）")
        with gr.Row():
            g_dit = gr.Textbox(label="★ DiT 模型路径 (--dit，必填)",
                                 placeholder="path/to/flux2-dev.safetensors", scale=3)
            g_model_ver = gr.Dropdown(label="模型版本 (--model_version)",
                                        choices=["dev", "klein-4b", "klein-base-4b", "klein-9b", "klein-base-9b"],
                                        value="dev", scale=1)
        with gr.Row():
            g_vae = gr.Textbox(label="★ VAE 路径 (--vae，必填)",
                               placeholder="path/to/ae.safetensors", scale=2)
            g_text_encoder = gr.Textbox(label="★ Text Encoder 路径 (--text_encoder，必填)",
                                      placeholder="path/to/00001-of-00010.safetensors", scale=2)
        with gr.Row():
            g_dataset = gr.Textbox(label="数据集配置文件 (--dataset_config)",
                                   value="dataset_flux2.toml", scale=2)
            g_output_dir = gr.Textbox(label="输出目录 (--output_dir)", value="./outputs", scale=2)
            g_output_name = gr.Textbox(label="输出模型名称 (--output_name)", value="flux2_lora", scale=2)
        with gr.Row():
            g_config_path = gr.Textbox(label="训练配置文件路径（保存/读取）",
                                       value="flux2_train_config.toml", scale=3)

        with gr.Row():
            load_btn = gr.Button("📂 载入配置文件", size="sm")
            save_top_btn = gr.Button("💾 快速保存配置", variant="primary", size="sm")
            reset_btn = gr.Button("↺ 恢复默认参数", size="sm")
            load_status = gr.Textbox(label="", interactive=False, scale=3)

    gr.Markdown("---")

    with gr.Tabs():

        # ══════════════════════════════════
        #  Tab 1: 数据集准备
        # ══════════════════════════════════
        with gr.Tab("🗂️ 1. 数据集准备 (Dataset & Cache)"):

            with gr.Accordion("📝 1.1 生成数据集配置文件 (TOML)", open=True):
                gr.Markdown("FLUX.2 是图像生成模型，请配置图像数据集。支持单图训练和多图训练，control_images 用于参考图像（编辑/参考生成任务）。")
                with gr.Row():
                    ds_res_w = gr.Number(label="Resolution Width", value=1024, precision=0)
                    ds_res_h = gr.Number(label="Resolution Height", value=1024, precision=0)
                    ds_cap_ext = gr.Textbox(label="Caption Extension", value=".txt")
                with gr.Row():
                    ds_batch_size = gr.Number(label="Batch Size", value=1, precision=0)
                    ds_enable_bucket = gr.Checkbox(label="Enable Bucket", value=True)
                    ds_bucket_no_upscale = gr.Checkbox(label="Bucket No Upscale", value=False)
                with gr.Row():
                    ds_img_dir = gr.Textbox(label="Image Directory (图像文件夹)", value="", placeholder="path/to/images")
                    ds_cache_dir = gr.Textbox(label="Cache Directory (缓存文件夹)", value="", placeholder="path/to/cache")
                with gr.Row():
                    ds_control_dir = gr.Textbox(label="Control Images Dir (参考图像文件夹，可选)", 
                                                  value="", placeholder="path/to/control_images")
                with gr.Row():
                    ds_save_path = gr.Textbox(label="保存路径", value="dataset_flux2.toml")
                    ds_gen_btn = gr.Button("📄 生成数据集配置", variant="primary")
                ds_gen_status = gr.Textbox(label="状态", interactive=False)
                ds_gen_btn.click(
                    fn=generate_dataset_toml,
                    inputs=[ds_res_w, ds_res_h, ds_cap_ext, ds_batch_size, ds_enable_bucket, ds_bucket_no_upscale,
                            ds_img_dir, ds_cache_dir, ds_control_dir, ds_save_path, g_config_path],
                    outputs=[ds_gen_status, g_dataset]
                )
                
            with gr.Accordion("🗂️ 1.2 预缓存 Latent 与文本特征", open=True):
                gr.Markdown("""
                预缓存图像 Latent 和文本编码器输出，大幅加速训练。  
            **模型路径和数据集配置文件均读取自顶部全局设置，无需重复填写。**
            """)
                with gr.Row():
                    with gr.Column():
                        gr.Markdown("#### Cache Latents（编码图像特征）")
                        with gr.Row():
                            c_vae_dtype = gr.Dropdown(label="VAE Dtype", 
                                                        choices=["", "float32", "bfloat16"], 
                                                        value="bfloat16",
                                                        info="默认float32，bfloat16可减少显存")
                            c_skip_lc = gr.Checkbox(label="跳过已缓存文件 (--skip_existing)", value=True)
                        cache_latents_btn = gr.Button("▶ 运行 Cache Latents", variant="primary")

                    with gr.Column():
                        gr.Markdown("#### Cache Text Encoder（编码文本特征）")
                        te_dtype = gr.Dropdown(label="Text Encoder Dtype",
                            choices=["", "fp16", "bf16", "fp32"], value="bf16")
                        te_skip = gr.Checkbox(label="跳过已缓存文件 (--skip_existing)", value=True)
                        te_fp8 = gr.Checkbox(label="FP8 Text Encoder (--fp8_text_encoder)", 
                                             value=False,
                                             info="dev(Mistral 3)不支持此选项")
                        cache_te_btn = gr.Button("▶ 运行 Cache Text Encoder", variant="primary")

                cache_status = gr.Textbox(label="操作状态", interactive=False)

                cache_latents_btn.click(
                    fn=run_cache_latents,
                    inputs=[g_dataset, g_dit, g_vae, c_vae_dtype, c_skip_lc, g_model_ver],
                    outputs=cache_status
                )
                cache_te_btn.click(
                    fn=run_cache_te,
                    inputs=[g_dataset, g_text_encoder, te_dtype, te_skip, g_model_ver, te_fp8],
                    outputs=cache_status
                )
                
        # ══════════════════════════════════
        #  Tab 2: 训练参数配置
        # ══════════════════════════════════
        with gr.Tab("⚙️ 2. 训练参数配置"):

            with gr.Accordion("🎨 模型与 LoRA 设置", open=True):
                with gr.Row():
                    network_module = gr.Dropdown(label="网络模块 (--network_module)",
                        choices=["networks.lora_flux_2"],
                        value="networks.lora_flux_2",
                        info="FLUX.2 必须使用 networks.lora_flux_2")
                    network_dim = gr.Number(label="Network Dim / Rank", value=32, precision=0)
                    network_alpha = gr.Number(label="Network Alpha", value=16)
                with gr.Row():
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
                    max_train_epochs = gr.Number(label="最大 Epoch 数", value=16, precision=0)
                    max_train_steps = gr.Textbox(label="最大训练步数 (空=不限)", value="")
                    batch_size = gr.Number(label="批次大小", value=1, precision=0)
                    seed = gr.Number(label="随机种子", value=42, precision=0)
                with gr.Row():
                    gradient_checkpointing = gr.Checkbox(label="梯度检查点", value=True)
                    gradient_accumulation_steps = gr.Number(label="梯度累积步数", value=1, precision=0)
                    caption_dropout_rate = gr.Number(label="Caption Dropout Rate", value=0.0)

            with gr.Accordion("⚡ 精度与内存优化", open=False):
                gr.Markdown("""
                **内存优化推荐设置：**
                - 基础优化：`--fp8_base --fp8_scaled` (DiT) + `--fp8_text_encoder` (Text Encoder, 除dev外)
                - 进一步：`--blocks_to_swap` (不同模型最大值: dev=29, klein-4b=13, klein-9b=16)
                - 极限：`--gradient_checkpointing_cpu_offload`
                """)
                with gr.Row():
                    mixed_precision = gr.Dropdown(label="混合精度",
                        choices=["bf16", "fp16", "no"], value="bf16",
                        info="FLUX.2 推荐使用 bf16")
                    vae_dtype = gr.Dropdown(label="VAE Dtype",
                        choices=["", "float32", "bfloat16"], value="")
                with gr.Row():
                    fp8_base = gr.Checkbox(label="FP8 Base (--fp8_base, DiT)", value=True)
                    fp8_scaled = gr.Checkbox(label="FP8 Scaled (--fp8_scaled, DiT)", value=True,
                                             info="使用fp8_base时建议同时启用")
                    fp8_text_encoder = gr.Checkbox(label="FP8 Text Encoder (--fp8_text_encoder)", value=False,
                                                   info="dev(Mistral 3)不支持")
                with gr.Row():
                    attention_mode = gr.Dropdown(label="注意力机制",
                        choices=["sdpa", "flash_attn", "xformers"], value="sdpa")
                    blocks_to_swap = gr.Number(label="Blocks to Swap", value=0, precision=0,
                                               info="dev最大29, klein-4b最大13, klein-9b最大16")
                    use_pinned_memory_for_block_swap = gr.Checkbox(label="Pinned Memory", value=True)
                with gr.Row():
                    gradient_checkpointing_cpu_offload = gr.Checkbox(label="梯度检查点 CPU Offload", value=False)

            with gr.Accordion("📊 Loss 设置", open=False):
                with gr.Row():
                    loss_type = gr.Dropdown(label="Loss 类型",
                        choices=["mse","mae","l1","huber","smooth_l1"], value="mse")
                    huber_delta = gr.Number(label="Huber Delta", value=1.0)

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
                        value="flux2_shift",
                        info="FLUX.2 推荐使用 flux2_shift")
                with gr.Row():
                    weighting_scheme = gr.Dropdown(label="加权方案",
                        choices=["none","sigma_sqrt","logit_normal","mode","cosmap"], 
                        value="none")
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

            with gr.Accordion("💾 保存与追踪监控", open=False):
                with gr.Row():
                    save_every_n_epochs = gr.Number(label="每N Epoch保存", value=1, precision=0)
                    save_every_n_steps = gr.Textbox(label="每N步保存 (空=用Epoch)", value="")
                    save_last_n_epochs = gr.Textbox(label="只保留最后N轮", value="")
                    save_last_n_steps = gr.Textbox(label="只保留最后N步", value="")
                with gr.Row():
                    save_state = gr.Checkbox(label="保存训练状态", value=False)
                    save_state_on_train_end = gr.Checkbox(label="仅结束时保存状态", value=False)
                    log_with = gr.Dropdown(label="日志监控 (--log_with)", 
                                           choices=["", "tensorboard", "wandb", "all"], 
                                           value="tensorboard")
                    logging_dir = gr.Textbox(label="日志路径 (--logging_dir)", value="./logs")

            with gr.Accordion("🔀 数据加载与杂项", open=False):
                with gr.Row():
                    max_data_loader_n_workers = gr.Number(label="数据加载线程数", value=8, precision=0)
                    persistent_data_loader_workers = gr.Checkbox(label="持久化工人", value=True)
                with gr.Row():
                    cuda_allow_tf32 = gr.Checkbox(label="cuda_allow_tf32", value=True)
                    cuda_cudnn_benchmark = gr.Checkbox(label="cuda_cudnn_benchmark", value=True)
                with gr.Row():
                    training_comment = gr.Textbox(label="训练备注", value="")
                    metadata_title = gr.Textbox(label="Metadata Title", value="")
                    metadata_author = gr.Textbox(label="Metadata Author", value="")
                    metadata_description = gr.Textbox(label="Metadata Description", value="")

            with gr.Row():
                save_btn = gr.Button("💾 保存配置", variant="primary", size="lg")
                save_status_tab = gr.Textbox(label="状态", interactive=False, scale=3)

        # ══════════════════════════════════
        #  Tab 3: 训练
        # ══════════════════════════════════
        with gr.Tab("🚀 3. 训练及进度显示"):
            gr.Markdown("点击【启动训练】后，将依据顶部配置文件路径所指定的 TOML 文件启动训练。")
            with gr.Row():
                train_btn = gr.Button("🚀 一键启动 FLUX.2 训练", variant="primary", size="lg")
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
        g_dit, g_vae, g_text_encoder, g_output_dir, g_output_name, g_dataset,
        g_model_ver, network_module,
        network_dim, network_alpha, network_dropout, scale_weight_norms,
        network_weights, dim_from_weights,
        enable_lora_plus, loraplus_lr_ratio,
        enable_blocks, exclude_patterns, include_patterns,
        max_train_epochs, max_train_steps, batch_size, seed,
        gradient_checkpointing, gradient_checkpointing_cpu_offload, gradient_accumulation_steps,
        caption_dropout_rate,
        mixed_precision, vae_dtype, 
        fp8_base, fp8_scaled, fp8_text_encoder,
        attention_mode, blocks_to_swap, use_pinned_memory_for_block_swap,
        loss_type, huber_delta,
        lr, lr_scheduler, lr_warmup_steps, lr_decay_steps,
        lr_scheduler_num_cycles, lr_scheduler_power,
        lr_scheduler_timescale, lr_scheduler_min_lr_ratio,
        timestep_sampling, weighting_scheme, logit_mean, logit_std, mode_scale,
        min_timestep, max_timestep,
        optimizer_type, max_grad_norm,
        save_every_n_epochs, save_every_n_steps,
        save_last_n_epochs, save_last_n_steps,
        save_state, save_state_on_train_end, resume,
        log_with, logging_dir,
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
        ds_img_dir, ds_cache_dir, ds_control_dir, ds_save_path,
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
    app.launch(server_name="127.0.0.1", server_port=7861, inbrowser=True)
